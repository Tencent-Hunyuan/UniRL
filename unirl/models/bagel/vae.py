"""BagelVAEDecodeStage — LatentSegment → Images via unpatchify + VAE decode."""

from __future__ import annotations

from contextlib import nullcontext
from math import isqrt
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from unirl.models.types.codec import DecodeStage
from unirl.types.primitives import Images
from unirl.types.segments.latent import LatentSegment

if TYPE_CHECKING:
    from unirl.types.conditions.image import ImageLatentCondition

    from .bundle import BagelBundle


def bagel_latent_geometry(
    image_shape: Tuple[int, int],
    *,
    latent_downsample: int,
) -> Tuple[int, int]:
    """Token grid ``(h, w)`` for an ``(H, W)`` image: ``h = H // latent_downsample``."""
    H, W = int(image_shape[0]), int(image_shape[1])
    return H // int(latent_downsample), W // int(latent_downsample)


def bagel_latent_shape(
    image_shape: Tuple[int, int],
    *,
    latent_downsample: int,
    latent_patch_size: int,
    latent_channels: int,
) -> Tuple[int, int]:
    """Packed per-sample noise shape ``(seq, p²·z)`` for an ``(H, W)`` image."""
    h, w = bagel_latent_geometry(image_shape, latent_downsample=latent_downsample)
    return h * w, int(latent_patch_size) ** 2 * int(latent_channels)


def unpatchify_latent(
    packed: torch.Tensor,
    *,
    h: int,
    w: int,
    patch_size: int,
    latent_channels: int,
) -> torch.Tensor:
    """Unpatchify packed ``[N, h·w, p²·z]`` → spatial ``[N, z, h·p, w·p]``."""
    n = int(packed.shape[0])
    p, z = int(patch_size), int(latent_channels)
    x = packed.reshape(n, h, w, p, p, z)
    x = torch.einsum("nhwpqc->nchpwq", x)
    return x.reshape(n, z, h * p, w * p)


class BagelVAEDecodeStage(DecodeStage[LatentSegment, Images]):
    """BAGEL VAE decode: unpatchify final packed latent then decode to pixels."""

    def __init__(self, bundle: "BagelBundle", *, decode_batch_size: int = 4) -> None:
        self.bundle = bundle
        # Decode VAE batches in chunks to bound fp32 convolution memory.
        self.decode_batch_size = max(1, int(decode_batch_size))

    def decode(
        self,
        s: LatentSegment,
        *,
        image_shape: Optional[Tuple[int, int]] = None,
        grad: bool = False,
        activation_checkpoint: bool = False,
    ) -> Images:
        """Decode the final clean latent in *s* into ``[N, 3, H, W]`` pixels in ``[0, 1]``."""
        if s.latents is None:
            raise ValueError("BagelVAEDecodeStage.decode: segment.latents is None")
        if s.latents.ndim != 4:
            raise ValueError(
                f"BagelVAEDecodeStage.decode: expected packed latents [N, K, seq, C], got {tuple(s.latents.shape)}"
            )
        clean = s.latents[:, -1]
        n, seq, _ = clean.shape

        p = int(self.bundle.latent_patch_size)
        z = int(self.bundle.latent_channels)
        if image_shape is not None:
            h, w = bagel_latent_geometry(image_shape, latent_downsample=int(self.bundle.latent_downsample))
        else:
            side = isqrt(seq)
            if side * side != seq:
                raise ValueError(
                    f"BagelVAEDecodeStage.decode: seq={seq} is not a perfect square; "
                    f"pass image_shape=(H, W) for non-square latents."
                )
            h = w = side
        if h * w != seq:
            raise ValueError(f"BagelVAEDecodeStage.decode: image_shape grid h*w={h * w} != packed seq={seq}.")

        def _decode(lat: torch.Tensor) -> torch.Tensor:
            spatial = unpatchify_latent(lat.float(), h=h, w=w, patch_size=p, latent_channels=z)
            vae_fp32 = self.bundle.vae.to(torch.float32)
            bs = self.decode_batch_size
            if n <= bs:
                return vae_fp32.decode(spatial)
            return torch.cat([vae_fp32.decode(spatial[i : i + bs]) for i in range(0, n, bs)], dim=0)

        with nullcontext() if grad else torch.no_grad():
            if grad and activation_checkpoint and clean.requires_grad:
                from torch.utils.checkpoint import checkpoint

                decoded = checkpoint(_decode, clean, use_reentrant=False)
            else:
                decoded = _decode(clean)
        pixels = (decoded * 0.5 + 0.5).clamp(0.0, 1.0)
        # Return decoded pixels on CPU to avoid driver-side GPU gathers.
        return Images.from_dense(pixels.cpu())


def patchify_latent(
    spatial: torch.Tensor,
    *,
    h: int,
    w: int,
    patch_size: int,
    latent_channels: int,
) -> torch.Tensor:
    """Spatial ``[N, z, h·p, w·p]`` → packed ``[N, h·w, p²·z]`` (inverse of"""
    p, z = patch_size, latent_channels
    n = spatial.shape[0]
    cropped = spatial[:, :, : h * p, : w * p]
    blocks = cropped.reshape(n, z, h, p, w, p)
    return torch.einsum("nchpwq->nhwpqc", blocks).reshape(n, h * w, p * p * z)


class BagelVAEEncodeStage:
    """Images → packed clean latents ``[B, h·w, p²·z]`` — inverse of the decode stage."""

    def __init__(self, bundle: "BagelBundle") -> None:
        self.bundle = bundle

    @torch.no_grad()
    def encode(self, p: Images) -> "ImageLatentCondition":
        from unirl.types.conditions.image import ImageLatentCondition

        try:
            pixels = p.to_dense()
        except ValueError as exc:
            raise ValueError(
                f"BagelVAEEncodeStage.encode requires a non-empty batch with uniform image shapes; {exc}"
            ) from exc
        if not isinstance(pixels, torch.Tensor) or pixels.ndim != 4 or pixels.shape[1] != 3:
            raise ValueError(
                f"BagelVAEEncodeStage.encode: expected pixels [B, 3, H, W], got "
                f"{tuple(pixels.shape) if isinstance(pixels, torch.Tensor) else type(pixels).__name__}"
            )
        height, width = pixels.shape[2], pixels.shape[3]
        down = self.bundle.latent_downsample
        if height % down or width % down:
            raise ValueError(
                f"BagelVAEEncodeStage.encode: image {height}x{width} must be divisible by latent_downsample={down}."
            )
        h, w = bagel_latent_geometry((height, width), latent_downsample=down)
        vae = self.bundle.vae
        reg = getattr(vae, "reg", None)
        if reg is None or not hasattr(reg, "sample"):
            raise RuntimeError("BagelVAEEncodeStage: bundle.vae has no .reg gaussian — vendored API changed?")
        x = (pixels.to(device=self.bundle.device, dtype=self.bundle.vae_dtype) * 2.0 - 1.0).contiguous()
        prev_sample = reg.sample
        reg.sample = False  # deterministic posterior mean, not a draw
        try:
            spatial = vae.encode(x)
        finally:
            reg.sample = prev_sample
        packed = patchify_latent(
            spatial.float(),
            h=h,
            w=w,
            patch_size=self.bundle.latent_patch_size,
            latent_channels=self.bundle.latent_channels,
        )
        return ImageLatentCondition(latents=packed)


__all__ = [
    "BagelVAEDecodeStage",
    "BagelVAEEncodeStage",
    "bagel_latent_geometry",
    "bagel_latent_shape",
    "patchify_latent",
    "unpatchify_latent",
]
