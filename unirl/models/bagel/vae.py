"""BagelVAEDecodeStage — LatentSegment → Images via unpatchify + VAE decode.

Bagel's trajectory latents are stored packed (navit) as ``[N, seq, p²·z]``, so decode
unpatchifies the final clean latent to spatial ``[N, z, h·p, w·p]`` then runs the
FLUX-style autoencoder (which applies its own scale/shift, so no external factor).
"""

from __future__ import annotations

from math import isqrt
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from unirl.models.types.codec import DecodeStage
from unirl.types.primitives import Images
from unirl.types.segments.latent import LatentSegment

if TYPE_CHECKING:
    from .bundle import BagelBundle


def bagel_latent_geometry(
    image_shape: Tuple[int, int],
    *,
    latent_downsample: int,
) -> Tuple[int, int]:
    """Token grid ``(h, w)`` for an ``(H, W)`` image (``latent_downsample``=16 for
    BAGEL folds VAE /8 × patch 2)."""
    H, W = int(image_shape[0]), int(image_shape[1])
    return H // int(latent_downsample), W // int(latent_downsample)


def bagel_latent_shape(
    image_shape: Tuple[int, int],
    *,
    latent_downsample: int,
    latent_patch_size: int,
    latent_channels: int,
) -> Tuple[int, int]:
    """Packed per-sample noise shape ``(seq, p²·z)`` for an ``(H, W)`` image
    (Bagel's x_T is packed ``[h·w, p²·z]``, not spatial ``[C, H, W]``)."""
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
    """Unpatchify packed ``[N, h·w, p²·z]`` → spatial ``[N, z, h·p, w·p]`` (inverse of
    the vendored patchify; mirrors ``decode_image``)."""
    n = int(packed.shape[0])
    p, z = int(patch_size), int(latent_channels)
    x = packed.reshape(n, h, w, p, p, z)
    x = torch.einsum("nhwpqc->nchpwq", x)
    return x.reshape(n, z, h * p, w * p)


class BagelVAEDecodeStage(DecodeStage[LatentSegment, Images]):
    """BAGEL VAE decode: unpatchify final packed latent then decode to pixels."""

    def __init__(self, bundle: "BagelBundle") -> None:
        self.bundle = bundle

    def decode(self, s: LatentSegment, *, image_shape: Optional[Tuple[int, int]] = None) -> Images:
        """Decode the final clean latent (``s.latents[:, -1]``) into ``[N, 3, H, W]``
        pixels in ``[0, 1]``; ``image_shape`` fixes the grid (else square from isqrt)."""
        if s.latents is None:
            raise ValueError("BagelVAEDecodeStage.decode: segment.latents is None")
        if s.latents.ndim != 4:
            raise ValueError(
                f"BagelVAEDecodeStage.decode: expected packed latents [N, K, seq, C], got {tuple(s.latents.shape)}"
            )
        clean = s.latents[:, -1]  # [N, seq, p²·z]
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

        spatial = unpatchify_latent(clean.float(), h=h, w=w, patch_size=p, latent_channels=z)
        with torch.no_grad():
            decoded = self.bundle.vae.to(torch.float32).decode(spatial)
        pixels = (decoded * 0.5 + 0.5).clamp(0.0, 1.0)
        return Images(pixels=pixels)


__all__ = ["BagelVAEDecodeStage", "bagel_latent_geometry", "bagel_latent_shape", "unpatchify_latent"]
