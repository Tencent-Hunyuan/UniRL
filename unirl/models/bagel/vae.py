"""BagelVAEDecodeStage — LatentSegment → Images via unpatchify + VAE decode.

Implements ``DecodeStage[LatentSegment, Images]``. Bagel differs from SD3: its
trajectory latents are stored **packed** (navit) as ``[N, seq, p²·z]`` (seq =
h·w image tokens, p = ``latent_patch_size``, z = ``latent_channel``), not as a
spatial ``[N, C, H, W]`` tensor. So decode first **unpatchifies** the final clean
latent (``segment.latents[:, -1]``) back to spatial ``[N, z, h·p, w·p]`` and then
runs the FLUX-style autoencoder.

Math mirrors the vendored ``InterleaveInferencer.decode_image``
(``vendor/inferencer.py``): ``reshape(N, h, w, p, p, z)`` →
``einsum('nhwpqc->nchpwq')`` → ``reshape(N, z, h·p, w·p)`` → ``vae.decode`` →
``*0.5 + 0.5`` clamp. The Bagel ``AutoEncoder.decode`` applies its own
scale/shift internally, so (unlike SD3) no external scaling_factor is needed.

``h, w`` come from the generation ``image_shape`` (height, width); the packed
seq alone is ambiguous for non-square images, so ``decode`` takes an optional
``image_shape`` (the pipeline passes ``conditions.image_shape``). When omitted it
assumes a square grid (``h = w = isqrt(seq)``) and raises if seq isn't a perfect
square.
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
    """Token grid ``(h, w)`` for an ``(H, W)`` image: ``h = H // latent_downsample``.

    ``latent_downsample`` (=16 for BAGEL: VAE /8 × patch 2) folds both the VAE
    spatial downsample and the latent patchify into one factor.
    """
    H, W = int(image_shape[0]), int(image_shape[1])
    return H // int(latent_downsample), W // int(latent_downsample)


def bagel_latent_shape(
    image_shape: Tuple[int, int],
    *,
    latent_downsample: int,
    latent_patch_size: int,
    latent_channels: int,
) -> Tuple[int, int]:
    """Packed per-sample noise shape ``(seq, p²·z)`` for an ``(H, W)`` image.

    Bagel's ``x_T`` is packed ``[h·w, p²·z]`` (the ``packed_init_noises`` shape
    ``generate_image`` consumes), NOT spatial ``[C, H, W]``. Provided for
    driver-side noise bookkeeping / pipeline ``latent_shape`` parity; note the
    trainside sampler currently draws ``x_T`` inside ``generate_image`` itself.
    """
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
    """Unpatchify packed ``[N, h·w, p²·z]`` → spatial ``[N, z, h·p, w·p]``.

    Exact inverse of the vendored patchify (``forward_cache_update_vae``); mirrors
    ``decode_image``'s ``reshape → einsum('nhwpqc->nchpwq') → reshape``.
    """
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
        """Decode the final clean latent in *s* into ``[N, 3, H, W]`` pixels in ``[0, 1]``.

        Reads ``s.latents[:, -1]`` — the final clean latent ``diffuse`` stores
        (packed ``[N, seq, p²·z]``). ``image_shape`` (height, width) fixes the
        token grid; omitted ⇒ square grid from ``isqrt(seq)``.
        """
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
            vae = self.bundle.vae
            orig_dtype = next(vae.parameters()).dtype
            decoded = vae.to(torch.float32).decode(spatial)
            # Restore the VAE's loaded dtype: the image-edit path also ENCODES the
            # source with this shared VAE on the next rollout, and a left-over fp32
            # cast would make encode emit fp32 latents that mismatch the bf16 vae2llm.
            if orig_dtype != torch.float32:
                vae.to(orig_dtype)
        pixels = (decoded * 0.5 + 0.5).clamp(0.0, 1.0)
        return Images(pixels=pixels)


__all__ = ["BagelVAEDecodeStage", "bagel_latent_geometry", "bagel_latent_shape", "unpatchify_latent"]
