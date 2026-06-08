"""SD3VAEDecodeStage — LatentSegment → Images via VAE decode.

Implements ``DecodeStage[LatentSegment, Images]``. Reads the final stored
position from ``LatentSegment.latents[:, -1]`` (``SD3DiffusionStage`` always
stores position ``T``, the clean latent), runs VAE decode in fp32 (bf16 is
unsupported by most VAE implementations), and normalizes the output from
``[-1, 1]`` to ``[0, 1]`` before wrapping in ``Images``.

No ``SD3VAEEncodeStage`` here — legacy SD3 supports only text-to-image
(``models/sd3.py:492-495``); the encoder is unused. Add when img2img /
SDEdit / ControlNet lands.

Decode math copied from ``models/sd3.py:533-550`` and
``samplers/fsdp/base_sampler.py:162-180`` (do NOT import legacy code).
"""

from __future__ import annotations

import torch

from unirl.models.types.codec import DecodeStage
from unirl.types.primitives import Images
from unirl.types.segments import LatentSegment

from .bundle import SD3Bundle


class SD3VAEDecodeStage(DecodeStage[LatentSegment, Images]):
    """SD3 VAE decode stage."""

    def __init__(self, bundle: SD3Bundle) -> None:
        self.bundle = bundle

    def decode(self, s: LatentSegment) -> Images:
        """Decode the final-step latents in *s* into pixel images.

        Reads ``s.latents[:, -1]`` (the final stored position, which is
        ``T`` — the clean latent ``x_0``). VAE forward runs in fp32; output
        is clamped to ``[0, 1]`` before being wrapped in ``Images``.
        """
        if s.latents is None:
            raise ValueError("SD3VAEDecodeStage.decode: segment.latents is None")
        if s.latents.ndim < 5:
            raise ValueError(
                f"SD3VAEDecodeStage.decode: expected latents shape [N, K, C, H, W], got {tuple(s.latents.shape)}"
            )
        clean = s.latents[:, -1]
        scaling_factor = self.bundle.vae.config.scaling_factor
        shift_factor = getattr(self.bundle.vae.config, "shift_factor", None)
        with torch.no_grad():
            latents_f32 = clean.to(dtype=torch.float32) / scaling_factor
            if shift_factor is not None:
                latents_f32 = latents_f32 + float(shift_factor)
            decoded = self.bundle.vae.to(torch.float32).decode(latents_f32).sample
        pixels = ((decoded + 1.0) / 2.0).clamp(0.0, 1.0)
        return Images(pixels=pixels)


__all__ = ["SD3VAEDecodeStage"]
