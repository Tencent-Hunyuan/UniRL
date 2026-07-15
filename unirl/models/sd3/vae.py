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

from contextlib import nullcontext

import torch

from unirl.models.types.codec import DecodeStage
from unirl.types.primitives import Images
from unirl.types.segments import LatentSegment

from .bundle import SD3Bundle


class SD3VAEDecodeStage(DecodeStage[LatentSegment, Images]):
    """SD3 VAE decode stage."""

    def __init__(self, bundle: SD3Bundle) -> None:
        self.bundle = bundle

    def decode(self, s: LatentSegment, *, grad: bool = False, activation_checkpoint: bool = False) -> Images:
        """Decode the final-step latents in *s* into pixel images.

        Reads ``s.latents[:, -1]`` (the final stored position, which is ``T`` —
        the clean latent ``x_0``). VAE forward runs in fp32; output is clamped to
        ``[0, 1]`` before being wrapped in ``Images``.

        ``grad=False`` (default) keeps the rollout path under ``torch.no_grad()``.
        ``grad=True`` (ReFL direct-reward backprop) runs the decode WITH grad so it
        flows from the reward through the frozen VAE into ``clean``; the VAE has no
        trainable params, so only ``clean``'s graph is extended. ``activation_checkpoint``
        (grad only) recomputes the decode in backward to trade compute for memory.
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
        vae = self.bundle.vae.to(torch.float32)

        def _decode(lat: torch.Tensor) -> torch.Tensor:
            latents_f32 = lat.to(dtype=torch.float32) / scaling_factor
            if shift_factor is not None:
                latents_f32 = latents_f32 + float(shift_factor)
            return vae.decode(latents_f32).sample

        with nullcontext() if grad else torch.no_grad():
            if grad and activation_checkpoint and clean.requires_grad:
                from torch.utils.checkpoint import checkpoint

                decoded = checkpoint(_decode, clean, use_reentrant=False)
            else:
                decoded = _decode(clean)
        pixels = ((decoded + 1.0) / 2.0).clamp(0.0, 1.0)
        return Images(pixels=pixels)


__all__ = ["SD3VAEDecodeStage"]
