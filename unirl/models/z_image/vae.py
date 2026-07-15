"""ZImageVAEDecodeStage — LatentSegment → Images via VAE decode.

Implements ``DecodeStage[LatentSegment, Images]``. Reads the final stored
position from ``LatentSegment.latents[:, -1]`` (``ZImageDiffusionStage``
always stores position ``T``, the clean latent), runs the
``AutoencoderKL`` decode in fp32, and normalizes the output from
``[-1, 1]`` to ``[0, 1]`` before wrapping in ``Images``.

Z-Image uses the flux-style 16-channel ``AutoencoderKL`` with both
``scaling_factor`` and ``shift_factor`` — identical to SD3. The latent
un-normalization mirrors the diffusers ``ZImagePipeline`` decode path:
``x = latent / scaling_factor + shift_factor``.

No ``ZImageVAEEncodeStage`` here — the reference pipeline supports only
text-to-image; the encoder is unused. Add when img2img / SDEdit /
ControlNet lands.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch

from unirl.models.types.codec import DecodeStage
from unirl.types.primitives import Images
from unirl.types.segments import LatentSegment

from .bundle import ZImageBundle


class ZImageVAEDecodeStage(DecodeStage[LatentSegment, Images]):
    """Z-Image VAE decode stage."""

    def __init__(self, bundle: ZImageBundle) -> None:
        self.bundle = bundle

    def decode(self, s: LatentSegment, *, grad: bool = False, activation_checkpoint: bool = False) -> Images:
        """Decode the final-step latents in *s* into pixel images.

        Reads ``s.latents[:, -1]`` (the final stored position, which is
        ``T`` — the clean latent ``x_0`` in spatial shape
        ``[B, C, H, W]``). VAE forward runs in fp32; output is clamped to
        ``[0, 1]`` before being wrapped in ``Images``.

        ``grad=False`` (default) keeps the rollout path under ``torch.no_grad()``.
        ``grad=True`` (ReFL direct-reward backprop) runs the decode WITH grad so it
        flows from the reward through the frozen VAE into ``clean``; the VAE has no
        trainable params, so only ``clean``'s graph is extended. ``activation_checkpoint``
        (grad only) recomputes the decode in backward to trade compute for memory.
        """
        if s.latents is None:
            raise ValueError("ZImageVAEDecodeStage.decode: segment.latents is None")
        if s.latents.ndim < 5:
            raise ValueError(
                f"ZImageVAEDecodeStage.decode: expected latents shape [N, K, C, H, W], got {tuple(s.latents.shape)}"
            )
        clean = s.latents[:, -1]  # [B, C, H, W]

        scaling_factor = self.bundle.vae.config.scaling_factor
        shift_factor = getattr(self.bundle.vae.config, "shift_factor", None)

        def _decode(lat: torch.Tensor) -> torch.Tensor:
            latents_f32 = lat.to(dtype=torch.float32) / scaling_factor
            if shift_factor is not None:
                latents_f32 = latents_f32 + float(shift_factor)
            return self.bundle.vae.to(torch.float32).decode(latents_f32).sample

        with nullcontext() if grad else torch.no_grad():
            if grad and activation_checkpoint and clean.requires_grad:
                from torch.utils.checkpoint import checkpoint

                decoded = checkpoint(_decode, clean, use_reentrant=False)
            else:
                decoded = _decode(clean)
        pixels = ((decoded + 1.0) / 2.0).clamp(0.0, 1.0)
        return Images(pixels=pixels)


__all__ = ["ZImageVAEDecodeStage"]
