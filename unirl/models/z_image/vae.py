"""ZImageVAEDecodeStage — LatentSegment → Images via VAE decode."""

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
        """Decode the final-step latents in *s* into pixel images."""
        if self.bundle.vae is None:
            raise RuntimeError(
                "ZImageVAEDecodeStage.decode: no VAE loaded (load_vae=False). "
                "The trainer-side pipeline cannot decode latents in this "
                "configuration — separate-engine recipes decode in the "
                "rollout engine; trainside rollout requires load_vae=True."
            )
        if s.latents is None:
            raise ValueError("ZImageVAEDecodeStage.decode: segment.latents is None")
        if s.latents.ndim < 5:
            raise ValueError(
                f"ZImageVAEDecodeStage.decode: expected latents shape [N, K, C, H, W], got {tuple(s.latents.shape)}"
            )
        clean = s.latents[:, -1]

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
        return Images.from_dense(pixels)


__all__ = ["ZImageVAEDecodeStage"]
