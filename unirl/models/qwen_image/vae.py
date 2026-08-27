"""QwenImageVAEDecodeStage — LatentSegment → Images via VAE decode."""

from __future__ import annotations

from contextlib import nullcontext

import torch

from unirl.models.types.codec import DecodeStage
from unirl.types.primitives import Images
from unirl.types.segments import LatentSegment

from .bundle import QwenImageBundle


class QwenImageVAEDecodeStage(DecodeStage[LatentSegment, Images]):
    """Qwen-Image VAE decode stage."""

    def __init__(self, bundle: QwenImageBundle) -> None:
        self.bundle = bundle

    def decode(self, s: LatentSegment, *, grad: bool = False, activation_checkpoint: bool = False) -> Images:
        """Decode the final-step latents in *s* into pixel images."""
        if self.bundle.vae is None:
            raise RuntimeError(
                "QwenImageVAEDecodeStage.decode: no VAE loaded "
                "(load_vae=False). The trainer-side pipeline cannot decode "
                "latents in this configuration — separate-engine recipes "
                "decode in the rollout engine; trainside rollout requires "
                "load_vae=True."
            )
        if s.latents is None:
            raise ValueError("QwenImageVAEDecodeStage.decode: segment.latents is None")
        if s.latents.ndim < 5:
            raise ValueError(
                f"QwenImageVAEDecodeStage.decode: expected latents shape [N, K, C, H, W], got {tuple(s.latents.shape)}"
            )
        clean = s.latents[:, -1]

        vae = self.bundle.vae
        z_dim = int(vae.config.z_dim)
        device = clean.device

        def _decode(lat: torch.Tensor) -> torch.Tensor:
            latents_f32 = lat.to(dtype=torch.float32)
            latents_5d = latents_f32.unsqueeze(2)
            latents_mean = torch.tensor(vae.config.latents_mean, device=device, dtype=torch.float32).view(
                1, z_dim, 1, 1, 1
            )
            latents_std = torch.tensor(vae.config.latents_std, device=device, dtype=torch.float32).view(
                1, z_dim, 1, 1, 1
            )
            latents_5d = latents_5d * latents_std + latents_mean
            return vae.to(torch.float32).decode(latents_5d, return_dict=False)[0]

        with nullcontext() if grad else torch.no_grad():
            if grad and activation_checkpoint and clean.requires_grad:
                from torch.utils.checkpoint import checkpoint

                decoded = checkpoint(_decode, clean, use_reentrant=False)
            else:
                decoded = _decode(clean)
        pixels = ((decoded[:, :, 0] + 1.0) / 2.0).clamp(0.0, 1.0)
        return Images.from_dense(pixels)


__all__ = ["QwenImageVAEDecodeStage"]
