"""SD3 VAE codec stages — LatentSegment ↔ Images."""

from __future__ import annotations

from contextlib import nullcontext

import torch

from unirl.models.types.codec import DecodeStage, EncodeStage
from unirl.types.conditions.image import ImageLatentCondition
from unirl.types.primitives import Images
from unirl.types.segments import LatentSegment

from .bundle import SD3Bundle


class SD3VAEDecodeStage(DecodeStage[LatentSegment, Images]):
    """SD3 VAE decode stage."""

    def __init__(self, bundle: SD3Bundle) -> None:
        self.bundle = bundle

    def decode(self, s: LatentSegment, *, grad: bool = False, activation_checkpoint: bool = False) -> Images:
        """Decode the final-step latents in *s* into pixel images."""
        if self.bundle.vae is None:
            raise RuntimeError(
                "SD3VAEDecodeStage.decode: no VAE loaded (load_vae=False). "
                "The trainer-side pipeline cannot decode latents in this "
                "configuration — separate-engine recipes decode in the "
                "rollout engine; trainside rollout requires load_vae=True."
            )
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
        return Images.from_dense(pixels)


class SD3VAEEncodeStage(EncodeStage[Images, ImageLatentCondition]):
    """SD3 VAE encode stage — the strict inverse of :class:`SD3VAEDecodeStage`."""

    def __init__(self, bundle: SD3Bundle) -> None:
        self.bundle = bundle

    @torch.no_grad()
    def encode(self, p: Images) -> ImageLatentCondition:
        if self.bundle.vae is None:
            raise RuntimeError(
                "SD3VAEEncodeStage.encode: no VAE loaded (load_vae=False). "
                "The trainer-side pipeline cannot encode images in this "
                "configuration — separate-engine recipes encode in the "
                "rollout engine; trainside / SFT paths require load_vae=True."
            )
        try:
            pixels = p.to_dense()
        except ValueError as exc:
            raise ValueError(
                f"SD3VAEEncodeStage.encode requires a non-empty batch with uniform image shapes; {exc}"
            ) from exc
        if not isinstance(pixels, torch.Tensor) or pixels.ndim != 4 or pixels.shape[1] != 3:
            raise ValueError(
                f"SD3VAEEncodeStage.encode: expected pixels [B, 3, H, W], got "
                f"{tuple(pixels.shape) if isinstance(pixels, torch.Tensor) else type(pixels).__name__}"
            )
        scaling_factor = self.bundle.vae.config.scaling_factor
        shift_factor = getattr(self.bundle.vae.config, "shift_factor", None)
        vae = self.bundle.vae.to(torch.float32)
        x = pixels.to(device=self.bundle.device, dtype=torch.float32) * 2.0 - 1.0
        z = vae.encode(x).latent_dist.mode()
        if shift_factor is not None:
            z = z - float(shift_factor)
        z = z * scaling_factor
        return ImageLatentCondition(latents=z.to(dtype=torch.float32))


__all__ = ["SD3VAEDecodeStage", "SD3VAEEncodeStage"]
