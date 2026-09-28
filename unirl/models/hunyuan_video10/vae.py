"""HunyuanVideo10VAEDecodeStage — 5D ``[B, C, T_lat, H_lat, W_lat]`` to varlen ``Videos`` (``[T, C, H, W]`` frames)."""

from __future__ import annotations

from contextlib import nullcontext

import torch

from unirl.models.types.codec import DecodeStage
from unirl.types.primitives import Video, Videos
from unirl.types.segments import LatentSegment

from .bundle import HunyuanVideo10Bundle


class HunyuanVideo10VAEDecodeStage(DecodeStage[LatentSegment, Videos]):
    """HunyuanVideo-1.0 3D VAE decode stage."""

    def __init__(self, bundle: HunyuanVideo10Bundle) -> None:
        self.bundle = bundle

    def decode(self, s: LatentSegment, *, grad: bool = False, activation_checkpoint: bool = False) -> Videos:
        """Decode the final-step latents in *s* into a packed ``Videos`` payload."""
        if s.latents is None:
            raise ValueError("HunyuanVideo10VAEDecodeStage.decode: segment.latents is None")
        if s.latents.ndim != 6:
            raise ValueError(
                f"HunyuanVideo10VAEDecodeStage.decode: expected latents shape "
                f"[N, K, C, T_lat, H_lat, W_lat], got {tuple(s.latents.shape)}"
            )
        clean = s.latents[:, -1]
        if clean.ndim != 5:
            raise ValueError(
                f"HunyuanVideo10VAEDecodeStage.decode: expected 5D clean latents "
                f"[B, C, T_lat, H_lat, W_lat], got {tuple(clean.shape)}"
            )

        vae = self.bundle.vae
        scaling_factor = float(getattr(vae.config, "scaling_factor", 0.476986))

        def _decode(lat: torch.Tensor) -> torch.Tensor:
            latents_f32 = lat.to(dtype=torch.float32) / scaling_factor
            return vae.to(torch.float32).decode(latents_f32, return_dict=False)[0]

        with nullcontext() if grad else torch.no_grad():
            if grad and activation_checkpoint and clean.requires_grad:
                from torch.utils.checkpoint import checkpoint

                decoded = checkpoint(_decode, clean, use_reentrant=False)
            else:
                decoded = _decode(clean)

        decoded = ((decoded + 1.0) / 2.0).clamp(0.0, 1.0)

        videos = [Video(frames=decoded[i].permute(1, 0, 2, 3).contiguous()) for i in range(int(decoded.shape[0]))]
        return Videos.from_list(videos)


__all__ = ["HunyuanVideo10VAEDecodeStage"]
