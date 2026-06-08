"""HunyuanVideoVAEDecodeStage -- LatentSegment -> Videos via 3D VAE decode.

Implements ``DecodeStage[LatentSegment, Videos]``. Reads the final
stored position from ``LatentSegment.latents[:, -1]`` (the clean
latent at ``T``, which :class:`HunyuanVideoDiffusionStage` always
stores) as a 5D channel-first tensor ``[B, C, T_lat, H_lat, W_lat]``,
runs VAE decode in fp32 (bf16 unsupported by most VAE implementations),
normalizes pixels from ``[-1, 1]`` to ``[0, 1]``, then packs each
sample into a ``Video`` (``[T, C, H, W]`` frames layout) and emits a
varlen ``Videos`` primitive.

HunyuanVideo-1.0 VAE config:
- spatial_compression_ratio: 8
- temporal_compression_ratio: 4
- latent_channels: 16
- scaling_factor: 0.476986
"""

from __future__ import annotations

import torch

from unirl.models.types.codec import DecodeStage
from unirl.types.primitives import Video, Videos
from unirl.types.segments import LatentSegment

from .bundle import HunyuanVideoBundle


class HunyuanVideoVAEDecodeStage(DecodeStage[LatentSegment, Videos]):
    """HunyuanVideo-1.0 3D VAE decode stage."""

    def __init__(self, bundle: HunyuanVideoBundle) -> None:
        self.bundle = bundle

    def decode(self, s: LatentSegment) -> Videos:
        """Decode the final-step latents in *s* into a packed ``Videos`` payload."""
        if s.latents is None:
            raise ValueError("HunyuanVideoVAEDecodeStage.decode: segment.latents is None")
        if s.latents.ndim != 6:
            raise ValueError(
                f"HunyuanVideoVAEDecodeStage.decode: expected latents shape "
                f"[N, K, C, T_lat, H_lat, W_lat], got {tuple(s.latents.shape)}"
            )
        clean = s.latents[:, -1]
        if clean.ndim != 5:
            raise ValueError(
                f"HunyuanVideoVAEDecodeStage.decode: expected 5D clean latents "
                f"[B, C, T_lat, H_lat, W_lat], got {tuple(clean.shape)}"
            )

        vae = self.bundle.vae
        scaling_factor = float(getattr(vae.config, "scaling_factor", 0.476986))

        with torch.no_grad():
            latents_f32 = clean.to(dtype=torch.float32) / scaling_factor
            decoded = vae.to(torch.float32).decode(latents_f32, return_dict=False)[0]

        # Decoded layout: [B, C, T_dec, H_dec, W_dec] in [-1, 1].
        decoded = ((decoded + 1.0) / 2.0).clamp(0.0, 1.0)

        # Pack into the varlen ``Videos`` primitive: ``Video.frames`` is
        # ``[T, C, H, W]`` so we permute each sample from (C, T, H, W) to
        # (T, C, H, W) and let ``Videos.from_list`` concat along T.
        videos = [Video(frames=decoded[i].permute(1, 0, 2, 3).contiguous()) for i in range(int(decoded.shape[0]))]
        return Videos.from_list(videos)


__all__ = ["HunyuanVideoVAEDecodeStage"]
