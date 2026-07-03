"""HunyuanVideo15VAEDecodeStage — LatentSegment → Videos via 3D VAE decode.

Implements ``DecodeStage[LatentSegment, Videos]``. Reads the final
stored position from ``LatentSegment.latents[:, -1]`` (the clean
latent at ``T``, which :class:`HunyuanVideo15DiffusionStage` always
stores) as a 5D channel-first tensor ``[B, C, T_lat, H_lat, W_lat]``,
runs VAE decode in fp32 (bf16 unsupported by most VAE implementations),
normalizes pixels from ``[-1, 1]`` to ``[0, 1]``, then packs each
sample into a ``Video`` (``[T, C, H, W]`` frames layout) and emits a
varlen ``Videos`` primitive.

No ``HunyuanVideo15VAEEncodeStage`` here — v1 is text-to-video only;
encode is only needed for I2V (first-frame conditioning), which is
deferred along with the SigLIP vision stage.

Diverges from :class:`unirl.models.wan21.WAN21VAEDecodeStage`
only in the un-normalization branch: HunyuanVideo-1.5's VAE config
ships with a scalar ``scaling_factor`` only (no per-channel
``latents_mean`` / ``latents_std``), so we always take the scalar path.

Math derived from the original HunyuanVideo-1.5 VAE decode path
(PR #101). This module is self-contained.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch

from unirl.models.types.codec import DecodeStage
from unirl.types.primitives import Video, Videos
from unirl.types.segments import LatentSegment

from .bundle import HunyuanVideo15Bundle


class HunyuanVideo15VAEDecodeStage(DecodeStage[LatentSegment, Videos]):
    """HunyuanVideo-1.5 3D VAE decode stage."""

    def __init__(self, bundle: HunyuanVideo15Bundle) -> None:
        self.bundle = bundle

    def decode(self, s: LatentSegment, *, grad: bool = False, activation_checkpoint: bool = False) -> Videos:
        """Decode the final-step latents in *s* into a packed ``Videos`` payload.

        ``grad=False`` (default) keeps the rollout path under ``torch.no_grad()``.
        ``grad=True`` (ReFL direct-reward backprop) runs the decode WITH grad so it
        flows from the reward through the frozen VAE into ``clean``; the VAE has no
        trainable params, so only ``clean``'s graph is extended. ``activation_checkpoint``
        (grad only) recomputes the decode in backward to trade compute for memory.
        """
        if s.latents is None:
            raise ValueError("HunyuanVideo15VAEDecodeStage.decode: segment.latents is None")
        if s.latents.ndim != 6:
            raise ValueError(
                f"HunyuanVideo15VAEDecodeStage.decode: expected latents shape "
                f"[N, K, C, T_lat, H_lat, W_lat], got {tuple(s.latents.shape)}"
            )
        clean = s.latents[:, -1]
        if clean.ndim != 5:
            raise ValueError(
                f"HunyuanVideo15VAEDecodeStage.decode: expected 5D clean latents "
                f"[B, C, T_lat, H_lat, W_lat], got {tuple(clean.shape)}"
            )

        vae = self.bundle.vae
        scaling_factor = float(getattr(vae.config, "scaling_factor", 1.0))

        def _decode(lat: torch.Tensor) -> torch.Tensor:
            latents_f32 = lat.to(dtype=torch.float32) / scaling_factor
            return vae.to(torch.float32).decode(latents_f32, return_dict=False)[0]

        with nullcontext() if grad else torch.no_grad():
            if grad and activation_checkpoint and clean.requires_grad:
                from torch.utils.checkpoint import checkpoint

                decoded = checkpoint(_decode, clean, use_reentrant=False)
            else:
                decoded = _decode(clean)

        # Decoded layout: [B, C, T_dec, H_dec, W_dec] in [-1, 1].
        decoded = ((decoded + 1.0) / 2.0).clamp(0.0, 1.0)

        # Pack into the varlen ``Videos`` primitive: ``Video.frames`` is
        # ``[T, C, H, W]`` so we permute each sample from (C, T, H, W) to
        # (T, C, H, W) and let ``Videos.from_list`` concat along T.
        videos = [Video(frames=decoded[i].permute(1, 0, 2, 3).contiguous()) for i in range(int(decoded.shape[0]))]
        return Videos.from_list(videos)


__all__ = ["HunyuanVideo15VAEDecodeStage"]
