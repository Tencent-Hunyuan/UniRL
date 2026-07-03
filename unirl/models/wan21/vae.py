"""WAN21VAEDecodeStage — LatentSegment → Videos via 3D VAE decode.

Implements ``DecodeStage[LatentSegment, Videos]``. Reads the final stored
position from ``LatentSegment.latents[:, -1]`` (the clean latent at
``T``, which ``WAN21DiffusionStage`` always stores), denormalizes using
either per-channel ``latents_mean`` / ``latents_std`` (when the VAE
config carries them, as recent diffusers ``AutoencoderKLWan`` does) or
the scalar ``scaling_factor`` fallback, runs VAE decode in fp32 (bf16 is
unsupported by most VAE implementations), and packs the 5D
``[B, C, T, H, W]`` output into a varlen-batched ``Videos`` primitive.

**Why per-channel mean/std support (vs the legacy scalar-only path):**
diffusers' canonical Wan VAE ships with ``latents_mean`` /
``latents_std`` arrays in its config; using only ``scaling_factor``
yields off-distribution decodes. Legacy ``models/wan21.py::decode_latents``
uses the scalar fallback alone — a known latent-norm bug. The new path
follows the diffusers spec.

VAE encode for I2V's image-condition latent lives in
``WAN21ImageLatentEncodeStage`` (``image_encode.py``) — it shares the
same per-channel norm helper path but as a sibling stage so the decode
side stays focused on its single job. There is intentionally no
generic ``WAN21VAEEncodeStage`` here; the I2V encode is the only
encode path we need.

Decode math derived from diffusers' ``AutoencoderKLWan`` reference
and ``unirl/models/wan21.py:325-342, 450-462`` (do NOT import
legacy code).
"""

from __future__ import annotations

from contextlib import nullcontext

import torch

from unirl.models.types.codec import DecodeStage
from unirl.types.primitives import Video, Videos
from unirl.types.segments import LatentSegment

from .bundle import WAN21Bundle


class WAN21VAEDecodeStage(DecodeStage[LatentSegment, Videos]):
    """WAN 2.1 3D VAE decode stage."""

    def __init__(self, bundle: WAN21Bundle) -> None:
        self.bundle = bundle

    def decode(self, s: LatentSegment, *, grad: bool = False, activation_checkpoint: bool = False) -> Videos:
        """Decode the final-step latents in *s* into a packed ``Videos`` payload.

        Reads ``s.latents[:, -1]`` (the final stored position, which is
        ``T`` — the clean latent ``x_0``) as a 5D channel-first tensor
        ``[B, C, T_lat, H_lat, W_lat]``. VAE forward runs in fp32; output
        is normalized from ``[-1, 1]`` to ``[0, 1]`` and clamped before
        being packed sample-by-sample into a ``Videos`` primitive.

        ``grad=False`` (default) keeps the rollout path under ``torch.no_grad()``.
        ``grad=True`` (ReFL direct-reward backprop) runs the decode WITH grad so it
        flows from the reward through the frozen VAE into ``clean``; the VAE has no
        trainable params, so only ``clean``'s graph is extended. ``activation_checkpoint``
        (grad only) recomputes the decode in backward to trade compute for memory.
        """
        if s.latents is None:
            raise ValueError("WAN21VAEDecodeStage.decode: segment.latents is None")
        if s.latents.ndim != 6:
            raise ValueError(
                f"WAN21VAEDecodeStage.decode: expected latents shape "
                f"[N, K, C, T_lat, H_lat, W_lat], got {tuple(s.latents.shape)}"
            )
        clean = s.latents[:, -1]
        if clean.ndim != 5:
            raise ValueError(
                f"WAN21VAEDecodeStage.decode: expected 5D clean latents "
                f"[B, C, T_lat, H_lat, W_lat], got {tuple(clean.shape)}"
            )

        vae = self.bundle.vae
        vae_config = vae.config
        latents_mean = getattr(vae_config, "latents_mean", None)
        latents_std = getattr(vae_config, "latents_std", None)
        scaling_factor = getattr(vae_config, "scaling_factor", 1.0)

        device = clean.device

        def _decode(lat: torch.Tensor) -> torch.Tensor:
            latents_f32 = lat.to(dtype=torch.float32)
            if latents_mean is not None and latents_std is not None:
                # diffusers Wan VAE spec: latent ↦ latent * std + mean
                # (un-normalize). Reshape to [1, C, 1, 1, 1] to broadcast
                # over the (B, T, H, W) axes.
                z_dim = int(getattr(vae_config, "z_dim", lat.shape[1]))
                mean = torch.tensor(latents_mean, device=device, dtype=torch.float32).view(1, z_dim, 1, 1, 1)
                std = torch.tensor(latents_std, device=device, dtype=torch.float32).view(1, z_dim, 1, 1, 1)
                latents_f32 = latents_f32 * std + mean
            else:
                latents_f32 = latents_f32 / float(scaling_factor)

            return vae.to(torch.float32).decode(latents_f32).sample

        with nullcontext() if grad else torch.no_grad():
            if grad and activation_checkpoint and clean.requires_grad:
                from torch.utils.checkpoint import checkpoint

                decoded = checkpoint(_decode, clean, use_reentrant=False)
            else:
                decoded = _decode(clean)

        # Decoded layout is [B, C, T_dec, H_dec, W_dec] in [-1, 1].
        # Normalize to [0, 1] and clamp before packing.
        decoded = ((decoded + 1.0) / 2.0).clamp(0.0, 1.0)

        # Pack into the varlen ``Videos`` primitive: ``Video.frames`` is
        # ``[T, C, H, W]`` so we permute each sample from (C, T, H, W) →
        # (T, C, H, W) and let ``Videos.from_list`` concat along T.
        videos = [Video(frames=decoded[i].permute(1, 0, 2, 3).contiguous()) for i in range(int(decoded.shape[0]))]
        return Videos.from_list(videos)


__all__ = ["WAN21VAEDecodeStage"]
