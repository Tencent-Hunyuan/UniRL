"""WAN21ImageLatentEncodeStage — Images → 20-channel mask+VAE latent payload.

Mirrors diffusers ``pipelines/wan/pipeline_wan_i2v.py:423-481`` for the
``expand_timesteps=False`` path (WAN 2.1 I2V + WAN 2.2 14B I2V). Encode
uses ``vae.encode(x).latent_dist.mode()`` (deterministic) so rollout /
replay produce bitwise-equal image latents — sampling here would drift
the GRPO logp ratio. Per-channel normalization is the strict inverse of
``wan21/vae.py:78-87`` (decode = ``latent * std + mean``).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch
import torch.nn.functional as F

from unirl.models.types.codec import EncodeStage
from unirl.types.conditions import ImageLatentCondition
from unirl.types.primitives import Images

_SPATIAL_DOWNSAMPLE: int = 8
_TEMPORAL_DOWNSAMPLE: int = 4


@runtime_checkable
class _VAEBundle(Protocol):
    """Structural Protocol for bundles that own a 3D VAE.

    Both :class:`WAN21Bundle` and :class:`WAN22Bundle` satisfy this
    structurally; this stage is shared across both pipelines.
    """

    vae: Any
    device: torch.device
    dtype: torch.dtype


class WAN21ImageLatentEncodeStage(EncodeStage[Images, ImageLatentCondition]):
    """Encode a reference image into the 20-channel WAN I2V condition payload."""

    def __init__(
        self,
        bundle: _VAEBundle,
        *,
        num_frames: int,
        height: int,
        width: int,
    ) -> None:
        self.bundle = bundle
        self.num_frames = int(num_frames)
        self.height = int(height)
        self.width = int(width)

    def encode(self, p: Images) -> ImageLatentCondition:
        if self.bundle.vae is None:
            raise RuntimeError(
                "WAN21ImageLatentEncodeStage.encode: no VAE loaded "
                "(load_vae=False). The trainer-side pipeline cannot encode "
                "reference images in this configuration — separate-engine "
                "recipes encode in the rollout engine; trainside I2V "
                "requires load_vae=True."
            )
        if not isinstance(p, Images):
            raise TypeError(f"WAN21ImageLatentEncodeStage.encode: expected Images, got {type(p).__name__}")
        pixels = p.pixels
        if pixels is None or pixels.ndim != 4 or pixels.shape[1] != 3:
            raise ValueError(
                f"WAN21ImageLatentEncodeStage.encode: expected pixels [B, 3, H, W], "
                f"got shape {None if pixels is None else tuple(pixels.shape)}"
            )

        device = self.bundle.device
        dtype = self.bundle.dtype
        vae = self.bundle.vae

        batch_size = int(pixels.shape[0])
        target_h = int(self.height)
        target_w = int(self.width)
        num_frames = int(self.num_frames)
        latent_h = target_h // _SPATIAL_DOWNSAMPLE
        latent_w = target_w // _SPATIAL_DOWNSAMPLE
        # Latent temporal axis: pixel-space frames collapse to
        # ``(num_frames - 1) // 4 + 1``. Caller has already gated on the
        # ``(num_frames - 1) % 4 == 0`` invariant.
        latent_t = (num_frames - 1) // _TEMPORAL_DOWNSAMPLE + 1

        pixels = pixels.to(device=device, dtype=torch.float32)
        resized = F.interpolate(
            pixels,
            size=(target_h, target_w),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        # [0, 1] → [-1, 1] (VAE input convention).
        scaled = resized * 2.0 - 1.0
        # Add T axis then zero-pad to (num_frames - 1) blanks → [B, 3, T_pix, H, W].
        video_condition = torch.cat(
            [scaled.unsqueeze(2), scaled.new_zeros(batch_size, 3, num_frames - 1, target_h, target_w)],
            dim=2,
        ).to(dtype=vae.dtype)

        with torch.no_grad():
            # ``.mode()`` (deterministic) vs ``.sample()`` — diffusers I2V
            # uses ``retrieve_latents(..., sample_mode="argmax")`` which is
            # ``.mode()``. ``.sample()`` would drift rollout/replay.
            latent_condition = vae.encode(video_condition).latent_dist.mode()

        latent_condition = latent_condition.to(device=device, dtype=dtype)

        # Per-channel normalization — strict inverse of ``wan21/vae.py:78-87``
        # decode: ``latent_decoded = stored * std + mean``.
        vae_config = vae.config
        latents_mean = getattr(vae_config, "latents_mean", None)
        latents_std = getattr(vae_config, "latents_std", None)
        if latents_mean is not None and latents_std is not None:
            z_dim = int(getattr(vae_config, "z_dim", latent_condition.shape[1]))
            mean = torch.tensor(latents_mean, device=device, dtype=dtype).view(1, z_dim, 1, 1, 1)
            std = torch.tensor(latents_std, device=device, dtype=dtype).view(1, z_dim, 1, 1, 1)
            latent_condition = (latent_condition - mean) / std
        else:
            scaling_factor = float(getattr(vae_config, "scaling_factor", 1.0))
            latent_condition = latent_condition * scaling_factor

        # 4-channel first-frame mask (mirrors diffusers
        # ``pipeline_wan_i2v.py:468-479``): 1.0 at first pixel-time slot,
        # 0.0 elsewhere; then view-reshape + transpose so the temporal
        # downsampling factor ends up on the channel axis.
        mask_lat_size = torch.ones(batch_size, 1, num_frames, latent_h, latent_w, device=device, dtype=dtype)
        mask_lat_size[:, :, 1:] = 0.0
        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = first_frame_mask.repeat_interleave(_TEMPORAL_DOWNSAMPLE, dim=2)
        mask_lat_size = torch.cat([first_frame_mask, mask_lat_size[:, :, 1:]], dim=2)
        mask_lat_size = mask_lat_size.view(batch_size, -1, _TEMPORAL_DOWNSAMPLE, latent_h, latent_w)
        mask_lat_size = mask_lat_size.transpose(1, 2)
        # Now ``mask_lat_size`` is [B, 4, T_lat, latent_h, latent_w].

        # Defensive: when ``num_frames=1`` the reshape above would produce
        # T_lat=1; for the standard 5,9,13,... family it lines up.
        if mask_lat_size.shape[2] != latent_t:
            raise RuntimeError(
                f"WAN21ImageLatentEncodeStage.encode: mask T_lat={mask_lat_size.shape[2]} "
                f"!= expected latent_t={latent_t} for num_frames={num_frames}"
            )

        payload = torch.cat([mask_lat_size, latent_condition], dim=1)
        return ImageLatentCondition(latents=payload)


__all__ = ["WAN21ImageLatentEncodeStage"]
