"""WAN21ImageLatentEncodeStage — packed Images → mask+VAE latent payload."""

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
    """Structural Protocol for bundles that own a 3D VAE."""

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
        pixels_list = [image.pixels for image in p.to_list()]
        if not pixels_list or any(pixels is None or pixels.ndim != 3 or pixels.shape[0] != 3 for pixels in pixels_list):
            raise ValueError(
                "WAN21ImageLatentEncodeStage.encode: expected per-sample pixels [3, H, W], "
                f"got {[None if pixels is None else tuple(pixels.shape) for pixels in pixels_list]}"
            )

        device = self.bundle.device
        dtype = self.bundle.dtype
        vae = self.bundle.vae

        batch_size = len(pixels_list)
        target_h = int(self.height)
        target_w = int(self.width)
        num_frames = int(self.num_frames)
        latent_h = target_h // _SPATIAL_DOWNSAMPLE
        latent_w = target_w // _SPATIAL_DOWNSAMPLE
        latent_t = (num_frames - 1) // _TEMPORAL_DOWNSAMPLE + 1

        resized_items = []
        for pixels in pixels_list:
            pixels = pixels.to(device=device, dtype=torch.float32).unsqueeze(0)
            if tuple(pixels.shape[-2:]) != (target_h, target_w):
                pixels = F.interpolate(
                    pixels,
                    size=(target_h, target_w),
                    mode="bicubic",
                    align_corners=False,
                    antialias=True,
                )
            resized_items.append(pixels)
        resized = torch.cat(resized_items, dim=0)
        # [0, 1] → [-1, 1] (VAE input convention).
        scaled = resized * 2.0 - 1.0
        video_condition = torch.cat(
            [scaled.unsqueeze(2), scaled.new_zeros(batch_size, 3, num_frames - 1, target_h, target_w)],
            dim=2,
        ).to(dtype=vae.dtype)

        with torch.no_grad():
            latent_condition = vae.encode(video_condition).latent_dist.mode()

        latent_condition = latent_condition.to(device=device, dtype=dtype)

        mask_lat_size = torch.ones(batch_size, 1, num_frames, latent_h, latent_w, device=device, dtype=dtype)
        mask_lat_size[:, :, 1:] = 0.0
        first_frame_mask = mask_lat_size[:, :, 0:1]
        first_frame_mask = first_frame_mask.repeat_interleave(_TEMPORAL_DOWNSAMPLE, dim=2)
        mask_lat_size = torch.cat([first_frame_mask, mask_lat_size[:, :, 1:]], dim=2)
        mask_lat_size = mask_lat_size.view(batch_size, -1, _TEMPORAL_DOWNSAMPLE, latent_h, latent_w)
        mask_lat_size = mask_lat_size.transpose(1, 2)

        if mask_lat_size.shape[2] != latent_t:
            raise RuntimeError(
                f"WAN21ImageLatentEncodeStage.encode: mask T_lat={mask_lat_size.shape[2]} "
                f"!= expected latent_t={latent_t} for num_frames={num_frames}"
            )

        payload = torch.cat([mask_lat_size, latent_condition], dim=1)
        return ImageLatentCondition(latents=payload)


__all__ = ["WAN21ImageLatentEncodeStage"]
