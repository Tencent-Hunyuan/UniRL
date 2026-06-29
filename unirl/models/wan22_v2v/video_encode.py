"""WAN22VideoLatentEncodeStage — input ``Videos`` → clean 3D VAE latents."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch
import torch.nn.functional as F

from unirl.types.conditions import ImageLatentCondition
from unirl.types.primitives import Videos

_SPATIAL_DOWNSAMPLE: int = 8
_TEMPORAL_DOWNSAMPLE: int = 4


@runtime_checkable
class _VAEBundle(Protocol):
    vae: Any
    device: torch.device
    dtype: torch.dtype


class WAN22VideoLatentEncodeStage:
    """Encode V2V input videos into normalized clean WAN VAE latents."""

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

    def encode(self, p: Videos) -> ImageLatentCondition:
        if not isinstance(p, Videos):
            raise TypeError(f"WAN22VideoLatentEncodeStage.encode: expected Videos, got {type(p).__name__}")

        items = p.to_list()
        if not items:
            raise ValueError("WAN22VideoLatentEncodeStage.encode: empty Videos batch")

        device = self.bundle.device
        vae = self.bundle.vae
        target_h = int(self.height)
        target_w = int(self.width)
        num_frames = int(self.num_frames)
        if (num_frames - 1) % _TEMPORAL_DOWNSAMPLE != 0:
            raise ValueError(
                f"WAN VAE temporal_downsample={_TEMPORAL_DOWNSAMPLE} requires "
                f"(num_frames - 1) % {_TEMPORAL_DOWNSAMPLE} == 0, got num_frames={num_frames}; "
                f"valid choices: 1, 5, 9, 13, 17, 21, ..."
            )
        latent_t = (num_frames - 1) // _TEMPORAL_DOWNSAMPLE + 1

        per_sample = []
        for idx, video in enumerate(items):
            frames = video.frames
            if frames is None or frames.ndim != 4 or int(frames.shape[1]) != 3:
                raise ValueError(
                    f"WAN22VideoLatentEncodeStage.encode: sample {idx} expected frames [T, 3, H, W], "
                    f"got shape {None if frames is None else tuple(frames.shape)}"
                )
            frames = self._sample_frames(frames, target_frames=num_frames)
            x = frames.to(device=device, dtype=torch.float32).clamp_(0.0, 1.0)
            x = F.interpolate(x, size=(target_h, target_w), mode="bicubic", align_corners=False, antialias=True)
            x = x * 2.0 - 1.0
            per_sample.append(x.permute(1, 0, 2, 3).contiguous())

        video_in = torch.stack(per_sample, dim=0)
        with torch.no_grad():
            latents = vae.encode(video_in.to(dtype=vae.dtype)).latent_dist.mode()

        if int(latents.shape[2]) != latent_t:
            raise RuntimeError(
                f"WAN22VideoLatentEncodeStage.encode: VAE produced T_lat={int(latents.shape[2])} "
                f"!= expected latent_t={latent_t} for num_frames={num_frames}"
            )

        latents = latents.to(device=device, dtype=self.bundle.dtype)
        vae_config = vae.config
        latents_mean = getattr(vae_config, "latents_mean", None)
        latents_std = getattr(vae_config, "latents_std", None)
        if latents_mean is not None and latents_std is not None:
            z_dim = int(getattr(vae_config, "z_dim", latents.shape[1]))
            mean = torch.tensor(latents_mean, device=device, dtype=self.bundle.dtype).view(1, z_dim, 1, 1, 1)
            std = torch.tensor(latents_std, device=device, dtype=self.bundle.dtype).view(1, z_dim, 1, 1, 1)
            latents = (latents - mean) / std
        else:
            scaling_factor = float(getattr(vae_config, "scaling_factor", 1.0))
            latents = latents * scaling_factor

        return ImageLatentCondition(latents=latents)

    @staticmethod
    def _sample_frames(frames: torch.Tensor, *, target_frames: int) -> torch.Tensor:
        total = int(frames.shape[0])
        if total < 1:
            raise ValueError("WAN22VideoLatentEncodeStage.encode: condition video has no frames")
        if total == int(target_frames):
            return frames
        indices = torch.linspace(0, total - 1, steps=int(target_frames), device=frames.device)
        indices = indices.round().to(dtype=torch.long).clamp_(0, total - 1)
        return frames.index_select(0, indices)


__all__ = ["WAN22VideoLatentEncodeStage"]
