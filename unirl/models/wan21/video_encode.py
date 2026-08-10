"""WAN video VAE encode stage — target ``Videos`` → clean normalized latents."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import torch
import torch.nn.functional as F

from unirl.models.types.codec import EncodeStage
from unirl.types.conditions import ImageLatentCondition
from unirl.types.primitives import Videos

_SPATIAL_DOWNSAMPLE = 8
_TEMPORAL_DOWNSAMPLE = 4


@runtime_checkable
class _VAEBundle(Protocol):
    vae: Any
    device: torch.device
    dtype: torch.dtype


class WANVideoLatentEncodeStage(EncodeStage[Videos, ImageLatentCondition]):
    """Encode fixed-length RGB videos with the WAN 3D VAE.

    Input videos use the framework layout ``[T, 3, H, W]`` with values in
    ``[0, 1]``. The stage uniformly samples ``num_frames``, resizes each frame,
    maps pixels to ``[-1, 1]``, and returns the normalized latent convention
    consumed by WAN's diffusion transformer.
    """

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
        if self.num_frames < 1 or (self.num_frames - 1) % _TEMPORAL_DOWNSAMPLE != 0:
            raise ValueError(
                f"WAN VAE temporal_downsample={_TEMPORAL_DOWNSAMPLE} requires "
                f"(num_frames - 1) % {_TEMPORAL_DOWNSAMPLE} == 0, got num_frames={self.num_frames}; "
                "valid choices: 1, 5, 9, 13, 17, 21, ..."
            )
        if self.height < 1 or self.width < 1:
            raise ValueError(
                f"WANVideoLatentEncodeStage: height/width must be positive, got {self.height}x{self.width}."
            )
        if self.height % _SPATIAL_DOWNSAMPLE or self.width % _SPATIAL_DOWNSAMPLE:
            raise ValueError(
                f"WAN VAE spatial_downsample={_SPATIAL_DOWNSAMPLE} requires height/width divisible by "
                f"{_SPATIAL_DOWNSAMPLE}, got {self.height}x{self.width}."
            )

    @torch.no_grad()
    def encode(self, p: Videos) -> ImageLatentCondition:
        if self.bundle.vae is None:
            raise RuntimeError(
                "WANVideoLatentEncodeStage.encode: no VAE loaded (load_vae=False). "
                "Trainer-side video SFT requires load_vae=True."
            )
        if not isinstance(p, Videos):
            raise TypeError(f"WANVideoLatentEncodeStage.encode: expected Videos, got {type(p).__name__}.")

        items = p.to_list()
        if not items:
            raise ValueError("WANVideoLatentEncodeStage.encode: empty Videos batch.")

        per_sample = []
        for idx, video in enumerate(items):
            frames = video.frames
            if frames is None or frames.ndim != 4 or int(frames.shape[1]) != 3:
                raise ValueError(
                    f"WANVideoLatentEncodeStage.encode: sample {idx} expected frames [T, 3, H, W], "
                    f"got shape {None if frames is None else tuple(frames.shape)}."
                )
            frames = self._sample_frames(frames, target_frames=self.num_frames)
            frames = frames.to(device=self.bundle.device, dtype=torch.float32).clamp_(0.0, 1.0)
            frames = F.interpolate(
                frames,
                size=(self.height, self.width),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            ).clamp_(0.0, 1.0)
            per_sample.append((frames * 2.0 - 1.0).permute(1, 0, 2, 3).contiguous())

        video_in = torch.stack(per_sample, dim=0)
        vae = self.bundle.vae
        # The bundle's WanVideoVAE applies its checkpoint mean/std inside encode.
        latents = vae.encode(video_in.to(dtype=vae.dtype)).latent_dist.mode()

        expected_shape = (
            (self.num_frames - 1) // _TEMPORAL_DOWNSAMPLE + 1,
            self.height // _SPATIAL_DOWNSAMPLE,
            self.width // _SPATIAL_DOWNSAMPLE,
        )
        if tuple(latents.shape[2:]) != expected_shape:
            raise RuntimeError(
                f"WANVideoLatentEncodeStage.encode: VAE produced latent geometry {tuple(latents.shape[2:])} "
                f"!= expected {expected_shape} for {self.num_frames}x{self.height}x{self.width}."
            )

        return ImageLatentCondition(latents=latents.to(device=self.bundle.device, dtype=self.bundle.dtype))

    @staticmethod
    def _sample_frames(frames: torch.Tensor, *, target_frames: int) -> torch.Tensor:
        total = int(frames.shape[0])
        if total < 1:
            raise ValueError("WANVideoLatentEncodeStage.encode: target video has no frames.")
        if total == int(target_frames):
            return frames
        indices = torch.linspace(0, total - 1, steps=int(target_frames), device=frames.device)
        indices = indices.round().to(dtype=torch.long).clamp_(0, total - 1)
        return frames.index_select(0, indices)


__all__ = ["WANVideoLatentEncodeStage"]
