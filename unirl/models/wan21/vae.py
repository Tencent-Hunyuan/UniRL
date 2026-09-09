"""WAN 3D VAE codec stages."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Protocol

import torch
import torch.nn.functional as F

from unirl.models.types.codec import DecodeStage, EncodeStage
from unirl.types.conditions import ImageLatentCondition
from unirl.types.primitives import Video, Videos
from unirl.types.segments import LatentSegment

from .bundle import WAN21Bundle

_SPATIAL_DOWNSAMPLE = 8
_TEMPORAL_DOWNSAMPLE = 4


class _VAEBundle(Protocol):
    vae: Any
    device: torch.device
    dtype: torch.dtype


class WANVideoLatentEncodeStage(EncodeStage[Videos, ImageLatentCondition]):
    """Encode fixed-length RGB videos with the WAN 3D VAE."""

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
            raise ValueError(f"{type(self).__name__}: height/width must be positive, got {self.height}x{self.width}.")
        if self.height % _SPATIAL_DOWNSAMPLE or self.width % _SPATIAL_DOWNSAMPLE:
            raise ValueError(
                f"WAN VAE spatial_downsample={_SPATIAL_DOWNSAMPLE} requires height/width divisible by "
                f"{_SPATIAL_DOWNSAMPLE}, got {self.height}x{self.width}."
            )

    @torch.no_grad()
    def encode(self, p: Videos) -> ImageLatentCondition:
        owner = type(self).__name__
        if self.bundle.vae is None:
            raise RuntimeError(
                f"{owner}.encode: no VAE loaded (load_vae=False). Trainer-side video encoding requires load_vae=True."
            )
        if not isinstance(p, Videos):
            raise TypeError(f"{owner}.encode: expected Videos, got {type(p).__name__}.")

        items = p.to_list()
        if not items:
            raise ValueError(f"{owner}.encode: empty Videos batch.")

        per_sample = []
        for idx, video in enumerate(items):
            frames = video.frames
            if frames is None or frames.ndim != 4 or int(frames.shape[1]) != 3:
                raise ValueError(
                    f"{owner}.encode: sample {idx} expected frames [T, 3, H, W], "
                    f"got shape {None if frames is None else tuple(frames.shape)}."
                )
            frames = self._sample_frames(frames)
            frames = frames.to(device=self.bundle.device, dtype=torch.float32).clamp_(0.0, 1.0)
            frames = F.interpolate(
                frames,
                size=(self.height, self.width),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            frames = self._postprocess_resized_frames(frames)
            per_sample.append((frames * 2.0 - 1.0).permute(1, 0, 2, 3).contiguous())

        video_in = torch.stack(per_sample, dim=0)
        vae = self.bundle.vae
        latents = vae.encode(video_in.to(dtype=vae.dtype)).latent_dist.mode()

        expected_shape = (
            (self.num_frames - 1) // _TEMPORAL_DOWNSAMPLE + 1,
            self.height // _SPATIAL_DOWNSAMPLE,
            self.width // _SPATIAL_DOWNSAMPLE,
        )
        if tuple(latents.shape[2:]) != expected_shape:
            raise RuntimeError(
                f"{owner}.encode: VAE produced latent geometry {tuple(latents.shape[2:])} "
                f"!= expected {expected_shape} for {self.num_frames}x{self.height}x{self.width}."
            )

        latents = latents.to(device=self.bundle.device, dtype=self.bundle.dtype)
        return ImageLatentCondition(latents=self._postprocess_latents(latents))

    def _postprocess_resized_frames(self, frames: torch.Tensor) -> torch.Tensor:
        return frames.clamp_(0.0, 1.0)

    def _postprocess_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return latents

    def _sample_frames(self, frames: torch.Tensor) -> torch.Tensor:
        total = int(frames.shape[0])
        if total < 1:
            raise ValueError("WAN video encoder: target video has no frames.")
        if total == self.num_frames:
            return frames
        indices = torch.linspace(0, total - 1, steps=self.num_frames, device=frames.device)
        indices = indices.round().to(dtype=torch.long).clamp_(0, total - 1)
        return frames.index_select(0, indices)


class WAN21VAEDecodeStage(DecodeStage[LatentSegment, Videos]):
    """WAN 2.1 3D VAE decode stage."""

    def __init__(self, bundle: WAN21Bundle) -> None:
        self.bundle = bundle

    def decode(self, s: LatentSegment, *, grad: bool = False, activation_checkpoint: bool = False) -> Videos:
        """Decode the final-step latents in *s* into a packed ``Videos`` payload."""
        if self.bundle.vae is None:
            raise RuntimeError(
                "WAN21VAEDecodeStage.decode: no VAE loaded (load_vae=False). "
                "The trainer-side pipeline cannot decode latents in this "
                "configuration — separate-engine recipes decode in the "
                "rollout engine; trainside rollout requires load_vae=True."
            )
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

        with nullcontext() if grad else torch.no_grad():
            decoded = self._vae_decode(clean)

        decoded = ((decoded + 1.0) / 2.0).clamp(0.0, 1.0)

        return self._pack_videos(decoded)

    def decode_with_grad(self, z_final: torch.Tensor) -> torch.Tensor:
        """Differentiable VAE decode: ``z_final → pixels`` with autograd alive."""
        if z_final.ndim != 5:
            raise ValueError(
                f"WAN21VAEDecodeStage.decode_with_grad: expected 5D z_final "
                f"[B, C, T_lat, H_lat, W_lat], got {tuple(z_final.shape)}"
            )
        return self._vae_decode(z_final)

    def _vae_decode(self, clean: torch.Tensor) -> torch.Tensor:
        """Run ``WanVideoVAE.decode`` and return pixels in the VAE's native ``[-1, 1]``."""
        vae = self.bundle.vae
        vae_dtype = vae.dtype
        latents = clean.to(dtype=vae_dtype)
        decoded = vae.decode(latents, device=latents.device, tiled=True)
        return decoded

    def _pack_videos(self, decoded: torch.Tensor) -> Videos:
        """Pack a dense ``[B, C, T_dec, H_dec, W_dec]`` pixel tensor into ``Videos``."""
        videos = [Video(frames=decoded[i].permute(1, 0, 2, 3).contiguous()) for i in range(int(decoded.shape[0]))]
        return Videos.from_list(videos)


__all__ = ["WAN21VAEDecodeStage", "WANVideoLatentEncodeStage"]
