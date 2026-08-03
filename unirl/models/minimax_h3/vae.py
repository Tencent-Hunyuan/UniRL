"""MiniMax-H3 decode stages -- packed rows -> Videos / stereo Audios."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from unirl.types.primitives import Audio, Audios, Video, Videos

from .config import MINIMAX_H3_PATCH_SIZE
from .packing import MiniMaxH3Geometry
from .vendor import (
    MINIMAX_H3_AUDIO_LATENTS_PER_SECOND,
    unpack_audio_tokens,
    unpatchify_video_tokens,
)

if TYPE_CHECKING:
    from .bundle import MiniMaxH3Bundle

# The audio VAE hops 800 samples at 32 kHz -- i.e. the 40 latents/s the packing
# module pins the rotary clock to. Derived rather than hardcoded so the two
# constants cannot drift apart.
MINIMAX_H3_AUDIO_HOP = 800
MINIMAX_H3_AUDIO_SAMPLE_RATE = MINIMAX_H3_AUDIO_LATENTS_PER_SECOND * MINIMAX_H3_AUDIO_HOP

# The video VAE's pixel convention: ImageNet-normalized RGB over a [0, 1] base.
_PIXEL_MEAN = (0.485, 0.456, 0.406)
_PIXEL_STD = (0.229, 0.224, 0.225)


class MiniMaxH3VideoDecodeStage:
    """Packed video rows -> ``Videos``."""

    def __init__(self, bundle: "MiniMaxH3Bundle") -> None:
        self.vae = bundle.vae

    @torch.no_grad()
    def decode(self, rows: torch.Tensor, geometry: MiniMaxH3Geometry) -> Videos:
        device = self.vae.device
        dtype = next(self.vae.parameters()).dtype
        latents = unpatchify_video_tokens(
            rows.to(device=device, dtype=dtype),
            num_latent_frames=geometry.num_latent_frames,
            latent_height=geometry.latent_height,
            latent_width=geometry.latent_width,
            patch_size=MINIMAX_H3_PATCH_SIZE,
        )
        mean = torch.tensor(self.vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
        std = torch.tensor(self.vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
        latents = latents * std + mean

        video = self.vae.decode(latents.to(dtype), return_dict=False)[0]
        pixel_mean = torch.tensor(_PIXEL_MEAN, device=device).view(1, -1, 1, 1, 1)
        pixel_std = torch.tensor(_PIXEL_STD, device=device).view(1, -1, 1, 1, 1)
        video = (video.float() * pixel_std + pixel_mean).clamp(0, 1)

        # [B, C, T, H, W] -> per-sample [T, C, H, W], the Videos frame layout.
        return Videos.from_list([Video(frames=v.permute(1, 0, 2, 3).contiguous().cpu()) for v in video])


class MiniMaxH3AudioDecodeStage:
    """Packed audio rows -> stereo ``Audios``.

    MiniMax-H3 carries stereo as two channel-major blocks of audio rows, and the
    audio VAE itself is MONO -- the two channels ride through it as two batch
    items. The decoded result is therefore ``[1, 2, L]`` channel-first, which is
    NOT what ``Audios`` wants: ``Batch.pack`` treats dim 0 as the varlen length
    axis, so a ``[2, L]`` waveform would be silently read as a two-sample batch.
    This stage transposes to length-first ``[L, 2]``.

    (LTX-2's audio stage instead mean-downmixes to mono ``[L]``; that is a
    lossy shortcut this model cannot take -- H3's whole audio pitch is stereo.)
    """

    def __init__(self, bundle: "MiniMaxH3Bundle") -> None:
        self.audio_vae = bundle.audio_vae

    @torch.no_grad()
    def decode(self, rows: torch.Tensor, geometry: MiniMaxH3Geometry) -> Audios:
        device = self.audio_vae.device
        dtype = next(self.audio_vae.parameters()).dtype
        latents = unpack_audio_tokens(rows.to(device=device, dtype=dtype), num_audio_latents=geometry.num_audio_latents)
        mean = torch.tensor(self.audio_vae.config.latents_mean, device=device).view(1, -1, 1)
        std = torch.tensor(self.audio_vae.config.latents_std, device=device).view(1, -1, 1)
        latents = latents * std + mean

        audio = self.audio_vae.decode(latents.to(dtype), return_dict=False)[0]
        # [channels-as-batch, 1, L] -> [1, channels, L], matching the reference.
        audio = audio.float().permute(1, 0, 2)
        # -> per-sample length-first [L, C] for the varlen pack.
        return Audios.from_list([Audio(waveform=a.transpose(0, 1).contiguous().cpu()) for a in audio])


__all__ = [
    "MINIMAX_H3_AUDIO_SAMPLE_RATE",
    "MiniMaxH3AudioDecodeStage",
    "MiniMaxH3VideoDecodeStage",
]
