"""LTX2 VAE stages — video encode/decode (and optional audio decode)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from unirl.types.primitives import Video, Videos

if TYPE_CHECKING:
    from .bundle import LTX2Bundle


class LTX2VAEDecodeStage:
    """Decode denormalized video latents into varlen-packed ``Videos`` with frames in ``[0, 1]``."""

    def __init__(self, bundle: "LTX2Bundle") -> None:
        self.vae = bundle.vae
        self.dtype = bundle.dtype
        self.device = bundle.device

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> Videos:
        """Decode (already-denormalized) video latents → packed ``Videos``."""
        vae = self.vae
        latents_f32 = latents.to(torch.float32)

        timestep = None
        if vae.config.timestep_conditioning:
            timestep = torch.zeros(latents_f32.shape[0], device=latents_f32.device, dtype=latents_f32.dtype)

        decoded = vae.to(torch.float32).decode(latents_f32, timestep, return_dict=False)[0]

        decoded = ((decoded + 1.0) / 2.0).clamp(0.0, 1.0).to(self.dtype)

        videos = [Video(frames=decoded[i].permute(1, 0, 2, 3).contiguous()) for i in range(int(decoded.shape[0]))]
        return Videos.from_list(videos)


class LTX2VAEEncodeStage:
    """Encode frames ``(B, C, T, H, W)`` or ``(B, C, H, W)`` in ``[0, 1]`` into latents."""

    def __init__(self, bundle: "LTX2Bundle") -> None:
        self.vae = bundle.vae
        self.dtype = bundle.dtype
        self.device = bundle.device

    @torch.no_grad()
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode frames → latents."""
        if frames.dim() == 4:
            frames = frames.unsqueeze(2)
        frames = frames.to(dtype=self.vae.dtype)
        latents = self.vae.encode(frames).latent_dist.sample()
        return latents.to(self.dtype)


class LTX2AudioDecodeStage:
    """Decode packed audio latents ``[B, S, D]`` — denormalized while packed, then unpacked to ``[B, C, L, M]``."""

    def __init__(self, bundle: "LTX2Bundle") -> None:
        if bundle.audio_vae is None or bundle.vocoder is None:
            raise RuntimeError("LTX2AudioDecodeStage requires audio_vae and vocoder (LTX-2.3 checkpoint).")
        self.audio_vae = bundle.audio_vae
        self.vocoder = bundle.vocoder
        self.dtype = bundle.dtype

    @torch.no_grad()
    def decode(self, audio_latents: torch.Tensor, *, audio_latent_length: int) -> torch.Tensor:
        """Decode packed audio latents → waveform."""
        mel_bins = self.audio_vae.config.mel_bins
        mel_compression = self.audio_vae.mel_compression_ratio
        latent_mel_bins = mel_bins // mel_compression

        aud = audio_latents.float()
        latents_mean = self.audio_vae.latents_mean.to(aud.device, aud.dtype)
        latents_std = self.audio_vae.latents_std.to(aud.device, aud.dtype)
        aud = aud * latents_std + latents_mean
        aud = aud.unflatten(2, (-1, latent_mel_bins)).transpose(1, 2).to(torch.float32)
        mel = self.audio_vae.to(torch.float32).decode(aud, return_dict=False)[0]
        waveform = self.vocoder.to(torch.float32)(mel)
        return waveform


__all__ = ["LTX2VAEDecodeStage", "LTX2VAEEncodeStage", "LTX2AudioDecodeStage"]
