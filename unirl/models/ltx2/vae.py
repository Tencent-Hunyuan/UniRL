"""LTX2 VAE stages — video encode/decode (and optional audio decode)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from unirl.types.primitives import Video, Videos

if TYPE_CHECKING:
    from .bundle import LTX2Bundle


class LTX2VAEDecodeStage:
    """Decode latents → video frames via the LTX2 3D-VAE.

    The LTX2 VAE uses 32x spatial and 8x temporal compression with 128
    latent channels. Latents are in shape (B, C, T_lat, H_lat, W_lat).
    """

    def __init__(self, bundle: "LTX2Bundle") -> None:
        self.vae = bundle.vae
        self.dtype = bundle.dtype
        self.device = bundle.device

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> Videos:
        """Decode (already-denormalized) video latents → packed ``Videos``.

        Args:
            latents: (B, C, T_lat, H_lat, W_lat) in VAE latent space,
                ALREADY denormalized by the pipeline (``_denormalize_latents``).

        Returns:
            ``Videos`` (varlen-packed) with per-frame values in ``[0, 1]``.
        """
        # Decode in fp32: LTX2's VAE decoder (like most) is numerically
        # unstable in bf16. Mirror WAN21VAEDecodeStage.
        vae = self.vae
        latents_f32 = latents.to(torch.float32)

        # The LTX2 VAE is timestep-conditioned: its decoder multiplies a
        # required ``temb`` by a scale factor, so passing ``None`` crashes
        # (None * Parameter). diffusers' pipeline feeds decode_timestep=0.0
        # (and decode_noise_scale defaults to it → the pre-decode noise
        # injection is a no-op), so a zeros timestep reproduces inference.
        timestep = None
        if bool(getattr(vae.config, "timestep_conditioning", False)):
            timestep = torch.zeros(latents_f32.shape[0], device=latents_f32.device, dtype=latents_f32.dtype)

        decoded = vae.to(torch.float32).decode(latents_f32, timestep, return_dict=False)[0]

        # Decoder emits [B, C, T, H, W] in [-1, 1]; normalize to [0, 1].
        decoded = ((decoded + 1.0) / 2.0).clamp(0.0, 1.0).to(self.dtype)

        # Pack into the varlen ``Videos`` primitive: ``Video.frames`` is
        # [T, C, H, W], so permute each sample (C, T, H, W) → (T, C, H, W)
        # and let ``Videos.from_list`` concat along T (computing cu_seqlens).
        videos = [Video(frames=decoded[i].permute(1, 0, 2, 3).contiguous()) for i in range(int(decoded.shape[0]))]
        return Videos.from_list(videos)


class LTX2VAEEncodeStage:
    """Encode video frames → latents for I2V conditioning.

    Used to encode the first frame (source image) into latent space
    for image-to-video conditioning.
    """

    def __init__(self, bundle: "LTX2Bundle") -> None:
        self.vae = bundle.vae
        self.dtype = bundle.dtype
        self.device = bundle.device

    @torch.no_grad()
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode frames → latents.

        Args:
            frames: (B, C, T, H, W) or (B, C, H, W) pixel values in [0, 1].

        Returns:
            Latents (B, C_lat, T_lat, H_lat, W_lat).
        """
        if frames.dim() == 4:
            # Single frame → add temporal dim
            frames = frames.unsqueeze(2)
        frames = frames.to(dtype=self.vae.dtype)
        latents = self.vae.encode(frames).latent_dist.sample()
        return latents.to(self.dtype)


class LTX2AudioDecodeStage:
    """Decode audio latents → waveform via audio VAE + vocoder (LTX-2.3)."""

    def __init__(self, bundle: "LTX2Bundle") -> None:
        if bundle.audio_vae is None or bundle.vocoder is None:
            raise RuntimeError("LTX2AudioDecodeStage requires audio_vae and vocoder (LTX-2.3 checkpoint).")
        self.audio_vae = bundle.audio_vae
        self.vocoder = bundle.vocoder
        self.dtype = bundle.dtype

    @torch.no_grad()
    def decode(self, audio_latents: torch.Tensor) -> torch.Tensor:
        """Decode audio latents → waveform.

        Args:
            audio_latents: Audio latent tensor from the diffusion stage.

        Returns:
            Audio waveform tensor.
        """
        # Audio VAE decode → mel spectrogram
        mel = self.audio_vae.decode(audio_latents.to(self.audio_vae.dtype)).sample
        # Vocoder → waveform
        waveform = self.vocoder(mel)
        return waveform


__all__ = ["LTX2VAEDecodeStage", "LTX2VAEEncodeStage", "LTX2AudioDecodeStage"]
