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
    """Decode packed audio latents → waveform via audio VAE + vocoder (LTX-2.3).

    Mirrors diffusers ``LTX2Pipeline`` audio decode (and Flow-Factory's
    ``decode_latents`` audio branch): the operation order is
    **denormalize → unpack → audio_vae.decode → vocoder** — note that unlike
    video, audio is denormalized while still PACKED ``[B, S, D]`` and only then
    unpacked to the spectrogram layout ``[B, C, L, M]``.
    """

    def __init__(self, bundle: "LTX2Bundle") -> None:
        if bundle.audio_vae is None or bundle.vocoder is None:
            raise RuntimeError("LTX2AudioDecodeStage requires audio_vae and vocoder (LTX-2.3 checkpoint).")
        self.audio_vae = bundle.audio_vae
        self.vocoder = bundle.vocoder
        self.dtype = bundle.dtype

    @staticmethod
    def _denormalize_audio_latents(
        latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor
    ) -> torch.Tensor:
        """Inverse of the audio VAE normalization, on the packed ``[B, S, D]``
        latent (verbatim from diffusers ``_denormalize_audio_latents``)."""
        latents_mean = latents_mean.to(latents.device, latents.dtype)
        latents_std = latents_std.to(latents.device, latents.dtype)
        return latents * latents_std + latents_mean

    @staticmethod
    def _unpack_audio_latents(latents: torch.Tensor, latent_length: int, num_mel_bins: int) -> torch.Tensor:
        """Packed ``[B, L, C*M]`` → spectrogram ``[B, C, L, M]`` (verbatim from
        diffusers ``_unpack_audio_latents``, default no-patch path: implicit
        ``patch_size = M``, ``patch_size_t = 1``)."""
        return latents.unflatten(2, (-1, num_mel_bins)).transpose(1, 2)

    @torch.no_grad()
    def decode(self, audio_latents: torch.Tensor, *, audio_latent_length: int) -> torch.Tensor:
        """Decode packed audio latents → waveform.

        Args:
            audio_latents: Packed audio latents ``[B, S, D]`` (the clean
                final-step audio from the diffusion stage's ``aux_latents``).
            audio_latent_length: Number of audio LATENT frames ``L`` (so the
                unpack can recover ``[B, C, L, M]``).

        Returns:
            Waveform tensor from the vocoder.
        """
        # M = latent mel bins = mel_bins // mel_compression_ratio.
        mel_bins = int(getattr(self.audio_vae.config, "mel_bins", 64))
        mel_compression = int(getattr(self.audio_vae, "mel_compression_ratio", 4))
        latent_mel_bins = mel_bins // mel_compression

        # 1. Denormalize FIRST (on the packed latent), then unpack — order
        #    differs from video (which unpacks first).
        aud = self._denormalize_audio_latents(
            audio_latents.float(), self.audio_vae.latents_mean, self.audio_vae.latents_std
        )
        # 2. Unpack: [B, L, C*M] -> [B, C, L, M]
        aud = self._unpack_audio_latents(aud, audio_latent_length, num_mel_bins=latent_mel_bins)
        # 3. Audio VAE decode -> mel spectrogram (fp32: BigVGAN vocoder uses
        #    snake activation + Kaiser sinc filters that overflow in bf16).
        aud = aud.to(torch.float32)
        mel = self.audio_vae.to(torch.float32).decode(aud, return_dict=False)[0]
        # 4. Vocoder -> waveform (fp32 for numerical stability)
        waveform = self.vocoder.to(torch.float32)(mel)
        return waveform


__all__ = ["LTX2VAEDecodeStage", "LTX2VAEEncodeStage", "LTX2AudioDecodeStage"]
