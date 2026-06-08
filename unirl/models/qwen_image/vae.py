"""QwenImageVAEDecodeStage — LatentSegment → Images via VAE decode.

Implements ``DecodeStage[LatentSegment, Images]``. Reads the final
stored position from ``LatentSegment.latents[:, -1]``
(``QwenImageDiffusionStage`` always stores position ``T``, the clean
latent), runs the per-channel un-normalization Qwen-Image's VAE
expects, lifts the spatial latent into the VAE's 5D
``[B, C, T=1, H, W]`` input shape, decodes in fp32, then normalizes
the output pixels from ``[-1, 1]`` to ``[0, 1]``.

No ``QwenImageVAEEncodeStage`` here — PR #104 supports only t2i
(``models/qwen_image.py:367-370`` rejects ``image=`` and ``video=``
inputs); the encoder is unused. Add when img2img / SDEdit /
ControlNet lands.

Per-channel normalization math mirrors PR #104's ``decode_latents``.
"""

from __future__ import annotations

import torch

from unirl.models.types.codec import DecodeStage
from unirl.types.primitives import Images
from unirl.types.segments import LatentSegment

from .bundle import QwenImageBundle


class QwenImageVAEDecodeStage(DecodeStage[LatentSegment, Images]):
    """Qwen-Image VAE decode stage."""

    def __init__(self, bundle: QwenImageBundle) -> None:
        self.bundle = bundle

    def decode(self, s: LatentSegment) -> Images:
        """Decode the final-step latents in *s* into pixel images.

        Reads ``s.latents[:, -1]`` (the final stored position, which is
        ``T`` — the clean latent ``x_0`` in spatial shape
        ``[B, C, H, W]``). Lifts to 5D for the video VAE, applies the
        per-channel un-normalization from ``vae.config.latents_mean`` /
        ``latents_std``, decodes in fp32, then normalizes pixels to
        ``[0, 1]``.
        """
        if s.latents is None:
            raise ValueError("QwenImageVAEDecodeStage.decode: segment.latents is None")
        if s.latents.ndim < 5:
            raise ValueError(
                f"QwenImageVAEDecodeStage.decode: expected latents shape [N, K, C, H, W], got {tuple(s.latents.shape)}"
            )
        clean = s.latents[:, -1]  # [B, C, H, W]

        vae = self.bundle.vae
        z_dim = int(vae.config.z_dim)
        device = clean.device

        with torch.no_grad():
            latents_f32 = clean.to(dtype=torch.float32)
            # Lift to 5D for the video VAE (T=1).
            latents_5d = latents_f32.unsqueeze(2)  # [B, C, 1, H, W]
            latents_mean = torch.tensor(vae.config.latents_mean, device=device, dtype=torch.float32).view(
                1, z_dim, 1, 1, 1
            )
            latents_std = torch.tensor(vae.config.latents_std, device=device, dtype=torch.float32).view(
                1, z_dim, 1, 1, 1
            )
            # Recover raw latents: x = z * std + mean.
            latents_5d = latents_5d * latents_std + latents_mean
            decoded = vae.to(torch.float32).decode(latents_5d, return_dict=False)[0]
        # Drop the temporal dim (Qwen-Image t2i uses T=1) and clamp.
        pixels = ((decoded[:, :, 0] + 1.0) / 2.0).clamp(0.0, 1.0)
        return Images(pixels=pixels)


__all__ = ["QwenImageVAEDecodeStage"]
