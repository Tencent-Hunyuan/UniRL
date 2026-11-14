"""QwenImage21VAEDecodeStage - LatentSegment -> RGB Images via the RGBA VAE, alpha composited over white."""

from __future__ import annotations

import torch

from unirl.models.types.codec import DecodeStage
from unirl.types.primitives import Images
from unirl.types.segments import LatentSegment

from .bundle import QwenImage21Bundle


class QwenImage21VAEDecodeStage(DecodeStage[LatentSegment, Images]):
    """Qwen-Image-2.1 RGBA VAE decode stage."""

    def __init__(self, bundle: QwenImage21Bundle) -> None:
        self.bundle = bundle

    def decode(self, s: LatentSegment) -> Images:
        """Decode the final-step latents ``[N, K, 64, h, w]`` into RGB pixels ``[N, 3, 16h, 16w]`` in ``[0, 1]``."""
        if s.latents is None or s.latents.ndim != 5:
            raise ValueError(
                "QwenImage21VAEDecodeStage.decode: expected segment.latents [N, K, C, H, W], got "
                f"{None if s.latents is None else tuple(s.latents.shape)}"
            )
        vae = self.bundle.vae
        z_dim = int(vae.config.z_dim)
        clean = s.latents[:, -1].to(torch.float32).unsqueeze(2)  # [N, 64, 1, h, w]
        mean = torch.tensor(vae.config.latents_mean, device=clean.device).view(1, z_dim, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std, device=clean.device).view(1, z_dim, 1, 1, 1)
        with torch.no_grad():
            rgba = vae.decode(clean * std + mean, return_dict=False)[0][:, :, 0]  # [N, 4, H, W] in [-1, 1]
        rgba = ((rgba + 1.0) / 2.0).clamp(0.0, 1.0)
        rgb, alpha = rgba[:, :3], rgba[:, 3:]
        return Images.from_dense(rgb * alpha + (1.0 - alpha))


__all__ = ["QwenImage21VAEDecodeStage"]
