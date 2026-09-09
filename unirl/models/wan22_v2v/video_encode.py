"""WAN22VideoLatentEncodeStage — input ``Videos`` → clean 3D VAE latents."""

import torch

from unirl.models.wan21.vae import WANVideoLatentEncodeStage


class WAN22VideoLatentEncodeStage(WANVideoLatentEncodeStage):
    """Encode V2V input videos into normalized clean WAN VAE latents."""

    def _postprocess_resized_frames(self, frames: torch.Tensor) -> torch.Tensor:
        return frames

    def _postprocess_latents(self, latents: torch.Tensor) -> torch.Tensor:
        vae = self.bundle.vae
        vae_config = vae.config
        latents_mean = getattr(vae_config, "latents_mean", None)
        latents_std = getattr(vae_config, "latents_std", None)
        if latents_mean is not None and latents_std is not None:
            z_dim = int(getattr(vae_config, "z_dim", latents.shape[1]))
            mean = torch.tensor(latents_mean, device=latents.device, dtype=latents.dtype).view(1, z_dim, 1, 1, 1)
            std = torch.tensor(latents_std, device=latents.device, dtype=latents.dtype).view(1, z_dim, 1, 1, 1)
            latents = (latents - mean) / std
        else:
            scaling_factor = float(getattr(vae_config, "scaling_factor", 1.0))
            latents = latents * scaling_factor
        return latents


__all__ = ["WAN22VideoLatentEncodeStage"]
