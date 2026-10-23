"""WAN22VideoLatentEncodeStage — input ``Videos`` → clean 3D VAE latents."""

import torch

from unirl.models.wan21.vae import WANVideoLatentEncodeStage


class WAN22VideoLatentEncodeStage(WANVideoLatentEncodeStage):
    """Encode V2V input videos into normalized clean WAN VAE latents."""

    def _postprocess_resized_frames(self, frames: torch.Tensor) -> torch.Tensor:
        return frames

    def _postprocess_latents(self, latents: torch.Tensor) -> torch.Tensor:
        vae_config = self.bundle.vae.config
        z_dim = vae_config.z_dim
        mean = torch.tensor(vae_config.latents_mean, device=latents.device, dtype=latents.dtype).view(1, z_dim, 1, 1, 1)
        std = torch.tensor(vae_config.latents_std, device=latents.device, dtype=latents.dtype).view(1, z_dim, 1, 1, 1)
        return (latents - mean) / std


__all__ = ["WAN22VideoLatentEncodeStage"]
