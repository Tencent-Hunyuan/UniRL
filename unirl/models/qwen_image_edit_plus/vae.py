"""QwenImageEditPlus VAE stages — source-image encoder + (reused) decoder."""

from __future__ import annotations

import math
from typing import Dict, List

import torch

from unirl.models.types.codec import EncodeStage
from unirl.types.primitives import ImagePrimitive, as_image_sets

from .bundle import QwenImageEditPlusBundle
from .conditions import QwenImageEditPlusLatentCondition

# Resize source images to the upstream 1024-square grid so rollout latent shapes match.
_VAE_IMAGE_AREA = 1024 * 1024
_VAE_SIZE_ALIGN = 32  # upstream rounds to 32-pixel multiples


def _vae_size_for_aspect(width: int, height: int) -> tuple[int, int]:
    """Aspect-preserving resize target matching upstream ``VAE_IMAGE_SIZE``."""
    ratio = float(width) / float(height)
    vae_width = math.sqrt(_VAE_IMAGE_AREA * ratio)
    vae_height = vae_width / ratio
    vae_width = round(vae_width / _VAE_SIZE_ALIGN) * _VAE_SIZE_ALIGN
    vae_height = round(vae_height / _VAE_SIZE_ALIGN) * _VAE_SIZE_ALIGN
    return int(vae_width), int(vae_height)


class QwenImageEditPlusVAEEncodeStage(EncodeStage[ImagePrimitive, QwenImageEditPlusLatentCondition]):
    """Encode ordered source-image rows into VAE-latent conditions."""

    def __init__(self, bundle: QwenImageEditPlusBundle) -> None:
        self.bundle = bundle

    @torch.no_grad()
    def encode(self, images: ImagePrimitive) -> QwenImageEditPlusLatentCondition:
        """Encode source pixels into row-aligned ragged latent conditions."""
        if self.bundle.vae is None:
            raise RuntimeError(
                "QwenImageEditPlusVAEEncodeStage.encode: no VAE loaded "
                "(load_vae=False). The trainer-side pipeline cannot encode "
                "source images in this configuration — separate-engine "
                "recipes encode in the rollout engine (image_latent arrives "
                "captured); trainside rollout requires load_vae=True."
            )
        image_sets = as_image_sets(images)
        counts = image_sets.counts.tolist()
        if not counts or any(int(count) < 1 for count in counts):
            raise ValueError(
                f"QwenImageEditPlusVAEEncodeStage.encode requires at least one source image per row; counts={counts}."
            )
        flattened, owners = image_sets.flatten()
        if flattened is None:
            raise ValueError("QwenImageEditPlusVAEEncodeStage.encode: empty image-set batch")
        source_pils = flattened.to_pils()

        vae = self.bundle.vae
        device = self.bundle.device
        dtype = self.bundle.dtype
        vae_f32 = vae.to(torch.float32)

        # Stable shape buckets preserve input order while allowing each VAE
        # call to stay dense. Portrait and landscape samples never influence
        # one another's resize target.
        groups: Dict[tuple[int, int], List[int]] = {}
        for index, pil in enumerate(source_pils):
            vae_w, vae_h = _vae_size_for_aspect(*pil.size)
            groups.setdefault((vae_h, vae_w), []).append(index)

        # Per-channel normalization mirrors the upstream Edit-Plus pipeline.
        z_dim = int(vae.config.z_dim)
        latents_mean = torch.tensor(vae.config.latents_mean, device=device, dtype=torch.float32).view(1, z_dim, 1, 1)
        latents_std = torch.tensor(vae.config.latents_std, device=device, dtype=torch.float32).view(1, z_dim, 1, 1)

        import PIL.Image
        from torchvision.transforms.functional import pil_to_tensor

        by_index: Dict[int, torch.Tensor] = {}
        for (vae_h, vae_w), indices in groups.items():
            resized_items = []
            for index in indices:
                pil = source_pils[index]
                if pil.size != (vae_w, vae_h):
                    pil = pil.resize((vae_w, vae_h), PIL.Image.Resampling.LANCZOS)
                pixels = pil_to_tensor(pil).to(device=device, dtype=torch.float32).div_(255.0).unsqueeze(0)
                resized_items.append(pixels)
            pixels_batch = torch.cat(resized_items, dim=0)
            scaled_5d = (pixels_batch * 2.0 - 1.0).unsqueeze(2)
            image_latents = vae_f32.encode(scaled_5d).latent_dist.mode().squeeze(2)
            mean = latents_mean.to(image_latents.device, image_latents.dtype)
            std = latents_std.to(image_latents.device, image_latents.dtype)
            image_latents = ((image_latents - mean) / std).to(dtype=dtype)
            for local_index, source_index in enumerate(indices):
                by_index[source_index] = image_latents[local_index]

        rows: List[List[torch.Tensor]] = [[] for _ in image_sets.rows]
        for index, owner in enumerate(owners.tolist()):
            rows[int(owner)].append(by_index[index])
        return QwenImageEditPlusLatentCondition(latents=rows)


__all__ = ["QwenImageEditPlusVAEEncodeStage"]
