"""QwenImageEditPlus VAE stages — source-image encoder + (reused) decoder.

``QwenImageEditPlusVAEEncodeStage`` is the one genuinely new stage: it
turns source ``Images`` into a :class:`QwenImageEditPlusLatentCondition`
carrying one VAE-encoded spatial latent ``[16, H_i/8, W_i/8]`` per sample. The diffusion
step (:class:`QwenImageEditPlusDiffusionStep`) packs both the noise latent
and this image latent with the same 2×2 channel-pack before concatenating
along the token dimension — mirrors ``vde_editplus.py:232`` and the
FLUX.2-Klein image-edit pattern (``flux2_klein/vae.py:86-158``).

The decode side reuses :class:`unirl.models.qwen_image.QwenImageVAEDecodeStage`
unchanged (same VAE, same 5D un-normalization math) — re-exported here so
the Edit-Plus package is self-contained.
"""

from __future__ import annotations

import math
from typing import Dict, List

import torch

from unirl.models.types.codec import EncodeStage
from unirl.types.primitives import Images

from .bundle import QwenImageEditPlusBundle
from .conditions import QwenImageEditPlusLatentCondition

# Resize source images to the upstream 1024-square grid so rollout latent shapes match.
_VAE_IMAGE_AREA = 1024 * 1024
_VAE_SIZE_ALIGN = 32  # upstream rounds to 32-pixel multiples


def _vae_size_for_aspect(width: int, height: int) -> tuple[int, int]:
    """Aspect-preserving resize target matching upstream ``VAE_IMAGE_SIZE``.

    Returns ``(vae_width, vae_height)`` aligned to ``_VAE_SIZE_ALIGN`` with
    total area ≈ ``_VAE_IMAGE_AREA``. Mirrors
    ``sglang.multimodal_gen.utils.calculate_dimensions``.
    """
    ratio = float(width) / float(height)
    vae_width = math.sqrt(_VAE_IMAGE_AREA * ratio)
    vae_height = vae_width / ratio
    vae_width = round(vae_width / _VAE_SIZE_ALIGN) * _VAE_SIZE_ALIGN
    vae_height = round(vae_height / _VAE_SIZE_ALIGN) * _VAE_SIZE_ALIGN
    return int(vae_width), int(vae_height)


class QwenImageEditPlusVAEEncodeStage(EncodeStage[Images, QwenImageEditPlusLatentCondition]):
    """Encode a source image into a VAE-latent condition for token concat.

    Pipeline:

    1. Resize source pixels to the upstream ``VAE_IMAGE_SIZE`` grid
       (≈1024², aspect-preserving, 32-aligned). The data source loads
       condition images at native resolution (arbitrary H×W), but
       upstream ``QwenImageEditPlusPipelineConfig.calculate_condition_image_size``
       resizes to ``1024*1024`` total area for VAE encoding — the
       trainsite MUST match so the emitted ``image_latent`` shape is
       byte-identical to the sglang/vllm_omni rollout engines (the
       recipe YAMLs promise fixed-seed parity). The generation grid
       (e.g. 384²) is the *output* canvas size, NOT the source-image
       VAE size; using it here was a parity-breaking bug.
    2. ``[0, 1] → [-1, 1]`` (VAE input convention).
    3. Lift to 5D ``[B, 3, 1, H, W]`` (Qwen-Image VAE is a video VAE —
       ``_encode`` unpacks ``_, _, num_frame, height, width = x.shape``;
       a 4D input crashes).
    4. ``vae.encode(x).latent_dist.mode()`` — deterministic (matches
       diffusers' ``retrieve_latents(sample_mode="argmax")`` so
       rollout/replay don't drift). Mirrors ``flux2_klein/vae.py:141``.
    5. Per-channel normalize: ``(latents - latents_mean) / latents_std``
       (mirrors upstream ``QwenImageEditPlusPipeline._encode_vae_image``
       at ``pipeline_qwen_image_edit_plus.py:489-499``; the decode side
       in :class:`QwenImageVAEDecodeStage` applies the inverse, so
       skipping this would put rollout/trainsite image latents on a
       different scale than the transformer was trained on).
    6. Return one spatial latent ``[16, H_i/8, W_i/8]`` per sample wrapped in
       :class:`QwenImageEditPlusLatentCondition`. **Do NOT** ``_pack_latents`` here —
       the diffusion step packs both noise and image latents together so
       they share the same 2×2 pack logic; the condition carries the
       spatial latent (mirrors ``Flux2KleinConditions.image_latent``).
    """

    def __init__(self, bundle: QwenImageEditPlusBundle) -> None:
        self.bundle = bundle

    @torch.no_grad()
    def encode(self, images: Images) -> QwenImageEditPlusLatentCondition:
        """Encode source pixels into a ragged latent condition.

        Args:
            images: source images with one native ``[3, H_i, W_i]`` tensor per
                sample in ``[0, 1]``.

        Returns:
            :class:`QwenImageEditPlusLatentCondition` with one
            ``[16, H_vae_i/8, W_vae_i/8]`` tensor per sample.
        """
        if self.bundle.vae is None:
            raise RuntimeError(
                "QwenImageEditPlusVAEEncodeStage.encode: no VAE loaded "
                "(load_vae=False). The trainer-side pipeline cannot encode "
                "source images in this configuration — separate-engine "
                "recipes encode in the rollout engine (image_latent arrives "
                "captured); trainside rollout requires load_vae=True."
            )
        if not isinstance(images, Images):
            raise TypeError(f"QwenImageEditPlusVAEEncodeStage.encode: expected Images, got {type(images).__name__}")
        source_pils = images.to_pils()
        if not source_pils:
            raise ValueError("QwenImageEditPlusVAEEncodeStage.encode: empty image batch")

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

        return QwenImageEditPlusLatentCondition(latents=[by_index[index] for index in range(len(source_pils))])


__all__ = ["QwenImageEditPlusVAEEncodeStage"]
