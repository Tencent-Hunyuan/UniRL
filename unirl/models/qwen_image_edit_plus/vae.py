"""QwenImageEditPlus VAE stages — source-image encoder + (reused) decoder.

``QwenImageEditPlusVAEEncodeStage`` is the one genuinely new stage: it
turns a source ``Images`` primitive into an :class:`ImageLatentCondition`
carrying the VAE-encoded spatial latent ``[B, 16, H/8, W/8]``. The diffusion
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

import torch

from unirl.models.types.codec import EncodeStage
from unirl.types.conditions import ImageLatentCondition
from unirl.types.primitives import Images

from .bundle import QwenImageEditPlusBundle

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


class QwenImageEditPlusVAEEncodeStage(EncodeStage[Images, ImageLatentCondition]):
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
    6. Return the spatial latent ``[B, 16, H/8, W/8]`` wrapped in an
       :class:`ImageLatentCondition`. **Do NOT** ``_pack_latents`` here —
       the diffusion step packs both noise and image latents together so
       they share the same 2×2 pack logic; the condition carries the
       spatial latent (mirrors ``Flux2KleinConditions.image_latent``).
    """

    def __init__(self, bundle: QwenImageEditPlusBundle) -> None:
        self.bundle = bundle

    @torch.no_grad()
    def encode(self, images: Images, *, height: int, width: int) -> ImageLatentCondition:
        """Encode source pixels into an :class:`ImageLatentCondition`.

        Args:
            images: source ``Images`` with ``pixels`` ``[B, 3, H, W]`` in
                ``[0, 1]``.
            height, width: generation grid (must be divisible by 16: 8× VAE
                downsample + 2× patchify). Used only for the 16-alignment
                guard; the source image is resized to the upstream
                ``VAE_IMAGE_SIZE`` grid (≈1024²), NOT to the generation
                grid, matching sglang/vllm_omni rollout engines.

        Returns:
            :class:`ImageLatentCondition` with ``latents`` shape
            ``[B, 16, H_vae/8, W_vae/8]`` on the bundle device, in the
            bundle dtype.
        """
        if self.bundle.vae is None:
            raise RuntimeError(
                "QwenImageEditPlusVAEEncodeStage.encode: no VAE loaded "
                "(load_vae=False). The trainer-side pipeline cannot encode "
                "source images in this configuration — separate-engine "
                "recipes encode in the rollout engine (image_latent arrives "
                "captured); trainside rollout requires load_vae=True."
            )
        pixels = images.pixels
        if pixels is None or pixels.ndim != 4 or pixels.shape[1] != 3:
            raise ValueError(
                f"QwenImageEditPlusVAEEncodeStage.encode: expected pixels "
                f"[B, 3, H, W] in [0,1], got shape "
                f"{None if pixels is None else tuple(pixels.shape)}"
            )
        if int(height) % 16 != 0 or int(width) % 16 != 0:
            raise ValueError(
                f"QwenImageEditPlusVAEEncodeStage.encode: height ({height}) and "
                f"width ({width}) must be divisible by 16 (8× VAE + 2× patchify)"
            )

        vae = self.bundle.vae
        device = self.bundle.device
        dtype = self.bundle.dtype
        vae_f32 = vae.to(torch.float32)

        pixels = pixels.to(device=device, dtype=torch.float32)
        src_h = int(pixels.shape[-2])
        src_w = int(pixels.shape[-1])
        vae_w, vae_h = _vae_size_for_aspect(src_w, src_h)
        if src_h != vae_h or src_w != vae_w:
            pixels = torch.nn.functional.interpolate(pixels, size=(vae_h, vae_w), mode="bilinear", align_corners=False)

        scaled = pixels * 2.0 - 1.0

        scaled_5d = scaled.unsqueeze(2)

        image_latents = vae_f32.encode(scaled_5d).latent_dist.mode()

        image_latents = image_latents.squeeze(2)

        z_dim = int(vae.config.z_dim)
        latents_mean = (
            torch.tensor(vae.config.latents_mean, device=device, dtype=torch.float32)
            .view(1, z_dim, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )
        latents_std = (
            torch.tensor(vae.config.latents_std, device=device, dtype=torch.float32)
            .view(1, z_dim, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )
        image_latents = (image_latents - latents_mean) / latents_std

        return ImageLatentCondition(latents=image_latents.to(dtype=dtype))


__all__ = ["QwenImageEditPlusVAEEncodeStage"]
