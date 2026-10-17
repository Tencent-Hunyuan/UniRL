"""Qwen-Image-Edit-Plus family: image-edit modality (text+image → image)."""

from __future__ import annotations

from typing import Any, Dict, List

import torch

from unirl.models.qwen_image_edit_plus.conditions import QwenImageEditPlusLatentCondition
from unirl.rollout.engine.sglang_diffusion import utils
from unirl.rollout.engine.sglang_diffusion.adapters.base import register_adapter
from unirl.rollout.engine.sglang_diffusion.adapters.qwen_image import QwenImageAdapter
from unirl.rollout.engine.sglang_diffusion.backends import RawResult
from unirl.types.primitives import Texts
from unirl.types.sample import Sample
from unirl.types.sampling import DiffusionSamplingParams

_VAE_SCALE_FACTOR = 8


@register_adapter("qwen_image_edit_plus")
class QwenImageEditPlusAdapter(QwenImageAdapter):
    """Qwen-Image-Edit-Plus — text+image → image edit (single diffusion stage)."""

    pad_mask_to_embeds = True

    def build_prompts(self, sample: Sample) -> Dict[str, Any]:
        """Inject source-image PIL via ``condition_image`` sampling kwarg."""
        turns, image_batches = sample.vision_conditioning()
        text_turns = [turn.content for turn in turns if isinstance(turn.content, Texts)]
        if len(text_turns) != 1 or len(image_batches) != 1:
            raise ValueError(
                f"modality={self.model_family!r} requires exactly one text turn and one "
                f"image turn; got {len(text_turns)} text and {len(image_batches)} image."
            )
        gen_part = sample.frontier_gen_part(DiffusionSamplingParams)
        prompts = text_turns[0].texts
        unique_prompts, k = utils.deexpand_prompts_from_groups(prompts, gen_part.group_ids)
        images_prim = image_batches[0]
        pil_images = images_prim.to_pils()
        unique_pils = utils.first_per_group(pil_images, gen_part.group_ids) if k > 1 else pil_images
        out: Dict[str, Any] = {
            "prompt": unique_prompts if len(unique_prompts) > 1 else unique_prompts[0],
            "condition_image": unique_pils if len(unique_pils) > 1 else unique_pils[0],
        }
        if k > 1:
            out["num_outputs_per_prompt"] = k
        return out

    def build_condition(self, results: List[RawResult]) -> Dict[str, Any]:
        """T2I text-capture conditions + Edit-Plus ``image_latent``."""
        from unirl.models.qwen_image.diffusion import _unpack_latents

        cond_dict = super().build_condition(results)
        tensors: List[torch.Tensor] = []
        for r in results:
            packed_list = r.image_latent
            sizes_list = r.image_latent_sizes
            if not packed_list or not sizes_list:
                raise RuntimeError(
                    "build_condition: Qwen-Image-Edit-Plus rollout returned no "
                    "image_latent/image_latent_sizes. Check that patch_conditions "
                    "captured batch.image_latent (set by ImageVAEEncodingStage) "
                    "— the image_latent capture is required for trainer-side "
                    "replay (predict_noise concatenates it onto the noise latent)."
                )
            packed = packed_list[0]
            sizes = sizes_list[0]
            if len(sizes) != 1:
                raise NotImplementedError(
                    f"build_condition: multi-image Edit-Plus not supported (got {len(sizes)} source images per prompt)."
                )
            vae_width, vae_height = sizes[0]
            latent_h = vae_height // _VAE_SCALE_FACTOR
            latent_w = vae_width // _VAE_SCALE_FACTOR
            spatial = _unpack_latents(packed, latent_h=latent_h, latent_w=latent_w)
            tensors.append(spatial.squeeze(0))
        cond_dict["image_latent"] = QwenImageEditPlusLatentCondition(latents=tensors)
        return cond_dict


__all__ = ["QwenImageEditPlusAdapter"]
