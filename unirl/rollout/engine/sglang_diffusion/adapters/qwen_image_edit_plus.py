"""Qwen-Image-Edit-Plus family: image-edit modality (text+image → image)."""

from __future__ import annotations

from typing import Any, Dict, List

import torch

from unirl.models.qwen_image_edit_plus.conditions import QwenImageEditPlusLatentCondition
from unirl.rollout.engine.sglang_diffusion import utils
from unirl.rollout.engine.sglang_diffusion.adapters.base import register_adapter
from unirl.rollout.engine.sglang_diffusion.adapters.qwen_image import QwenImageAdapter
from unirl.rollout.engine.sglang_diffusion.backends import RawResult
from unirl.types.primitives import Texts, as_image_sets
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
        prompts = list(text_turns[0].texts)
        unique_prompts, k = utils.deexpand_prompts_from_groups(prompts, list(gen_part.group_ids))
        image_sets = as_image_sets(image_batches[0])
        if len(image_sets) != len(prompts):
            raise ValueError(
                f"{self.model_family}.build_prompts: image row count {len(image_sets)} != prompt count {len(prompts)}."
            )
        pil_rows = image_sets.to_pil_rows()
        counts = [len(row) for row in pil_rows]
        if any(count < 1 for count in counts):
            raise ValueError(f"{self.model_family}.build_prompts requires non-empty image rows; counts={counts}.")
        unique_rows = utils.first_per_group(pil_rows, list(gen_part.group_ids)) if k > 1 else pil_rows
        condition_image: Any
        if len(unique_rows) > 1:
            condition_image = unique_rows
        else:
            condition_image = unique_rows[0][0] if len(unique_rows[0]) == 1 else unique_rows[0]
        out: Dict[str, Any] = {
            "prompt": unique_prompts if len(unique_prompts) > 1 else unique_prompts[0],
            "condition_image": condition_image,
        }
        if k > 1:
            out["num_outputs_per_prompt"] = k
        return out

    def build_condition(self, results: List[RawResult]) -> Dict[str, Any]:
        """T2I text-capture conditions + Edit-Plus ``image_latent``."""
        cond_dict = super().build_condition(results)
        cond_dict["image_latent"] = QwenImageEditPlusLatentCondition(latents=self._collect_image_latents(results))
        return cond_dict

    def _collect_image_latents(self, results: List[RawResult]) -> List[List[torch.Tensor]]:
        """Rebuild ordered per-result source latents without forcing shared grids."""
        from unirl.models.qwen_image.diffusion import _unpack_latents

        rows: List[List[torch.Tensor]] = []
        for r in results:
            packed_list = getattr(r, "image_latent", None)
            sizes_list = getattr(r, "image_latent_sizes", None)
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
            if not sizes:
                raise RuntimeError("build_condition: Edit-Plus rollout returned an empty source-image size list.")
            offset = 0
            row: List[torch.Tensor] = []
            for vae_width, vae_height in sizes:
                latent_h = int(vae_height) // _VAE_SCALE_FACTOR
                latent_w = int(vae_width) // _VAE_SCALE_FACTOR
                token_count = (latent_h // 2) * (latent_w // 2)
                chunk = packed[:, offset : offset + token_count]
                if int(chunk.shape[1]) != token_count:
                    raise RuntimeError(
                        "build_condition: captured Edit-Plus image_latent is shorter than vae_image_sizes "
                        f"requires at source {len(row)} ({int(chunk.shape[1])} != {token_count})."
                    )
                row.append(_unpack_latents(chunk, latent_h=latent_h, latent_w=latent_w).squeeze(0))
                offset += token_count
            if offset != int(packed.shape[1]):
                raise RuntimeError(
                    "build_condition: captured Edit-Plus image_latent has trailing tokens after splitting "
                    f"{len(sizes)} sources ({int(packed.shape[1]) - offset} extra)."
                )
            rows.append(row)
        return rows


__all__ = ["QwenImageEditPlusAdapter"]
