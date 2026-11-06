"""Z-Image (S3-DiT single-stream) image adapter — frame-axis squeeze + caption mask."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from unirl.rollout.engine.sglang_diffusion import utils
from unirl.rollout.engine.sglang_diffusion.adapters.base import register_adapter
from unirl.rollout.engine.sglang_diffusion.adapters.image import ImageAdapter
from unirl.rollout.engine.sglang_diffusion.backends import RawResult
from unirl.types.conditions.text import TextEmbedCondition
from unirl.types.sample import Sample
from unirl.types.sampling import DiffusionSamplingParams


@register_adapter("z_image")
class ZImageAdapter(ImageAdapter):
    """Z-Image S3-DiT image adapter (image-form trajectory + caption mask backfill)."""

    def build_segment(
        self,
        sample: Sample,
        results: List[RawResult],
        *,
        num_steps: int,
        sde_indices: Optional[List[int]],
        emit_native_logprob: bool,
    ):
        traj = utils.collect_trajectory_latents(results)
        if traj.ndim == 6:
            if traj.shape[3] != 1:
                raise ValueError(
                    f"z_image: trajectory has a non-singleton frame axis "
                    f"(shape={tuple(traj.shape)}); expected [B, T, C, 1, H, W] for t2i."
                )
            traj = traj.squeeze(3)
        if traj.ndim != 5:
            raise ValueError(
                f"z_image: expected a 5-D image-form trajectory [B, T, C, H, W] after the "
                f"frame squeeze; got rank {traj.ndim}, shape {tuple(traj.shape)}."
            )
        return utils.build_latent_segment(
            traj,
            results=results,
            expected_sigmas=sample.frontier_gen_part(DiffusionSamplingParams).sampling_params.sigmas,
            num_steps=num_steps,
            sde_indices=sde_indices,
            emit_native_logprob=emit_native_logprob,
            segment_factory=self.segment_factory,
        )

    def build_condition(self, results: List[RawResult]) -> Dict[str, object]:
        out = super().build_condition(results)
        for key in ("text", "negative_text"):
            text = out.get(key)
            if text is not None and text.attn_mask is None and text.embeds is not None and text.embeds.dim() == 3:
                mask = (text.embeds != 0).any(dim=-1).to(torch.long)
                out[key] = TextEmbedCondition(embeds=text.embeds, pooled=text.pooled, attn_mask=mask)
        return out


__all__ = ["ZImageAdapter"]
