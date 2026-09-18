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


def _to_image_form_trajectory(traj: torch.Tensor) -> torch.Tensor:
    """Squeeze Z-Image's singleton frame axis: 6-D ``[B,T,C,1,H,W]`` -> 5-D ``[B,T,C,H,W]``."""
    if traj.ndim == 6:
        if int(traj.shape[3]) != 1:
            raise ValueError(
                f"z_image: trajectory has a non-singleton frame axis "
                f"(shape={tuple(traj.shape)}); expected [B, T, C, 1, H, W] for t2i."
            )
        return traj.squeeze(3)
    return traj


def _backfill_caption_mask(text: Optional[TextEmbedCondition]) -> Optional[TextEmbedCondition]:
    """Recover a per-token mask from non-zero rows — token-level ``[B, T, D]`` embeds with no ``attn_mask`` only."""
    if text is None or text.attn_mask is not None or text.embeds is None or text.embeds.dim() != 3:
        return text
    mask = (text.embeds != 0).any(dim=-1).to(torch.long)
    return TextEmbedCondition(embeds=text.embeds, pooled=text.pooled, attn_mask=mask)


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
        traj = _to_image_form_trajectory(utils.collect_trajectory_latents(results))
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
            if key in out:
                out[key] = _backfill_caption_mask(out[key])
        return out


__all__ = ["ZImageAdapter"]
