"""FLUX-family image adapters: plain FLUX + FLUX.2-Klein."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from unirl.config.require import require
from unirl.models.flux2_klein.image import resize_condition_pils
from unirl.rollout.engine.sglang_diffusion import utils
from unirl.rollout.engine.sglang_diffusion.adapters.base import register_adapter
from unirl.rollout.engine.sglang_diffusion.adapters.image import ImageAdapter
from unirl.rollout.engine.sglang_diffusion.backends import RawResult
from unirl.types.conditions.image import ImageLatentCondition
from unirl.types.primitives import Texts
from unirl.types.sample import Sample
from unirl.types.sampling import DiffusionSamplingParams


@register_adapter("flux")
class FluxAdapter(ImageAdapter):
    """FLUX — image-form 5-D trajectory throughout; default path."""

    pass


_KLEIN_DOWNSAMPLE = 16


@register_adapter("flux2_klein")
class Flux2KleinAdapter(ImageAdapter):
    """FLUX.2-Klein — packed sequence-style trajectory + model-specific schedule."""

    def validate(self) -> None:
        super().validate()
        require(
            callable(getattr(self.model_config, "build_schedule_policy", None)),
            "flux2_klein adapter requires model_config.build_schedule_policy() (model-specific compute_mu).",
        )

    def schedule_policy(self):
        return self.model_config.build_schedule_policy()

    def build_prompts(self, sample: Sample) -> Dict[str, Any]:
        """T2I payload, plus the source image when the request carries one."""
        if not sample.has_image_input():
            return super().build_prompts(sample)

        turns, image_batches = sample.vision_conditioning()
        text_turns = [turn.content for turn in turns if isinstance(turn.content, Texts)]
        if len(text_turns) != 1 or len(image_batches) != 1:
            raise ValueError(
                f"{self.model_family!r} ti2i needs exactly 1 text and 1 image turn; "
                f"got {len(text_turns)} text, {len(image_batches)} image."
            )
        gen_part = sample.frontier_gen_part(DiffusionSamplingParams)
        prompts = list(text_turns[0].texts)
        unique_prompts, k = utils.deexpand_prompts_from_groups(prompts, list(gen_part.group_ids))
        pil_images = image_batches[0].to_pils()
        unique_pils = utils.first_per_group(pil_images, list(gen_part.group_ids)) if k > 1 else pil_images
        unique_pils = resize_condition_pils(
            unique_pils,
            height=int(gen_part.sampling_params.height),
            width=int(gen_part.sampling_params.width),
        )
        out: Dict[str, Any] = {
            "prompt": unique_prompts if len(unique_prompts) > 1 else unique_prompts[0],
            "condition_image": unique_pils if len(unique_pils) > 1 else unique_pils[0],
        }
        if k > 1:
            out["num_outputs_per_prompt"] = k
        return out

    def build_condition(self, results: List[RawResult]) -> Dict[str, Any]:
        """Inherited text conditions, plus the ti2i source-image slots."""
        cond_dict = super().build_condition(results)
        tokens = self._concat_condition_field(results, "image_latent")
        if tokens is None:
            return cond_dict
        ids = self._concat_condition_field(results, "condition_image_latent_ids")
        require(ids is not None, "ti2i returned image_latent but no condition_image_latent_ids; replay needs both.")
        cond_dict["image_latent"] = ImageLatentCondition(latents=tokens)
        cond_dict["image_latent_ids"] = ImageLatentCondition(latents=ids)
        return cond_dict

    def _concat_condition_field(self, results: List[RawResult], name: str) -> Optional[torch.Tensor]:
        """Concat a per-result single-tensor conditions field over dim 0."""
        tensors: List[torch.Tensor] = []
        for r in results:
            value = getattr(r, name, None)
            if not value:
                continue
            tensors.append(value[0])
        if not tensors:
            return None
        require(
            len(tensors) == len(results),
            f"{name} on {len(tensors)}/{len(results)} results — expected all or none.",
        )
        shapes = {tuple(t.shape) for t in tensors}
        require(
            len(shapes) == 1,
            f"{name} has mixed shapes {sorted(shapes)} after fixed-size condition preprocessing.",
        )
        return torch.cat(tensors, dim=0)

    def build_segment(
        self,
        sample: Sample,
        results: List[RawResult],
        *,
        num_steps: int,
        sde_indices: Optional[List[int]],
        emit_native_logprob: bool,
    ):
        """Collect, unpack Klein's packed ``[B, T, H*W, C]`` to image form, assemble."""
        diffusion = sample.frontier_gen_part(DiffusionSamplingParams).sampling_params
        traj = utils.collect_trajectory_latents(results)
        if traj.ndim != 5:
            B, T, S, C, h_pat, w_pat = utils.validate_packed_trajectory(
                traj, diffusion, family="flux2_klein", downsample=_KLEIN_DOWNSAMPLE, require_divisible=True
            )
            from unirl.models.flux2_klein.flux2_klein_utils import unpack_latents

            flat = traj.reshape(B * T, S, C)
            traj = unpack_latents(flat, h_pat, w_pat).reshape(B, T, C, h_pat, w_pat).contiguous()
        return utils.build_latent_segment(
            traj,
            results=results,
            expected_sigmas=diffusion.sigmas,
            num_steps=num_steps,
            sde_indices=sde_indices,
            emit_native_logprob=emit_native_logprob,
            segment_factory=self.segment_factory,
        )


__all__ = ["FluxAdapter", "Flux2KleinAdapter"]
