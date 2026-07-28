"""FLUX-family image adapters: plain FLUX + FLUX.2-Klein.

``FluxAdapter`` is the default image path (5-D passthrough). ``Flux2KleinAdapter``
overrides ``build_segment`` (Klein emits packed ``[B, T, H*W, C]`` tokens, unpacked
to image form before assembly) and the schedule policy, and serves BOTH t2i and
text+image→image off one checkpoint — branching on ``Sample.has_image_input()``,
like the trainside ``Flux2KleinPipeline``. The ti2i deltas are in ``build_prompts`` /
``build_condition`` and no-op for t2i.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch

from unirl.config.require import require
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


# FLUX.2 patchified spatial size: pixel / (vae_scale_factor=8 * patchify_factor=2).
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

    # ---- Request side: ti2i delta (no-op without an image) ---- #

    def build_prompts(self, sample: Sample) -> Dict[str, Any]:
        """T2I payload, plus the source image when the request carries one.

        Without an image, the inherited T2I payload verbatim. With one, the PIL
        rides the ``condition_image`` sampling kwarg (``patch_sampling_io`` indexes
        it per prompt); upstream VAE-encodes it into ``image_latent`` +
        ``condition_image_latent_ids``.
        """
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
        unique_prompts, k = self._deexpand_prompts(prompts, gen_part.group_ids)
        pil_images = image_batches[0].to_pils()
        if len(pil_images) != len(prompts):
            raise ValueError(f"image count {len(pil_images)} != prompt count {len(prompts)}")
        # One source image per group, first-seen order (mirrors the prompt collapse;
        # ``pil_images[::k]`` would misalign interleaved group_ids).
        if k > 1:
            unique_pils = self._first_per_group(pil_images, list(gen_part.group_ids))
            if len(unique_pils) != len(unique_prompts):
                raise ValueError(f"collapsed image count {len(unique_pils)} != unique prompts {len(unique_prompts)}")
        else:
            unique_pils = pil_images
        out: Dict[str, Any] = {
            "prompt": unique_prompts if len(unique_prompts) > 1 else unique_prompts[0],
            "condition_image": unique_pils if len(unique_pils) > 1 else unique_pils[0],
        }
        if k > 1:
            out["num_outputs_per_prompt"] = k
        return out

    # ---- Response side ---- #

    def build_condition(self, results: List[RawResult]) -> Dict[str, Any]:
        """Inherited text conditions, plus the ti2i source-image slots.

        Emits ``image_latent`` (packed ``[B, N, 128]`` tokens) and ``image_latent_ids``
        (4-axis RoPE ids ``[B, N, 4]``); ``predict_noise`` needs both. The ids are
        captured, not recomputed — ``N`` alone does not give the ``h_pat × w_pat`` grid
        and FLUX.2 never sets ``vae_image_sizes``. Both ``None`` for pure T2I.
        """
        cond_dict = super().build_condition(results)
        tokens = self._stack_condition_field(results, "image_latent")
        if tokens is None:
            return cond_dict  # pure T2I
        ids = self._stack_condition_field(results, "condition_image_latent_ids")
        require(ids is not None, "ti2i returned image_latent but no condition_image_latent_ids; replay needs both.")
        cond_dict["image_latent"] = ImageLatentCondition(latents=tokens)
        cond_dict["image_latent_ids"] = ImageLatentCondition(latents=ids)
        return cond_dict

    def _stack_condition_field(self, results: List[RawResult], name: str) -> Optional[torch.Tensor]:
        """Concat a per-result single-tensor conditions field over dim 0.

        ``patch_conditions`` ships each field as a one-element list holding that
        result's ``[1, N, ...]`` slice. ``None`` when no result carries it (T2I).
        """
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
            f"{name} has mixed shapes {sorted(shapes)} — bucket by aspect ratio or normalize the dataset.",
        )
        return torch.cat(tensors, dim=0)

    @staticmethod
    def _first_per_group(items: List[Any], group_ids: List[str]) -> List[Any]:
        """First item of each group, in first-seen group order."""
        seen: set[str] = set()
        out: List[Any] = []
        for item, gid in zip(items, group_ids):
            if gid not in seen:
                seen.add(gid)
                out.append(item)
        return out

    def _deexpand_prompts(self, prompts: List[str], group_ids: List[str]):
        """Collapse K-expanded prompts back to unique + repeat count."""
        return utils.deexpand_prompts_from_groups(prompts, list(group_ids))

    # ---- Segment (inherited shape, Klein packed-token unpack) ---- #

    def build_segment(
        self,
        sample: Sample,
        results: List[RawResult],
        *,
        num_steps: int,
        sde_indices: Optional[List[int]],
        emit_native_logprob: bool,
    ):
        """Collect, unpack Klein's packed ``[B, T, H*W, C]`` to image form, assemble.

        5-D arrivals (image-form) skip the unpack. Condition tokens never enter the
        recorded trajectory (SGLang keeps them in a separate ``latent_model_input``),
        so this is identical for t2i and ti2i.
        """
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
