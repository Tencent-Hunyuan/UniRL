"""Qwen-Image image adapter — packed sequence trajectory, generic schedule."""

from __future__ import annotations

from typing import List, Optional

from unirl.rollout.engine.sglang_diffusion import utils
from unirl.rollout.engine.sglang_diffusion.adapters.base import register_adapter
from unirl.rollout.engine.sglang_diffusion.adapters.image import ImageAdapter
from unirl.rollout.engine.sglang_diffusion.backends import RawResult
from unirl.types.sample import Sample
from unirl.types.sampling import DiffusionSamplingParams

_QWEN_DOWNSAMPLE = 16


@register_adapter("qwen_image")
class QwenImageAdapter(ImageAdapter):
    """Qwen-Image — packed sequence-style trajectory unpacked to true channels."""

    def build_segment(
        self,
        sample: Sample,
        results: List[RawResult],
        *,
        num_steps: int,
        sde_indices: Optional[List[int]],
        emit_native_logprob: bool,
    ):
        """Collect, unpack Qwen's packed ``[B, T, S, C*4]`` to true channels, assemble."""
        diffusion = sample.frontier_gen_part(DiffusionSamplingParams).sampling_params
        traj = utils.collect_trajectory_latents(results)
        if traj.ndim != 5:
            B, T, S, C, h_pat, w_pat = utils.validate_packed_trajectory(
                traj, diffusion, family="qwen_image", downsample=_QWEN_DOWNSAMPLE
            )
            from unirl.models.qwen_image.diffusion import _unpack_latents

            flat = traj.reshape(B * T, S, C)
            unpacked = _unpack_latents(flat, latent_h=2 * h_pat, latent_w=2 * w_pat)
            traj = unpacked.reshape(B, T, C // 4, 2 * h_pat, 2 * w_pat).contiguous()
        return utils.build_latent_segment(
            traj,
            results=results,
            expected_sigmas=diffusion.sigmas,
            num_steps=num_steps,
            sde_indices=sde_indices,
            emit_native_logprob=emit_native_logprob,
            segment_factory=self.segment_factory,
        )


__all__ = ["QwenImageAdapter"]
