"""Diffusion sampling-kwargs assembly shared by every DiT-bearing input adapter."""

from __future__ import annotations

from typing import Any, Dict

from unirl.rollout.engine.vllm_omni.utils.sigmas import sigmas_list_from_diffusion


def core_diff_kwargs(diff_params: Any) -> Dict[str, Any]:
    """The diffusion sampling kwargs common to every DiT stage."""
    num_inference_steps = diff_params.num_inference_steps
    diff_kwargs: Dict[str, Any] = dict(
        height=diff_params.height,
        width=diff_params.width,
        num_inference_steps=num_inference_steps,
        guidance_scale=diff_params.guidance_scale,
        guidance_scale_provided=True,
        eta=diff_params.eta,
        return_trajectory_latents=True,
        return_trajectory_decoded=False,
        num_outputs_per_prompt=1,
    )
    sigmas = sigmas_list_from_diffusion(diff_params, num_inference_steps)
    if sigmas is not None:
        diff_kwargs["sigmas"] = sigmas
    return diff_kwargs


def sde_extra_args(diff_params: Any) -> Dict[str, Any]:
    """Sparse SDE step indices, normalized for the ``extra_args`` channel."""
    extra_args: Dict[str, Any] = {}
    sde_indices = diff_params.sde_indices
    if sde_indices is not None:
        extra_args["sde_indices"] = sorted(set(sde_indices))
    return extra_args


__all__ = ["core_diff_kwargs", "sde_extra_args"]
