"""Diffusion sampling-kwargs assembly shared by every DiT-bearing input adapter.

Moved off the adapter ABC (the bodies read no adapter state — every value
comes from the request's typed ``DiffusionSamplingParams``) so the
input-side sub-adapters, which are not ``ModelAdapter`` subclasses, can call
them as plain functions.
"""

from __future__ import annotations

from typing import Any, Dict

from unirl.rollout.engine.vllm_omni.utils.sigmas import sigmas_list_from_req
from unirl.types.rollout_req import RolloutReq


def core_diff_kwargs(req: RolloutReq, diff_params: Any) -> Dict[str, Any]:
    """The diffusion sampling kwargs common to every DiT stage.

    Every value reads off the request's typed ``DiffusionSamplingParams``
    — the engine keeps no sampling defaults. ``eta`` rides as a typed
    first-class field; ``guidance_scale_provided`` marks the explicit CFG
    choice; trajectory latents are always requested (dense — replay needs
    ``x_t`` at every slot).
    """
    num_inference_steps = int(diff_params.num_inference_steps)
    diff_kwargs: Dict[str, Any] = dict(
        height=int(diff_params.height),
        width=int(diff_params.width),
        num_inference_steps=num_inference_steps,
        guidance_scale=float(diff_params.guidance_scale),
        guidance_scale_provided=True,
        eta=float(diff_params.eta),
        return_trajectory_latents=True,
        return_trajectory_decoded=False,
        num_outputs_per_prompt=1,
    )
    sigmas = sigmas_list_from_req(req, num_inference_steps)
    if sigmas is not None:
        diff_kwargs["sigmas"] = sigmas
    return diff_kwargs


def sde_extra_args(diff_params: Any) -> Dict[str, Any]:
    """Sparse SDE step indices, normalized for the ``extra_args`` channel."""
    extra_args: Dict[str, Any] = {}
    sde_indices = getattr(diff_params, "sde_indices", None)
    if sde_indices is not None:
        extra_args["sde_indices"] = sorted({int(i) for i in sde_indices})
    return extra_args


__all__ = ["core_diff_kwargs", "sde_extra_args"]
