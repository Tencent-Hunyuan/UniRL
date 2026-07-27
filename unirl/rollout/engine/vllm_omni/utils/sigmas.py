"""σ-schedule helper. Pure (torch + RL types only)."""

from __future__ import annotations

from typing import Any, List, Optional

import torch

from unirl.config.require import require


def sigmas_list_from_diffusion(diff_params: Any, num_inference_steps: int) -> Optional[List[float]]:
    """Return ``diff_params.sigmas`` as a plain ``T``-length list[float].

    Worker side (upstream pipeline_sd3 / pipeline_hunyuan_image3) routes a
    non-None ``sampling_params.sigmas`` into the scheduler via
    ``retrieve_timesteps`` → ``set_timesteps(sigmas=...)``. We send the
    schedule the trainer will replay against (the engine-pinned
    ``DiffusionSamplingParams.sigmas``) so worker and replay use identical σ.
    ``None`` falls back to the worker's internal schedule (legacy behavior, kept
    for paths that bypass the engine's σ pin).

    **Shape contract: send ``T`` values, not ``T+1``.** ``sigmas`` is
    canonically ``T+1`` (terminal 0 included), but diffusers'
    ``set_timesteps(sigmas=...)`` takes ``len(sigmas)`` as
    ``num_inference_steps`` and appends a terminal 0 itself. Sending ``T+1``
    would run one extra loop iteration and leave ``scheduler.sigmas`` at
    ``T+2``. Matches the SGLang adapters (they also slice ``[:-1]``).
    """
    sigmas = diff_params.sigmas
    if sigmas is None:
        return None
    require(
        int(sigmas.shape[0]) == num_inference_steps + 1,
        f"diffusion.sigmas length {int(sigmas.shape[0])} != "
        f"num_inference_steps+1 ({num_inference_steps + 1}). Engine must "
        f"populate σ for the resolved num_inference_steps.",
    )
    return sigmas.detach().to(torch.float32).cpu().tolist()[:-1]


__all__ = ["sigmas_list_from_diffusion"]
