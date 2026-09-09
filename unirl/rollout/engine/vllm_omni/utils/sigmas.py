"""σ-schedule helper. Pure (torch + RL types only)."""

from __future__ import annotations

from typing import Any, List, Optional

import torch

from unirl.config.require import require


def sigmas_list_from_diffusion(diff_params: Any, num_inference_steps: int) -> Optional[List[float]]:
    """Return ``diff_params.sigmas`` as a plain ``T``-length list[float]."""
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
