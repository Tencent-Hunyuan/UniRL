"""FastVideo-specific sigma schedule adaptation and validation."""

from __future__ import annotations

import math
from typing import Any, List

import torch

from unirl.config.require import require
from unirl.rollout.engine.sigma_verify import verify_engine_used_sigmas

_FASTVIDEO_TIMESTEP_SCALE = 1000


def shift_preimage_sigmas(
    sigmas: torch.Tensor,
    shift: float,
    *,
    num_inference_steps: int,
) -> List[float]:
    """Invert FastVideo's static flow shift and drop the terminal sigma."""
    require(math.isfinite(shift) and shift > 0.0, f"FastVideo flow_shift must be finite and > 0; got {shift!r}")
    require(
        int(num_inference_steps) > 0,
        f"FastVideo num_inference_steps must be positive; got {num_inference_steps!r}",
    )

    sigmas_f64 = sigmas.detach().cpu().double()
    require(
        sigmas_f64.ndim == 1,
        f"FastVideo requested sigmas must be a one-dimensional [T+1] schedule; got {tuple(sigmas_f64.shape)}",
    )
    expected_values = int(num_inference_steps) + 1
    require(
        sigmas_f64.numel() == expected_values,
        f"FastVideo requested sigmas must contain num_inference_steps + 1 = {expected_values} values; "
        f"got {sigmas_f64.numel()}",
    )
    require(bool(torch.isfinite(sigmas_f64).all()), "FastVideo requested sigmas must all be finite")
    require(
        bool(torch.all((sigmas_f64 >= 0.0) & (sigmas_f64 <= 1.0))),
        "FastVideo requested sigmas must be normalized to [0, 1]",
    )
    require(
        bool(torch.all(sigmas_f64[:-1] >= sigmas_f64[1:])),
        "FastVideo requested sigmas must be monotonically non-increasing",
    )
    require(float(sigmas_f64[-1].item()) == 0.0, "FastVideo requested sigmas must end at terminal sigma 0")

    denominator = shift - sigmas_f64 * (shift - 1.0)
    require(
        bool(torch.isfinite(denominator).all() and torch.all(denominator != 0)),
        "FastVideo flow-shift inversion has a non-finite or zero denominator",
    )
    preimage = sigmas_f64 / denominator
    require(bool(torch.isfinite(preimage).all()), "FastVideo flow-shift pre-image sigmas must all be finite")
    return [float(x) for x in preimage.tolist()[:-1]]


def verify_fastvideo_used_sigmas(
    trajectory_timesteps: Any,
    *,
    expected: torch.Tensor,
    sample_index: int,
) -> None:
    """Adapt FastVideo's scaled ``[T]`` timesteps to the shared ``[T+1]`` verifier.

    FastVideo's default WAN ``FlowUniPCMultistepScheduler`` converts
    ``sigma * 1000`` to ``int64``. Its trajectory echo therefore contains the
    truncated integer model-conditioning timestep, not the scheduler's original
    floating-point sigma. Quantize the expected schedule by the same rule so the
    comparison remains exact at the observable wire representation; do not
    merely widen the verifier tolerance, which would also hide a one-tick drift.
    """
    actual = trajectory_timesteps
    expected_for_verification = expected
    if actual is not None:
        actual_t = actual.detach().cpu() if torch.is_tensor(actual) else torch.as_tensor(actual)
        if actual_t.ndim == 1:
            actual = torch.cat([actual_t, actual_t.new_zeros(1)])
        if not torch.is_floating_point(actual_t):
            expected_f32 = expected.detach().cpu().to(torch.float32)
            expected_for_verification = (
                torch.trunc(expected_f32 * _FASTVIDEO_TIMESTEP_SCALE) / _FASTVIDEO_TIMESTEP_SCALE
            )
    verify_engine_used_sigmas(
        actual,
        expected=expected_for_verification,
        engine_name=f"fastvideo (sample {sample_index})",
    )


__all__ = ["shift_preimage_sigmas", "verify_fastvideo_used_sigmas"]
