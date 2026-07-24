from __future__ import annotations

import math

import pytest
import torch

from unirl.rollout.engine.fastvideo.sigma import shift_preimage_sigmas, verify_fastvideo_used_sigmas


def test_shift_preimage_round_trips_fastvideo_static_shift() -> None:
    requested = torch.tensor([1.0, 0.75, 0.25, 0.0], dtype=torch.float64)
    shift = 3.0

    wire = torch.tensor(
        shift_preimage_sigmas(requested, shift, num_inference_steps=3),
        dtype=torch.float64,
    )
    reconstructed = shift * wire / (1.0 + (shift - 1.0) * wire)

    torch.testing.assert_close(reconstructed, requested[:-1], rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("shift", [0.0, -1.0, math.inf, math.nan])
def test_shift_preimage_rejects_invalid_shift(shift: float) -> None:
    requested = torch.tensor([1.0, 0.5, 0.0])

    with pytest.raises(ValueError, match="flow_shift must be finite and > 0"):
        shift_preimage_sigmas(requested, shift, num_inference_steps=2)


@pytest.mark.parametrize(
    ("requested", "num_steps", "message"),
    [
        (torch.tensor([[1.0, 0.5, 0.0]]), 2, "one-dimensional"),
        (torch.tensor([1.0, 0.5, 0.0]), 3, r"num_inference_steps \+ 1"),
        (torch.tensor([1.0, math.nan, 0.0]), 2, "all be finite"),
        (torch.tensor([1.0, 1.1, 0.0]), 2, r"normalized to \[0, 1\]"),
        (torch.tensor([1.0, 0.25, 0.5, 0.0]), 3, "monotonically non-increasing"),
        (torch.tensor([1.0, 0.5, 0.1]), 2, "terminal sigma 0"),
    ],
)
def test_shift_preimage_rejects_invalid_requested_schedule(
    requested: torch.Tensor,
    num_steps: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        shift_preimage_sigmas(requested, 3.0, num_inference_steps=num_steps)


def test_shift_preimage_rejects_non_positive_step_count() -> None:
    with pytest.raises(ValueError, match="num_inference_steps must be positive"):
        shift_preimage_sigmas(torch.tensor([0.0]), 3.0, num_inference_steps=0)


def test_verify_accepts_fastvideo_scaled_timesteps_without_terminal() -> None:
    expected = torch.tensor([1.0, 0.5, 0.0])

    verify_fastvideo_used_sigmas(
        torch.tensor([1000.0, 500.0]),
        expected=expected,
        sample_index=2,
    )


def test_verify_accepts_fastvideo_integer_timestep_quantization() -> None:
    expected = torch.tensor([1.0, 5.0 / 6.0, 0.0])

    verify_fastvideo_used_sigmas(
        torch.tensor([1000, 833]),
        expected=expected,
        sample_index=2,
    )


def test_verify_rejects_one_integer_timestep_of_drift() -> None:
    with pytest.raises(RuntimeError, match="value mismatch"):
        verify_fastvideo_used_sigmas(
            torch.tensor([1000, 832]),
            expected=torch.tensor([1.0, 5.0 / 6.0, 0.0]),
            sample_index=4,
        )


@pytest.mark.parametrize(
    ("actual", "message"),
    [
        (None, "did not echo trajectory_timesteps"),
        (torch.tensor([1000.0]), "shape mismatch"),
        (torch.tensor([1000.0, 400.0]), "value mismatch"),
        (torch.tensor([[1000.0, 500.0]]), "shape mismatch"),
    ],
)
def test_verify_rejects_missing_shortened_malformed_or_drifted_schedule(
    actual: torch.Tensor | None,
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        verify_fastvideo_used_sigmas(
            actual,
            expected=torch.tensor([1.0, 0.5, 0.0]),
            sample_index=4,
        )
