"""CPU unit tests for WAN21DiffusionStage._replay_per_sample (FlashGRPO S==1).

Each flash sample took its single stochastic step at its OWN
``sde_index_per_sample[n]``; the rollout stored that step's latent pair in fixed
slots (0 = before, 1 = after), so one batched ``step_with_logp`` with a
per-sample ``sigma`` ``[N]`` replays them all. These tests lock in the two
load-bearing behaviors — the per-sample sigma gather and the fixed-slot read —
plus every guard branch, using a stubbed ``step`` so no real model forward runs.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from unirl.models.wan21.diffusion import WAN21DiffusionStage
from unirl.types.segments.latent import make_video_segment

NUM_TIMESTEPS = 20


class _RecordingStep:
    """Stand-in for WAN21DiffusionStep: records the kwargs it saw, returns fakes."""

    def __init__(self, log_prob, prev_mean):
        self._log_prob = log_prob
        self._prev_mean = prev_mean
        self.seen: dict = {}

    def step_with_logp(
        self,
        model,
        conditions,
        *,
        strategy,
        sample,
        prev_sample,
        sigma,
        sigma_next,
        guidance_scale,
        eta,
        sigma_max,
        step_index,
    ):
        self.seen = {
            "sample": sample,
            "prev_sample": prev_sample,
            "sigma": sigma,
            "sigma_next": sigma_next,
            "step_index": step_index,
        }
        return (None, self._log_prob, self._prev_mean)


def _stage(step: _RecordingStep) -> WAN21DiffusionStage:
    """A bare stage exposing only what _replay_per_sample reads."""
    stage = object.__new__(WAN21DiffusionStage)
    stage.step = step
    stage.model = None
    stage.strategy = None
    stage.autocast_dtype = torch.float32
    stage.logprob_dtype = torch.float32
    stage.trajectory_dtype = torch.float32
    return stage


def _segment(steps, k_slots: int = 3):
    """A video segment with ``len(steps)`` samples, each stamped its own step."""
    n = len(steps)
    return make_video_segment(
        latents=torch.randn(n, k_slots, 4),  # [N, K, feat]; slots 0/1 = before/after
        sigmas=torch.linspace(0.99, 0.01, NUM_TIMESTEPS + 1),
        sde_index_per_sample=torch.tensor(steps, dtype=torch.long),
    )


_PARAMS = SimpleNamespace(guidance_scale=5.0, eta=1.0)


def test_replay_per_sample_happy_path_shapes_and_gather() -> None:
    steps = [0, 3, 5]
    step = _RecordingStep(log_prob=torch.zeros(3), prev_mean=torch.zeros(3, 4, 2))
    seg = _segment(steps)

    result = _stage(step)._replay_per_sample(None, segment=seg, params=_PARAMS)

    assert result.log_probs.shape == (3, 1)
    assert result.log_probs.dtype == torch.float32
    assert result.prev_sample_means.shape == (3, 1, 4, 2)
    # Each sample is replayed at ITS OWN sigma / sigma_next (the whole point).
    sigmas = seg.sigmas.float()
    idx = torch.tensor(steps)
    assert torch.allclose(step.seen["sigma"], sigmas[idx])
    assert torch.allclose(step.seen["sigma_next"], sigmas[idx + 1])
    # Fixed-slot read: slot 0 = before, slot 1 = after.
    assert torch.allclose(step.seen["sample"], seg.latents[:, 0])
    assert torch.allclose(step.seen["prev_sample"], seg.latents[:, 1])


def test_replay_per_sample_prev_mean_none_yields_no_means() -> None:
    step = _RecordingStep(log_prob=torch.zeros(2), prev_mean=None)
    result = _stage(step)._replay_per_sample(None, segment=_segment([1, 4]), params=_PARAMS)
    assert result.log_probs.shape == (2, 1)
    assert result.prev_sample_means is None


def test_replay_per_sample_requires_sigmas() -> None:
    seg = make_video_segment(
        latents=torch.randn(2, 3, 4),
        sde_index_per_sample=torch.tensor([0, 1], dtype=torch.long),
    )
    with pytest.raises(ValueError, match="sigmas / latents missing"):
        _stage(_RecordingStep(torch.zeros(2), None))._replay_per_sample(None, segment=seg, params=_PARAMS)


def test_replay_per_sample_requires_two_slots() -> None:
    seg = _segment([0, 1], k_slots=1)  # K == 1, no "after" slot
    with pytest.raises(ValueError, match="K >= 2"):
        _stage(_RecordingStep(torch.zeros(2), None))._replay_per_sample(None, segment=seg, params=_PARAMS)


def test_replay_per_sample_index_length_mismatch() -> None:
    seg = make_video_segment(
        latents=torch.randn(3, 3, 4),  # 3 samples
        sigmas=torch.linspace(0.99, 0.01, NUM_TIMESTEPS + 1),
        sde_index_per_sample=torch.tensor([0, 1], dtype=torch.long),  # only 2 indices
    )
    with pytest.raises(ValueError, match="length"):
        _stage(_RecordingStep(torch.zeros(3), None))._replay_per_sample(None, segment=seg, params=_PARAMS)


def test_replay_per_sample_none_logp_raises() -> None:
    step = _RecordingStep(log_prob=None, prev_mean=None)  # deterministic strategy
    with pytest.raises(RuntimeError, match="None log-prob"):
        _stage(step)._replay_per_sample(None, segment=_segment([0, 2]), params=_PARAMS)
