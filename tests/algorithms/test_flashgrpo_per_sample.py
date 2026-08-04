"""CPU unit tests for FlashGRPO per-sample temporal rectification (the S==1 path).

Each flash sample took its single stochastic step at its OWN
``sde_index_per_sample`` and is normalized by the mean coefficient over the
shared candidate pool (``rectification_indices``). These tests exercise that
per-sample weighting and its guard rails without constructing a full stage /
model — they touch only the pure tensor math, so a lightweight shell instance
carrying ``rectification_indices`` and ``params.eta`` is enough.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from unirl.algorithms.flashgrpo import FlashGRPO
from unirl.types.segments.latent import LatentSegment, make_video_segment

CPU = torch.device("cpu")


def _flash(rectification_indices, eta: float = 1.0) -> FlashGRPO:
    """A FlashGRPO shell exposing only what per-sample rectification reads."""
    algo = object.__new__(FlashGRPO)
    algo.rectification_indices = None if rectification_indices is None else [int(i) for i in rectification_indices]
    algo.params = SimpleNamespace(eta=eta)
    return algo


def _segment_with_steps(steps, num_timesteps: int = 20) -> LatentSegment:
    """A video segment whose samples took their SDE step at each of ``steps``."""
    sigmas = torch.linspace(0.99, 0.01, num_timesteps + 1)
    return make_video_segment(
        latents=torch.zeros(len(steps), 2, 3),
        sigmas=sigmas,
        sde_logp=torch.zeros(len(steps), 1),
        sde_index_per_sample=torch.tensor(steps, dtype=torch.long),
    )


def test_weights_shape_and_dtype() -> None:
    algo = _flash(rectification_indices=list(range(10)))
    seg = _segment_with_steps([0, 3, 5, 9])
    weights = algo._rectification_weights_per_sample(segment=seg, device=CPU)
    assert weights.shape == (4, 1)
    assert weights.dtype == torch.float32
    assert torch.isfinite(weights).all()
    assert (weights > 0).all()


def test_same_step_gets_equal_weight() -> None:
    """Two samples that took the same step share a weight; a third differs."""
    algo = _flash(rectification_indices=list(range(10)))
    weights = algo._rectification_weights_per_sample(segment=_segment_with_steps([3, 3, 7]), device=CPU).flatten()
    assert torch.allclose(weights[0], weights[1])
    assert not torch.allclose(weights[0], weights[2])


def test_shared_normalizer_preserves_relative_weights() -> None:
    """Weights are per-sample coeff / shared pool-mean, so their ratio equals
    the raw coefficient ratio (the normalizer cancels)."""
    algo = _flash(rectification_indices=list(range(10)))
    seg = _segment_with_steps([2, 6])
    weights = algo._rectification_weights_per_sample(segment=seg, device=CPU).flatten()
    coeff = algo._rectification_coefficients(sigmas=seg.sigmas.float(), steps=[2, 6], device=CPU)
    assert torch.allclose(weights[0] / weights[1], coeff[0] / coeff[1], atol=1e-5)


def test_requires_rectification_indices() -> None:
    algo = _flash(rectification_indices=None)
    with pytest.raises(ValueError, match="rectification_indices"):
        algo._rectification_weights_per_sample(segment=_segment_with_steps([0, 1]), device=CPU)


def test_requires_index_field() -> None:
    algo = _flash(rectification_indices=list(range(10)))
    seg = make_video_segment(latents=torch.zeros(2, 2, 3), sigmas=torch.linspace(0.99, 0.01, 21))
    with pytest.raises(ValueError, match="sde_index_per_sample"):
        algo._rectification_weights_per_sample(segment=seg, device=CPU)


def test_requires_sigmas() -> None:
    algo = _flash(rectification_indices=list(range(10)))
    seg = make_video_segment(
        latents=torch.zeros(2, 2, 3),
        sde_index_per_sample=torch.tensor([0, 1], dtype=torch.long),
    )
    with pytest.raises(ValueError, match="sigmas"):
        algo._rectification_weights_per_sample(segment=seg, device=CPU)


def test_step_out_of_range_raises() -> None:
    algo = _flash(rectification_indices=list(range(10)))
    seg = _segment_with_steps([0, 25], num_timesteps=20)  # 25 >= T == 20
    with pytest.raises(ValueError, match="out of range"):
        algo._rectification_weights_per_sample(segment=seg, device=CPU)
