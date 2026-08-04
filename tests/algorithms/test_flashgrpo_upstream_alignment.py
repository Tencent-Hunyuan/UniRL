"""Numerical alignment tests against upstream Flash-GRPO's published constants.

Upstream (``Shredded-Pork/Flash-GRPO`` commit
``bd6051f68e1ab444e5ec7c6ffe0a1f7eaf559a0d``) hardcodes the temporal-gradient-
rectification coefficient for its WAN 2.1 recipe as a lookup table in
``scripts/train_wan2_1_flash_1node.py``::

    value_dict = {'999': 7.4770, '982': 7.0414, ...}   # 10 candidate steps
    w = 1 / mean(value_tensor)
    loss = -w * value_tensor * advantages * ratio

That table is a *golden reference*: it is the coefficient
``1/(sqrt(-dt)/std_dev_t + std_dev_t*sqrt(-dt)*(1-sigma)/(2*sigma))`` evaluated
on WAN's 20-step, shift-3.0 schedule. These tests reproduce it through UniRL's
own :meth:`FlashGRPO._rectification_coefficients` so that any drift in the
formula — a flipped reciprocal, a wrong ``sigma_max``/``sigma_min`` convention,
a dropped ``eta`` — fails here rather than silently changing training dynamics.

Two levels are checked separately, because they can fail independently:

1. **Formula** — fed upstream's own sigma grid, our coefficients must match the
   published table to its 4-decimal precision. This isolates the math.
2. **Recipe** — fed UniRL's WAN sigma grid, the *normalized* weights (what
   actually multiplies the loss) must match upstream's normalized weights.
   UniRL's static-shift grid differs from WAN's UniPC grid by a uniform factor
   of ``1 - 1/num_train_timesteps`` (our first sigma is 1.0, upstream's 0.99967),
   a ~0.1% offset on raw coefficients that all but cancels once normalized.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from unirl.algorithms.flashgrpo import FlashGRPO
from unirl.sde.kernels import FlashSDEStrategy
from unirl.sde.runtime import get_sigma_schedule

# Upstream train_wan2_1_flash_1node.py:1064, keyed by integer timestep.
UPSTREAM_VALUE_DICT = {
    999: 7.4770,
    982: 7.0414,
    963: 6.6112,
    944: 6.1867,
    922: 5.7682,
    899: 5.3559,
    874: 4.9502,
    847: 4.5513,
    817: 4.1596,
    785: 3.7754,
}
UPSTREAM_NUM_STEPS = 20
UPSTREAM_SHIFT = 3.0
UPSTREAM_POOL = list(range(10))  # "only choose the first 10 steps"


def _upstream_sigma_grid(num_steps: int = UPSTREAM_NUM_STEPS, shift: float = UPSTREAM_SHIFT) -> torch.Tensor:
    """Rebuild WAN's UniPC flow-sigma grid, the one upstream's table was computed on.

    Mirrors ``UniPCMultistepScheduler.set_timesteps`` under ``use_flow_sigmas``:
    a ``linspace(1, 1/T, N+1)`` alpha grid, converted to sigmas, shifted, then
    flipped. The terminal ``0.0`` is appended so the result is the ``N+1``-long
    schedule UniRL segments carry.
    """
    alphas = np.linspace(1.0, 1.0 / 1000, num_steps + 1)
    sigmas = 1.0 - alphas
    sigmas = np.flip(shift * sigmas / (1 + (shift - 1) * sigmas))[:-1].copy()
    return torch.tensor(np.concatenate([sigmas, [0.0]]), dtype=torch.float32)


def _flash(rectification_indices=UPSTREAM_POOL, eta: float = 1.0) -> FlashGRPO:
    """A FlashGRPO shell exposing only what the rectification math reads."""
    algo = object.__new__(FlashGRPO)
    algo.rectification_indices = list(rectification_indices)
    algo.params = SimpleNamespace(eta=eta)
    return algo


def _coefficients(sigmas: torch.Tensor, *, eta: float = 1.0, device=torch.device("cpu")) -> torch.Tensor:
    return _flash(eta=eta)._rectification_coefficients(
        sigmas=sigmas.to(device=device, dtype=torch.float32),
        steps=UPSTREAM_POOL,
        device=device,
    )


def test_upstream_sigma_grid_reproduces_upstream_timesteps() -> None:
    """Sanity-check the reference grid: its timesteps must be value_dict's keys."""
    grid = _upstream_sigma_grid()
    timesteps = [int(grid[i].item() * 1000) for i in UPSTREAM_POOL]
    assert timesteps == list(UPSTREAM_VALUE_DICT)


def test_coefficients_match_upstream_value_dict_exactly() -> None:
    """On upstream's own grid, our coefficients == upstream's published table.

    Tolerance is set by the table's precision (4 decimals), not by ours; the
    observed agreement is ~1e-5 relative.
    """
    coeff = _coefficients(_upstream_sigma_grid()).double()
    expected = torch.tensor(list(UPSTREAM_VALUE_DICT.values()), dtype=torch.float64)
    torch.testing.assert_close(coeff, expected, rtol=2e-4, atol=0.0)


def test_normalized_weights_match_upstream_on_unirl_grid() -> None:
    """The weights that actually multiply the loss agree on UniRL's WAN grid.

    Upstream's effective weight is ``value / mean(value)``; ours is
    ``coeff / mean(coeff over pool)``. Upstream averages over the *sampled*
    batch of timesteps, ours over the whole candidate pool — equal in
    expectation for uniform draws, and ours is the zero-variance estimator.
    """
    ours = _coefficients(get_sigma_schedule(UPSTREAM_NUM_STEPS, shift=UPSTREAM_SHIFT)).double()
    ours = ours / ours.mean()
    upstream = torch.tensor(list(UPSTREAM_VALUE_DICT.values()), dtype=torch.float64)
    upstream = upstream / upstream.mean()
    torch.testing.assert_close(ours, upstream, rtol=1e-3, atol=0.0)


def test_per_sample_weights_match_upstream_normalized_weights() -> None:
    """End-to-end through the per-sample path used by the WAN recipe."""
    from unirl.types.segments.latent import make_video_segment

    sigmas = get_sigma_schedule(UPSTREAM_NUM_STEPS, shift=UPSTREAM_SHIFT)
    segment = make_video_segment(
        latents=torch.zeros(len(UPSTREAM_POOL), 2, 3),
        sigmas=sigmas,
        sde_logp=torch.zeros(len(UPSTREAM_POOL), 1),
        sde_index_per_sample=torch.tensor(UPSTREAM_POOL, dtype=torch.long),
    )
    weights = _flash()._rectification_weights_per_sample(segment=segment, device=torch.device("cpu"))
    assert weights.shape == (len(UPSTREAM_POOL), 1)

    upstream = torch.tensor(list(UPSTREAM_VALUE_DICT.values()), dtype=torch.float64)
    upstream = upstream / upstream.mean()
    torch.testing.assert_close(weights.flatten().double(), upstream, rtol=1e-3, atol=0.0)


def test_coefficient_is_reciprocal_not_denominator() -> None:
    """Guard the reciprocal direction.

    Upstream returns ``1/(...)`` from ``sde_step_with_logprob`` and multiplies
    the loss by it, so the coefficient DECREASES with step index (7.48 -> 3.78,
    early/noisy steps weighted up). Dropping the reciprocal would invert that
    ordering and silently flip the curriculum.
    """
    coeff = _coefficients(_upstream_sigma_grid())
    assert torch.all(coeff[:-1] > coeff[1:]), "coefficients must decrease with step index"
    assert coeff[0] > 1.0, "upstream coefficients are O(1..10), not O(0.1)"


def test_flash_std_dev_t_matches_upstream_closed_form() -> None:
    """``FlashSDEStrategy`` reproduces upstream's ``std_dev_t`` exactly.

    Upstream: ``std_dev_t = sigma_min + (sigma_max - sigma_min) * sigma`` with
    ``sigma_max = sigmas[1]`` and ``sigma_min = sigmas[-1]`` and no eta knob,
    so ``eta=1.0`` is the aligned setting.
    """
    grid = _upstream_sigma_grid()
    strategy = FlashSDEStrategy()
    strategy.init_schedule(grid)

    sigma = grid[torch.tensor(UPSTREAM_POOL)]
    sigma_next = grid[torch.tensor(UPSTREAM_POOL) + 1]
    sigma_max = float(grid[1].item())
    sigma_min = float(grid[-1].item())

    got = strategy._std_dev_t(sigma=sigma, sigma_next=sigma_next, eta=1.0, sigma_max=sigma_max)
    expected = sigma_min + (sigma_max - sigma_min) * sigma
    torch.testing.assert_close(got, expected)


def test_eta_scales_std_dev_t_and_leaves_upstream_at_one() -> None:
    """eta is UniRL's extra knob; only eta=1.0 reproduces upstream."""
    grid = _upstream_sigma_grid()
    strategy = FlashSDEStrategy()
    strategy.init_schedule(grid)
    sigma = grid[torch.tensor(UPSTREAM_POOL)]
    kwargs = dict(sigma=sigma, sigma_next=sigma, sigma_max=float(grid[1].item()))
    torch.testing.assert_close(
        strategy._std_dev_t(eta=0.5, **kwargs),
        0.5 * strategy._std_dev_t(eta=1.0, **kwargs),
    )


def test_zero_eta_rejected() -> None:
    """A trained SDE step with eta=0 has no noise, so the coefficient is undefined."""
    with pytest.raises(ValueError, match="eta > 0"):
        _coefficients(_upstream_sigma_grid(), eta=0.0)


@pytest.mark.parametrize("device_str", ["cpu", "cuda"])
def test_coefficients_device_parity(device_str: str) -> None:
    """The formula is device-independent: CUDA must match upstream's table too."""
    if device_str == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = torch.device(device_str)
    coeff = _coefficients(_upstream_sigma_grid(), device=device).double().cpu()
    expected = torch.tensor(list(UPSTREAM_VALUE_DICT.values()), dtype=torch.float64)
    torch.testing.assert_close(coeff, expected, rtol=2e-4, atol=0.0)
