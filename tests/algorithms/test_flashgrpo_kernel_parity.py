"""GPU/CPU numerical parity between UniRL's Flash SDE kernel + loss and upstream.

Upstream Flash-GRPO (``Shredded-Pork/Flash-GRPO`` commit
``bd6051f68e1ab444e5ec7c6ffe0a1f7eaf559a0d``) implements its transition in
``flow_grpo/diffusers_patch/wan2_1_pipeline_with_logprob2.py::sde_step_with_logprob``
and its objective inline in ``scripts/train_wan2_1_flash_1node.py``. Both are
transcribed verbatim below and compared element-wise against UniRL's
:class:`FlashSDEStrategy` / :class:`FlashGRPO`.

These are the tests that would catch a real divergence in the tensor math (as
opposed to :mod:`test_flashgrpo_upstream_alignment`, which pins the published
coefficient table). Every case runs on CPU and, when available, CUDA — the H20
path is where training actually happens, so device parity is asserted rather
than assumed.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from unirl.algorithms.flashgrpo import FlashGRPO
from unirl.sde.kernels import FlashSDEStrategy
from unirl.types.segments.latent import make_video_segment

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
NUM_STEPS = 20
SHIFT = 3.0
POOL = list(range(10))


def _wan_sigma_grid(num_steps: int = NUM_STEPS, shift: float = SHIFT) -> torch.Tensor:
    """WAN's UniPC flow-sigma grid (the schedule upstream samples on)."""
    alphas = np.linspace(1.0, 1.0 / 1000, num_steps + 1)
    sig = 1.0 - alphas
    sig = np.flip(shift * sig / (1 + (shift - 1) * sig))[:-1].copy()
    return torch.tensor(np.concatenate([sig, [0.0]]), dtype=torch.float32)


# ---------------------------------------------------------------------------
# Verbatim upstream transcriptions
# ---------------------------------------------------------------------------


def _upstream_sde_step_with_logprob(sigmas, model_output, sample, prev_sample, step_index):
    """Transcription of upstream ``sde_step_with_logprob`` (fp32, prev_sample given).

    Kept line-for-line faithful to upstream, including the ``.float()`` casts and
    the ``sigmas[1]`` / ``sigmas[-1]`` endpoints, so a divergence here is a real
    divergence and not a transcription artifact.
    """
    model_output = model_output.float()
    sample = sample.float()
    prev_sample = prev_sample.float()

    view = (-1,) + (1,) * (sample.ndim - 1)
    sigma = sigmas[step_index].view(view)
    sigma_prev = sigmas[step_index + 1].view(view)
    sigma_max = sigmas[1].item()
    sigma_min = sigmas[-1].item()
    dt = sigma_prev - sigma

    std_dev_t = sigma_min + (sigma_max - sigma_min) * sigma
    prev_sample_mean = (
        sample * (1 + std_dev_t**2 / (2 * sigma) * dt)
        + model_output * (1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)) * dt
    )

    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * ((std_dev_t * torch.sqrt(-1 * dt)) ** 2))
        - torch.log(std_dev_t * torch.sqrt(-1 * dt))
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    coe = 1 / (torch.sqrt(-1 * dt) / std_dev_t + (std_dev_t * torch.sqrt(-1 * dt) * (1 - sigma)) / (2 * sigma))
    return prev_sample_mean, log_prob, std_dev_t, coe


def _upstream_policy_loss(new_logp, old_logp, advantages, coe, clip_range):
    """Transcription of upstream's rectified GRPO objective (train script:1062-1082)."""
    ratio = torch.exp(new_logp - old_logp)
    w = 1.0 / coe.mean()
    unclipped = -w * coe * advantages * ratio
    clipped = -w * coe * advantages * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    return torch.mean(torch.maximum(unclipped, clipped))


def _flash_algo(param, base_logp, *, clip_range=1e-3, adv_clip_max=5.0):
    """A FlashGRPO shell wired to a differentiable fake stage."""
    algo = object.__new__(FlashGRPO)
    algo.rectification_indices = POOL
    algo.params = SimpleNamespace(eta=1.0)
    algo.stage = _FakeStage(param, base_logp)
    algo.conditions_cls = None
    algo.clip_range = clip_range
    algo.clip_schedule = "constant"
    algo.beta = 0.0
    algo.old_logp_source = "rollout"
    algo.adv_clip_max = adv_clip_max
    return algo


# ---------------------------------------------------------------------------
# SDE kernel parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device_str", DEVICES)
def test_flash_sde_step_matches_upstream(device_str: str) -> None:
    """prev_sample_mean, std_dev_t and log_prob all match upstream element-wise."""
    device = torch.device(device_str)
    torch.manual_seed(0)
    grid = _wan_sigma_grid().to(device)
    steps = torch.tensor([0, 3, 5, 9], device=device)
    n = len(steps)

    # [N, C, F, H, W] — a WAN-shaped latent, small enough to stay cheap.
    model_output = torch.randn(n, 4, 2, 6, 8, device=device)
    sample = torch.randn(n, 4, 2, 6, 8, device=device)
    prev_sample = torch.randn(n, 4, 2, 6, 8, device=device)

    strategy = FlashSDEStrategy()
    strategy.init_schedule(grid)
    view = (-1, 1, 1, 1, 1)
    sigma = grid[steps].view(view)
    sigma_next = grid[steps + 1].view(view)

    got_prev, got_mean, got_std_var = strategy.step(
        noise_pred=model_output,
        sample=sample,
        sigma=sigma,
        sigma_next=sigma_next,
        eta=1.0,
        prev_sample=prev_sample,
        sigma_max=float(grid[1].item()),
    )
    got_logp = strategy.compute_log_prob(prev_sample=got_prev, prev_sample_mean=got_mean, std_var=got_std_var)
    got_logp = got_logp.mean(dim=tuple(range(1, got_logp.ndim)))

    exp_mean, exp_logp, exp_std_dev_t, _ = _upstream_sde_step_with_logprob(
        grid, model_output, sample, prev_sample, steps
    )

    torch.testing.assert_close(got_mean, exp_mean)
    torch.testing.assert_close(got_logp, exp_logp)
    # std_var is std_dev_t * sqrt(-dt); recover std_dev_t to compare directly.
    torch.testing.assert_close(got_std_var, exp_std_dev_t * torch.sqrt(sigma - sigma_next))


@pytest.mark.parametrize("device_str", DEVICES)
def test_rectification_matches_upstream_coe(device_str: str) -> None:
    """Our rectification coefficient == upstream's returned ``coe``, per step."""
    device = torch.device(device_str)
    grid = _wan_sigma_grid().to(device)
    steps = torch.tensor(POOL, device=device)
    dummy = torch.zeros(len(POOL), 1, 1, 1, 1, device=device)

    _, _, _, exp_coe = _upstream_sde_step_with_logprob(grid, dummy, dummy, dummy, steps)

    algo = object.__new__(FlashGRPO)
    algo.rectification_indices = POOL
    algo.params = SimpleNamespace(eta=1.0)
    got = algo._rectification_coefficients(sigmas=grid, steps=POOL, device=device)

    torch.testing.assert_close(got, exp_coe.flatten())


# ---------------------------------------------------------------------------
# Loss parity
# ---------------------------------------------------------------------------


class _FakeStage:
    """A stage whose replay log-prob is a differentiable function of ``param``."""

    def __init__(self, param: torch.Tensor, base_logp: torch.Tensor) -> None:
        self.param = param
        self.base_logp = base_logp

    def replay(self, conditions, *, segment, params, step_indices):
        del conditions, params, step_indices
        return SimpleNamespace(
            log_probs=self.base_logp + self.param,
            prev_sample_means=None,
        )


@pytest.mark.parametrize("device_str", DEVICES)
def test_loss_and_gradient_match_upstream(device_str: str) -> None:
    """FlashGRPO's loss and its gradient equal upstream's rectified objective.

    Both sides get identical inputs; the only thing under test is the
    combination of clip, advantage broadcast and rectification weighting. The
    gradient is compared too, since a weight applied after ``.backward()``
    would still match on the scalar loss.
    """
    device = torch.device(device_str)
    torch.manual_seed(0)
    grid = _wan_sigma_grid().to(device)
    per_sample_steps = [0, 2, 5, 7, 9]
    n = len(per_sample_steps)
    clip_range = 1e-3

    old_logp = torch.randn(n, 1, device=device)
    base_logp = old_logp.clone()
    advantages = torch.tensor([1.0, -0.5, 0.25, 2.0, -1.5], device=device)

    segment = make_video_segment(
        latents=torch.zeros(n, 2, 3, device=device),
        sigmas=grid,
        sde_logp=old_logp,
        # A flash rollout records ONE stochastic transition, so the (now
        # non-authoritative) shared vector has length 1; the per-sample stamp is
        # what the loss actually reads. Mirrors _stamp_sde_index_per_sample.
        sde_indices=torch.tensor([per_sample_steps[0]], dtype=torch.long),
        sde_index_per_sample=torch.tensor(per_sample_steps, dtype=torch.long),
    )

    # --- ours ---
    param = torch.zeros(n, 1, device=device, requires_grad=True)
    algo = _flash_algo(param, base_logp, clip_range=clip_range)

    result = algo.compute_loss_and_backward(
        conditions={},
        segment=segment,
        advantages=advantages,
        training_progress=0.0,
        loss_scale=1.0,
    )
    assert result.has_backward
    got_loss = result.loss
    got_grad = param.grad.detach().clone()

    # --- upstream ---
    param_up = torch.zeros(n, 1, device=device, requires_grad=True)
    new_logp_up = base_logp + param_up
    coe_all = _upstream_sde_step_with_logprob(
        grid,
        torch.zeros(len(POOL), 1, device=device),
        torch.zeros(len(POOL), 1, device=device),
        torch.zeros(len(POOL), 1, device=device),
        torch.tensor(POOL, device=device),
    )[3].flatten()
    coe_sample = _upstream_sde_step_with_logprob(
        grid,
        torch.zeros(n, 1, device=device),
        torch.zeros(n, 1, device=device),
        torch.zeros(n, 1, device=device),
        torch.tensor(per_sample_steps, device=device),
    )[3].reshape(n, 1)
    # Upstream normalizes by the mean coefficient over the candidate pool.
    weights = coe_sample / coe_all.mean()
    ratio = torch.exp(new_logp_up - old_logp)
    adv = advantages.reshape(-1, 1)
    unclipped = -weights * adv * ratio
    clipped = -weights * adv * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    exp_loss = torch.mean(torch.maximum(unclipped, clipped))
    exp_loss.backward()

    assert got_loss == pytest.approx(float(exp_loss.detach().item()), rel=1e-5, abs=1e-8)
    torch.testing.assert_close(got_grad, param_up.grad, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("device_str", DEVICES)
def test_clip_engages_like_upstream(device_str: str) -> None:
    """With a large log-prob shift the clipped branch wins on both sides."""
    device = torch.device(device_str)
    n, clip_range = 4, 1e-3
    old_logp = torch.zeros(n, 1, device=device)
    advantages = torch.ones(n, device=device)
    grid = _wan_sigma_grid().to(device)
    steps = [0, 1, 2, 3]

    segment = make_video_segment(
        latents=torch.zeros(n, 2, 3, device=device),
        sigmas=grid,
        sde_logp=old_logp,
        sde_indices=torch.tensor([steps[0]], dtype=torch.long),
        sde_index_per_sample=torch.tensor(steps, dtype=torch.long),
    )
    param = torch.full((n, 1), 0.5, device=device, requires_grad=True)  # ratio ~ 1.65

    algo = _flash_algo(param, old_logp, clip_range=clip_range)

    result = algo.compute_loss_and_backward(
        conditions={},
        segment=segment,
        advantages=advantages,
        training_progress=0.0,
        loss_scale=1.0,
    )
    # Positive advantage past the ceiling -> clipped branch -> zero gradient.
    assert result.metrics["clip_fraction"] == pytest.approx(1.0)
    torch.testing.assert_close(param.grad, torch.zeros_like(param))


@pytest.mark.parametrize("device_str", DEVICES)
def test_advantage_clamp_matches_upstream(device_str: str) -> None:
    """Advantages are bounded to +-adv_clip_max before the ratio, as upstream does.

    Upstream (train script:1056) applies
    ``torch.clamp(sample["advantages"][:, 0], -adv_clip_max, +adv_clip_max)``
    with ``adv_clip_max = 5``. UniRL's shared ``compute_advantages`` has no such
    bound, so the clamp has to live here; without it a batch whose reward spread
    collapses divides by a tiny global std and one gradient spike undoes
    training. Both branches keep ratio == 1 so only the advantage path is
    under test.
    """
    device = torch.device(device_str)
    grid = _wan_sigma_grid().to(device)
    steps = [0, 3, 6, 9]
    n = len(steps)
    old_logp = torch.zeros(n, 1, device=device)

    def _run(advantages: torch.Tensor):
        segment = make_video_segment(
            latents=torch.zeros(n, 2, 3, device=device),
            sigmas=grid,
            sde_logp=old_logp,
            sde_indices=torch.tensor([steps[0]], dtype=torch.long),
            sde_index_per_sample=torch.tensor(steps, dtype=torch.long),
        )
        param = torch.zeros(n, 1, device=device, requires_grad=True)  # ratio == 1
        algo = _flash_algo(param, old_logp, adv_clip_max=5.0)
        result = algo.compute_loss_and_backward(
            conditions={},
            segment=segment,
            advantages=advantages,
            training_progress=0.0,
            loss_scale=1.0,
        )
        return result, param.grad.detach().clone()

    # An advantage far outside the bound must behave exactly like one AT the
    # bound -- loss and gradient both.
    wild = torch.tensor([50.0, -50.0, 1.0, -1.0], device=device)
    saturated, saturated_grad = _run(wild)
    bounded, bounded_grad = _run(torch.tensor([5.0, -5.0, 1.0, -1.0], device=device))
    assert saturated.loss == pytest.approx(bounded.loss, rel=1e-6, abs=1e-9)
    torch.testing.assert_close(saturated_grad, bounded_grad)

    # ... and the metrics report the raw (pre-clamp) magnitudes, which is what
    # makes a reward-regression run diagnosable from the logs.
    assert saturated.metrics["adv_clip_fraction"] == pytest.approx(0.5)
    assert saturated.metrics["adv_abs_max"] == pytest.approx(50.0)

    # Inside the bound the clamp is a no-op.
    inside, _ = _run(torch.tensor([4.0, -4.0, 1.0, -1.0], device=device))
    assert inside.metrics["adv_clip_fraction"] == pytest.approx(0.0)
    assert inside.loss != pytest.approx(bounded.loss)


def test_adv_clip_max_must_be_positive() -> None:
    """A non-positive bound would zero every advantage; reject it at construction."""
    with pytest.raises(ValueError, match="adv_clip_max"):
        FlashGRPO(
            params=SimpleNamespace(eta=1.0),
            stage=SimpleNamespace(),
            rectification_indices=POOL,
            adv_clip_max=0.0,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cpu_cuda_loss_agree() -> None:
    """The same inputs give the same loss on CPU and CUDA (no device drift)."""
    losses = []
    for device_str in ("cpu", "cuda"):
        device = torch.device(device_str)
        grid = _wan_sigma_grid().to(device)
        steps = [0, 2, 5, 7, 9]
        n = len(steps)
        old_logp = torch.linspace(-1.0, 1.0, n, device=device).reshape(n, 1)
        segment = make_video_segment(
            latents=torch.zeros(n, 2, 3, device=device),
            sigmas=grid,
            sde_logp=old_logp,
            sde_indices=torch.tensor([steps[0]], dtype=torch.long),
            sde_index_per_sample=torch.tensor(steps, dtype=torch.long),
        )
        param = torch.full((n, 1), 1e-4, device=device, requires_grad=True)
        algo = _flash_algo(param, old_logp)
        losses.append(
            algo.compute_loss_and_backward(
                conditions={},
                segment=segment,
                advantages=torch.linspace(-1.0, 1.0, n, device=device),
                training_progress=0.0,
                loss_scale=1.0,
            ).loss
        )
    assert losses[0] == pytest.approx(losses[1], rel=1e-6, abs=1e-9)
