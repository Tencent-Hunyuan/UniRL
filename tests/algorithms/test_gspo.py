"""Unit tests for GSPO (sequence-level Group Sequence Policy Optimization).

Verifies GSPO's defining property vs GRPO: the importance ratio is formed at the
SEQUENCE level as ``exp(mean_t(new_logp_t - old_logp_t))`` (length-normalized),
not per token. Uses a fake ARStage so the test is model-free and CPU-only.
"""

from __future__ import annotations

import math

import torch

from unirl.algorithms.gspo import GSPO
from unirl.types.segments import TextSegment


class _FakeStage:
    """Stands in for an ARStage: replay returns a fixed differentiable new_logp."""

    def __init__(self, new_logp: torch.Tensor) -> None:
        self._new_logp = new_logp

    def replay(self, conditions, *, segment, temperature: float = 1.0) -> torch.Tensor:
        return self._new_logp


def _make_segment(old_logp: torch.Tensor, lengths):
    tokens = [torch.arange(n) for n in lengths]
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + n)
    log_probs = [old_logp[cu[i] : cu[i + 1]] for i in range(len(lengths))]
    return TextSegment.pack(tokens=tokens, log_probs=log_probs)


def test_gspo_ratio_is_sequence_level():
    """ratio_i == exp(mean per-token log-ratio); loss == mean_i(-A_i * ratio_i) with no clip."""
    lengths = [2, 3]
    old = torch.tensor([-1.0, -1.0, -2.0, -2.0, -2.0])
    new = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], requires_grad=True)
    seg = _make_segment(old, lengths)
    adv = torch.tensor([1.0, 1.0])

    # Huge clip range → no clipping, exposes the raw sequence ratio.
    alg = GSPO(stage=_FakeStage(new), clip_range=10.0, conditions_cls=None)
    res = alg.compute_loss_and_backward(
        conditions={}, segment=seg, advantages=adv, training_progress=0.0, loss_scale=1.0
    )

    s0 = (0 - (-1.0) + 0 - (-1.0)) / 2  # 1.0
    s1 = (0 - (-2.0)) * 3 / 3  # 2.0
    expected = (-math.exp(s0) - math.exp(s1)) / 2
    assert abs(res.loss - expected) < 1e-4, (res.loss, expected)
    assert res.has_backward
    assert res.num_steps_or_tokens == sum(lengths)
    assert new.grad is not None  # gradient flows through the differentiable new_logp


def test_gspo_zero_logratio_gives_unit_ratio():
    """When new == old, every sequence ratio is 1.0 and loss == mean(-A_i)."""
    lengths = [2, 3]
    old = torch.tensor([-1.0, -1.0, -2.0, -2.0, -2.0])
    new = old.clone().requires_grad_(True)
    seg = _make_segment(old, lengths)
    adv = torch.tensor([1.0, -1.0])

    alg = GSPO(stage=_FakeStage(new), clip_range=10.0, conditions_cls=None)
    res = alg.compute_loss_and_backward(
        conditions={}, segment=seg, advantages=adv, training_progress=0.0, loss_scale=1.0
    )
    # ratios == 1 → loss = mean(-1*1, -(-1)*1) = mean(-1, +1) = 0
    assert abs(res.loss - 0.0) < 1e-5
    assert abs(res.metrics["ratio_mean"] - 1.0) < 1e-5


def test_gspo_empty_segment_no_backward():
    seg = TextSegment.pack(tokens=[torch.zeros(0, dtype=torch.long)], log_probs=[torch.zeros(0)])
    alg = GSPO(stage=_FakeStage(torch.zeros(0)), clip_range=1e-3, conditions_cls=None)
    res = alg.compute_loss_and_backward(
        conditions={}, segment=seg, advantages=torch.tensor([0.0]), training_progress=0.0, loss_scale=1.0
    )
    assert not res.has_backward
    assert res.num_steps_or_tokens == 0


# ---------------------------------------------------------------------------
# Reference implementation — the naive, obviously-correct GSPO loss, used to
# cross-check the vectorized/clamped production path.
# ---------------------------------------------------------------------------


def _gspo_reference_loss(new, old, lengths, adv, clip_low, clip_high=None, max_log_ratio=10.0):
    """Naive per-sequence GSPO loss (Python loop), matching the paper's Eq. 7-8."""
    high = clip_low if clip_high is None else clip_high
    new = new.detach()
    old = old.detach()
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + n)
    per_seq = []
    for i, n in enumerate(lengths):
        if n <= 0:
            continue
        s_new = float(new[cu[i] : cu[i + 1]].mean())
        s_old = float(old[cu[i] : cu[i + 1]].mean())
        ratio = math.exp(min(s_new - s_old, max_log_ratio))
        a = float(adv[i])
        unclipped = -a * ratio
        clipped = -a * min(max(ratio, 1.0 - clip_low), 1.0 + high)
        per_seq.append(max(unclipped, clipped))
    return sum(per_seq) / len(per_seq)


def _run_gspo(new, old, lengths, adv, *, training_progress=0.0, loss_scale=1.0, **kw):
    seg = _make_segment(old, lengths)
    alg = GSPO(stage=_FakeStage(new), conditions_cls=None, **kw)
    return alg.compute_loss_and_backward(
        conditions={}, segment=seg, advantages=adv, training_progress=training_progress, loss_scale=loss_scale
    )


# ---------------------------------------------------------------------------
# Clipping — GSPO's defining feature is the tight sequence-level clip.
# ---------------------------------------------------------------------------


def test_gspo_positive_adv_clips_high_side():
    """A > 0 with ratio above 1+high: the clipped (capped) branch must win."""
    lengths = [1]
    old = torch.tensor([0.0])
    new = torch.tensor([0.5], requires_grad=True)  # s = 0.5, ratio = e^0.5 ≈ 1.6487
    adv = torch.tensor([1.0])

    res = _run_gspo(new, old, lengths, adv, clip_range=0.1)

    ratio = math.exp(0.5)
    expected = -1.0 * (1.0 + 0.1)  # capped at 1+clip_range → -1.1, beats -ratio
    assert expected > -ratio  # sanity: clipped branch is the maximum
    assert abs(res.loss - expected) < 1e-4, (res.loss, expected)
    assert abs(res.metrics["clip_fraction"] - 1.0) < 1e-6
    assert abs(res.metrics["clipfrac_gt_one"] - 1.0) < 1e-6
    assert abs(res.metrics["clipfrac_lt_one"] - 0.0) < 1e-6


def test_gspo_asymmetric_clip_uses_high_bound():
    """clip_range_high (DAPO clip-higher) must be applied on the upper side,
    independently of the lower clip_range."""
    lengths = [1]
    old = torch.tensor([0.0])
    new = torch.tensor([0.5], requires_grad=True)  # ratio ≈ 1.6487
    adv = torch.tensor([1.0])

    res = _run_gspo(new, old, lengths, adv, clip_range=0.1, clip_range_high=0.3)

    # Upper cap is now 1+0.3 = 1.3 (not 1.1) → loss = -1.3, distinct from the
    # symmetric case above.
    assert abs(res.loss - (-1.3)) < 1e-4, res.loss
    ref = _gspo_reference_loss(new, old, lengths, adv, clip_low=0.1, clip_high=0.3)
    assert abs(res.loss - ref) < 1e-4


def test_gspo_negative_adv_clips_low_side():
    """A < 0 with ratio below 1-low: the clipped (floored) branch must win."""
    lengths = [1]
    old = torch.tensor([0.0])
    new = torch.tensor([-0.5], requires_grad=True)  # s = -0.5, ratio = e^-0.5 ≈ 0.6065
    adv = torch.tensor([-1.0])

    res = _run_gspo(new, old, lengths, adv, clip_range=0.1)

    # unclipped = -(-1)*0.6065 = +0.6065 ; clipped = -(-1)*clamp(0.6065,0.9,1.1)=+0.9
    # max(+0.6065, +0.9) = +0.9
    assert abs(res.loss - 0.9) < 1e-4, res.loss
    assert abs(res.metrics["clipfrac_lt_one"] - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Numerical stability — the pre-exp clamp guards against overflow to inf.
# ---------------------------------------------------------------------------


def test_gspo_large_logratio_is_clamped_and_finite():
    """A far-off-policy sequence (log-ratio 100) must NOT overflow: the clamp
    caps the log-ratio at _MAX_LOG_RATIO=10 before exp(), so loss == -e^10."""
    lengths = [1]
    old = torch.tensor([0.0])
    new = torch.tensor([100.0], requires_grad=True)  # unclamped exp(100) = inf in fp32
    adv = torch.tensor([1.0])

    # clip_range huge so the ratio clip does NOT fire (isolate the pre-exp clamp).
    res = _run_gspo(new, old, lengths, adv, clip_range=1e9)

    assert math.isfinite(res.loss), res.loss
    assert abs(res.loss - (-math.exp(GSPO._MAX_LOG_RATIO))) < 2.0, res.loss
    # gradient must also be finite (would be nan/inf without the clamp)
    assert new.grad is not None and torch.isfinite(new.grad).all()


def test_gspo_clamp_matches_reference_on_extreme_input():
    lengths = [2, 1]
    old = torch.tensor([0.0, 0.0, 0.0])
    new = torch.tensor([80.0, 80.0, -3.0], requires_grad=True)  # seq0 s=80 (clamped), seq1 s=-3
    adv = torch.tensor([1.0, 0.5])

    res = _run_gspo(new, old, lengths, adv, clip_range=1e9)
    ref = _gspo_reference_loss(new, old, lengths, adv, clip_low=1e9, max_log_ratio=GSPO._MAX_LOG_RATIO)
    assert math.isfinite(res.loss)
    assert abs(res.loss - ref) < 2.0, (res.loss, ref)


# ---------------------------------------------------------------------------
# Vectorized reduction — must match the naive per-sequence mean, drop 0-length
# sequences, and carry gradient correctly.
# ---------------------------------------------------------------------------


def test_reduce_to_sequences_matches_python_reference():
    new = torch.tensor([0.1, 0.2, -0.3, 0.4, 0.5, 0.6], requires_grad=True)
    old = torch.tensor([1.0, 1.0, 2.0, 3.0, 3.0, 3.0])
    adv = torch.tensor([0.7, -0.2, 1.5])
    lengths = torch.tensor([2, 1, 3])

    seq_new, seq_old, seq_adv = GSPO._reduce_to_sequences(new, old, adv, lengths)

    ref_new = torch.stack([new[0:2].mean(), new[2:3].mean(), new[3:6].mean()])
    ref_old = torch.stack([old[0:2].mean(), old[2:3].mean(), old[3:6].mean()])
    assert torch.allclose(seq_new, ref_new, atol=1e-6)
    assert torch.allclose(seq_old, ref_old, atol=1e-6)
    assert torch.allclose(seq_adv, adv, atol=1e-6)


def test_reduce_to_sequences_gradient_is_inverse_length():
    new = torch.tensor([0.1, 0.2, 0.4, 0.5, 0.6], requires_grad=True)
    old = torch.zeros(5)
    adv = torch.tensor([0.0, 0.0])
    lengths = torch.tensor([2, 3])

    seq_new, _, _ = GSPO._reduce_to_sequences(new, old, adv, lengths)
    seq_new.sum().backward()

    # d(mean over seq)/d(token) = 1/len_of_its_seq
    expected_grad = torch.tensor([1 / 2, 1 / 2, 1 / 3, 1 / 3, 1 / 3])
    assert torch.allclose(new.grad, expected_grad, atol=1e-6)


def test_reduce_to_sequences_drops_zero_length():
    new = torch.tensor([0.1, 0.2, 0.9], requires_grad=True)
    old = torch.zeros(3)
    adv = torch.tensor([5.0, 6.0, 7.0, 8.0])  # 4 sequences
    lengths = torch.tensor([0, 2, 0, 1])  # only seq 1 (len2) and seq 3 (len1) survive

    seq_new, seq_old, seq_adv = GSPO._reduce_to_sequences(new, old, adv, lengths)

    assert seq_new.shape[0] == 2
    assert torch.allclose(seq_new, torch.stack([new[0:2].mean(), new[2:3].mean()]), atol=1e-6)
    assert torch.allclose(seq_adv, torch.tensor([6.0, 8.0]), atol=1e-6)


def test_gspo_multi_sequence_matches_reference_randomized():
    torch.manual_seed(0)
    lengths = [3, 1, 4, 2]
    total = sum(lengths)
    for _ in range(20):
        old = torch.randn(total)
        new = (old + 0.25 * torch.randn(total)).detach().requires_grad_(True)
        adv = (2.0 * torch.rand(len(lengths)) - 1.0)  # in [-1, 1]

        res = _run_gspo(new, old, lengths, adv, clip_range=0.1, clip_range_high=0.3)
        ref = _gspo_reference_loss(new, old, lengths, adv, clip_low=0.1, clip_high=0.3)

        assert abs(res.loss - ref) < 1e-4, (res.loss, ref)
        assert res.has_backward and new.grad is not None
        assert torch.isfinite(new.grad).all()


# ---------------------------------------------------------------------------
# Clip-range schedule + loss_scale + metric surface.
# ---------------------------------------------------------------------------


def test_gspo_linear_decay_schedule_resolves_clip_range():
    lengths = [2]
    old = torch.tensor([-1.0, -1.0])
    new = old.clone().requires_grad_(True)
    alg = GSPO(stage=_FakeStage(new), clip_range=0.1, clip_schedule="linear_decay", conditions_cls=None)
    res = alg.compute_loss_and_backward(
        conditions={}, segment=_make_segment(old, lengths),
        advantages=torch.tensor([1.0]), training_progress=1.0, loss_scale=1.0,
    )
    # linear_decay: clip_range * (1 - 0.5 * progress) = 0.1 * 0.5 = 0.05
    assert abs(res.metrics["clip_range"] - 0.05) < 1e-6


def test_gspo_loss_scale_scales_gradient():
    lengths = [2, 3]
    old = torch.tensor([-1.0, -1.0, -2.0, -2.0, -2.0])
    adv = torch.tensor([1.0, -0.5])

    def grad_for(scale):
        new = torch.zeros(5, requires_grad=True)
        seg = _make_segment(old, lengths)
        alg = GSPO(stage=_FakeStage(new), clip_range=10.0, conditions_cls=None)
        alg.compute_loss_and_backward(
            conditions={}, segment=seg, advantages=adv, training_progress=0.0, loss_scale=scale
        )
        return new.grad.clone()

    g_full = grad_for(1.0)
    g_half = grad_for(0.5)
    assert torch.allclose(g_half, 0.5 * g_full, atol=1e-6)


def test_gspo_metrics_surface_and_values():
    lengths = [1]
    old = torch.tensor([0.0])
    new = torch.tensor([0.5], requires_grad=True)
    adv = torch.tensor([1.0])

    res = _run_gspo(new, old, lengths, adv, clip_range=0.1)
    m = res.metrics

    for key in (
        "policy_loss", "clip_range", "ratio_mean", "ratio_max", "approx_kl",
        "clip_fraction", "rollout_replay_logp_absdiff_mean", "rollout_replay_logp_absdiff_max",
    ):
        assert key in m, key

    assert abs(m["ratio_mean"] - math.exp(0.5)) < 1e-4
    assert abs(m["ratio_max"] - math.exp(0.5)) < 1e-4
    assert abs(m["approx_kl"] - 0.5 * 0.5**2) < 1e-5  # 0.5 * s^2 with s=0.5
    assert abs(m["rollout_replay_logp_absdiff_mean"] - 0.5) < 1e-5
