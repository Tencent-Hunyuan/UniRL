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
