"""CPU contract tests for AgenticTrainer's assembly logic (LIN-519).

Exercises the pure, error-prone pieces of ``train_step`` in isolation — ``<answer>``
extraction, group-relative GRPO advantage over a flat trajectory list (completion
order), and the DP-divisibility padding — without a GPU / Ray / the full trainer.
The end-to-end multi-turn on-policy run is the M1 GPU recipe.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")  # the unirl types import torch at module load

import torch  # noqa: E402

from unirl.trainer.agentic import AgenticTrainer, _extract_answer  # noqa: E402
from unirl.types.sample import Part  # noqa: E402

# --------------------------------------------------------------------------- #
# <answer> extraction
# --------------------------------------------------------------------------- #


def test_extract_answer_last_wins_and_fallback():
    assert _extract_answer("blah <answer> 42 </answer> tail") == "42"
    assert _extract_answer("<answer>a</answer> ... <answer> b </answer>") == "b"  # last wins
    # no tag -> whole text (the verifier is tolerant of an unwrapped / \boxed{} answer)
    assert _extract_answer("the answer is \\boxed{7}") == "the answer is \\boxed{7}"
    assert _extract_answer("") == ""
    assert _extract_answer(None) == ""
    assert _extract_answer("<answer>\nline1\nline2\n</answer>") == "line1\nline2"


# --------------------------------------------------------------------------- #
# group-relative GRPO advantage over the flat trajectory list
# --------------------------------------------------------------------------- #


def _adv(scope, by_std, rewards, group_ids):
    stub = SimpleNamespace(adv_normalization_scope=scope, normalize_adv_by_std=by_std)
    return AgenticTrainer._group_advantages(stub, torch.tensor(rewards), group_ids)


def test_group_advantages_group_zscore():
    # p0=[1,0] -> mean .5 popstd .5 -> [1,-1]; p1=[1,1] -> std 0 -> [0,0]
    adv = _adv("group", True, [1.0, 0.0, 1.0, 1.0], ["p0", "p0", "p1", "p1"])
    assert torch.allclose(adv, torch.tensor([1.0, -1.0, 0.0, 0.0]), atol=1e-4)


def test_group_advantages_completion_order_non_contiguous():
    # siblings interleaved (the engine returns trajectories in completion order):
    # grouping must be by root id, not position.
    adv = _adv("group", True, [1.0, 1.0, 0.0, 1.0], ["p0", "p1", "p0", "p1"])
    # p0=[idx0:1, idx2:0] -> [1,-1]; p1=[idx1:1, idx3:1] -> [0,0]
    assert torch.allclose(adv, torch.tensor([1.0, 0.0, -1.0, 0.0]), atol=1e-4)


def test_group_advantages_mean_center_only():
    adv = _adv("group", False, [1.0, 0.0, 1.0, 1.0], ["p0", "p0", "p1", "p1"])
    assert torch.allclose(adv, torch.tensor([0.5, -0.5, 0.0, 0.0]), atol=1e-4)


def test_group_advantages_global_is_mean_zero():
    adv = _adv("global", True, [1.0, 0.0, 1.0, 1.0], ["p0", "p0", "p1", "p1"])
    assert adv.shape == (4,)
    assert abs(float(adv.mean())) < 1e-5


# --------------------------------------------------------------------------- #
# DP-divisibility padding
# --------------------------------------------------------------------------- #


def _pad(part, dp):
    stub = SimpleNamespace(stack=SimpleNamespace(dp_size=dp), num_devices=dp)
    return AgenticTrainer._pad_to_dp_multiple(stub, part)


def test_pad_to_dp_multiple_pads_with_zero_advantage():
    part = Part(sample_ids=["a", "b", "c"], advantages=torch.tensor([1.0, 2.0, 3.0]))
    out = _pad(part, 2)
    assert out.batch_size == 4  # 3 -> 4 (a multiple of dp=2)
    assert float(out.advantages[-1]) == 0.0  # pad row -> zero gradient
    assert torch.allclose(out.advantages[:3], torch.tensor([1.0, 2.0, 3.0]))  # real rows preserved


def test_pad_to_dp_multiple_noop_when_divisible():
    part = Part(sample_ids=["a", "b"], advantages=torch.tensor([1.0, 2.0]))
    out = _pad(part, 2)
    assert out.batch_size == 2
    assert out is part  # exact no-op when already divisible
