"""Unit tests for micro-batch planning (unirl/train/stack/planner/) + the
TokenBudgetPlanner seq-mean guard. Pure / CPU-only — no GPU, no FSDP backend.

Run with ``pytest tests/train/test_packing.py`` or directly:
``python tests/train/test_packing.py``.
"""

from __future__ import annotations

import types

import pytest
import torch

from unirl.train.stack import TokenBudgetPlanner
from unirl.train.stack.planner.count import _count_plan
from unirl.train.stack.planner.packed import (
    _arrange_packed,
    _extract_samples,
    _sample,
    _sync_micro_count,
    balance_into_k,
    dense,
    first_fit_decreasing,
    varlen_sum,
)
from unirl.train.stack.planner.types import _build_micro_batch_slices, _update_ranges


def _range_indices(r):
    """Sample indices covered by a contiguous (start, end) micro range."""
    return list(range(r[0], r[1]))


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _flatten(bins):
    return [i for b in bins for i in b]


def _covers(bins, indices):
    """Every index appears exactly once across the bins."""
    flat = _flatten(bins)
    return sorted(flat) == sorted(indices) and len(flat) == len(indices)


def _samples(resp, prompt=None):
    """Build the Sample list a packer consumes (idx = position, clamped sizes)."""
    return [_sample(i, prompt=(0 if prompt is None else prompt[i]), resp=resp[i]) for i in range(len(resp))]


def _idx_bins(micros):
    """Micros (lists of Samples) -> lists of their original indices, for coverage checks."""
    return [[s.idx for s in m] for m in micros]


# --------------------------------------------------------------------------- #
# update / count partitioning
# --------------------------------------------------------------------------- #
def test_update_ranges_even():
    assert _update_ranges(total_size=8, num_updates=2) == ((0, 4), (4, 8))
    assert _update_ranges(total_size=12, num_updates=4) == ((0, 3), (3, 6), (6, 9), (9, 12))


def test_update_ranges_requires_divisibility():
    with pytest.raises(ValueError):
        _update_ranges(total_size=10, num_updates=3)


def test_count_micro_slices_cover():
    sl = _build_micro_batch_slices(total_size=10, micro_batch_size=4)
    assert sl == ((0, 4), (4, 8), (8, 10))  # last partial, full coverage


def test_count_plan_structure():
    # 8 samples, 2 updates, micro_batch_size 1 -> 2 updates x 4 single-sample micros
    plan = _count_plan(total=8, num_updates=2, micro_batch_size=1)
    assert len(plan) == 2
    assert plan[0] == [(0, 1), (1, 2), (2, 3), (3, 4)]
    assert plan[1] == [(4, 5), (5, 6), (6, 7), (7, 8)]


# --------------------------------------------------------------------------- #
# token-budget packing kernels — coverage + budget invariants
# --------------------------------------------------------------------------- #
DENSE_LENGTHS = [4000, 3500, 300, 250, 200, 180, 150, 120]


def test_ffd_dense_covers_and_respects_budget():
    micros = first_fit_decreasing(_samples(DENSE_LENGTHS), cost=dense, budget=10240)
    assert _covers(_idx_bins(micros), list(range(8)))
    for m in micros:
        assert dense(m) <= 10240 or len(m) == 1  # single oversize seq allowed its own micro
    assert all(m for m in micros)  # no empty micros


def test_ffd_varlen_sum_covers_and_respects_budget():
    micros = first_fit_decreasing(_samples(DENSE_LENGTHS), cost=varlen_sum, budget=8000)
    assert _covers(_idx_bins(micros), list(range(8)))
    for m in micros:
        assert varlen_sum(m) <= 8000 or len(m) == 1


def test_ffd_dense_2d_separates_anticorrelated_under_tight_budget():
    # idx0 = big prompt / small resp, idx1 = small prompt / big resp. Together the 2D
    # pad is (1000+1000)*2 = 4000, so a 2048 budget must keep them in separate micros
    # even though each alone (1050) fits — this is exactly what the 2D dense cost buys.
    prompt = [1000, 50]
    resp = [50, 1000]
    micros = first_fit_decreasing(_samples(resp, prompt), cost=dense, budget=2048)
    bins = _idx_bins(micros)
    assert _covers(bins, [0, 1])
    for m in micros:
        assert dense(m) <= 2048 or len(m) == 1
    assert not any(0 in b and 1 in b for b in bins)


def test_oversize_sequence_gets_its_own_micro():
    # one seq longer than the whole budget must still be placed (never dropped)
    micros = first_fit_decreasing(_samples([50, 99999, 60]), cost=dense, budget=1024)
    bins = _idx_bins(micros)
    assert _covers(bins, [0, 1, 2])
    big = next(b for b in bins if 1 in b)
    assert big == [1]


def test_ffd_rejects_nonpositive_budget():
    with pytest.raises(ValueError):
        first_fit_decreasing(_samples([10, 20]), cost=dense, budget=0)


def test_ffd_tie_break_by_idx_is_deterministic():
    # equal totals -> ordered by idx; budget fits exactly two per micro, so the first
    # micro holds the lowest indices. Guards against a future sort-key change.
    micros = first_fit_decreasing(_samples([100, 100, 100, 100]), cost=dense, budget=200)
    assert _idx_bins(micros) == [[0, 1], [2, 3]]


def test_sample_clamps_zero_resp():
    # a zero response length is clamped to 1 so total >= 1: a zero-cost sample would
    # otherwise pack unboundedly into one micro.
    s = _sample(0, prompt=0, resp=0)
    assert (s.prompt, s.resp, s.total) == (0, 1, 1)
    micros = first_fit_decreasing(_samples([0, 0, 0, 0]), cost=dense, budget=2)
    assert _covers(_idx_bins(micros), [0, 1, 2, 3])
    for m in micros:
        assert dense(m) <= 2  # clamp makes each row cost 1 -> at most 2 per micro


# --------------------------------------------------------------------------- #
# exact-K re-partition (NCCL micro-count parity)
# --------------------------------------------------------------------------- #
def test_balance_into_k_exact_and_covers():
    samples = _samples(DENSE_LENGTHS)
    for k in (1, 2, 3, 5, 8):
        micros = balance_into_k(samples, cost=dense, k=k)
        assert len(micros) == k
        assert all(len(m) >= 1 for m in micros)  # every micro non-empty
        assert _covers(_idx_bins(micros), list(range(8)))


def test_balance_into_k_out_of_range():
    samples = _samples([1, 2, 3])
    with pytest.raises(ValueError):
        balance_into_k(samples, cost=dense, k=0)
    with pytest.raises(ValueError):
        balance_into_k(samples, cost=dense, k=4)  # k > n


def test_sync_micro_count_noop_without_dist():
    # torch.distributed not initialized in a unit test -> returns the local count
    assert _sync_micro_count(7) == 7


# --------------------------------------------------------------------------- #
# plan equivalence: packing only regroups, never changes which samples an update trains on
# --------------------------------------------------------------------------- #
def test_packed_arrange_preserves_update_membership():
    # sort-then-slice: arrange reorders the track but each update's permuted block
    # must hold exactly its original [u*4, u*4+4) samples (membership unchanged), and
    # the plan's ranges must contiguously cover that permuted block.
    lengths = [4000, 3500, 300, 250, 200, 180, 150, 120]
    perm, plan = _arrange_packed(_samples(lengths), num_updates=2, token_budget=10240, cost_model="dense")
    assert sorted(perm) == list(range(8))  # a full permutation, no sample lost
    assert len(plan) == 2
    for u, update in enumerate(plan):
        block = list(range(u * 4, (u + 1) * 4))
        # the permuted positions in this update map back to the original update's samples
        assert sorted(perm[p] for p in block) == block
        # ranges contiguously tile the permuted update block
        covered = [p for m in update for p in _range_indices(m)]
        assert covered == block


def test_sample_share_weights_sum_to_one_per_update():
    lengths = [4000, 3500, 300, 250, 200, 180, 150, 120]
    _, plan = _arrange_packed(_samples(lengths), num_updates=2, token_budget=10240, cost_model="dense")
    for update in plan:
        update_total = sum(e - s for s, e in update)
        weights = [(e - s) / update_total for s, e in update]
        assert update_total == 4
        assert abs(sum(weights) - 1.0) < 1e-12


def test_extract_samples_returns_none_without_lengths():
    track = types.SimpleNamespace(batch_size=4, segment=None, conditions={})
    assert _extract_samples(track) is None


def test_arrange_falls_back_to_count_plan_without_lengths():
    # the fallback path returns a count plan and does NOT touch the track (no .select)
    track = types.SimpleNamespace(batch_size=4, segment=None, conditions={})
    out_track, plan = TokenBudgetPlanner(token_budget=1024).arrange(track, num_updates=2, micro_batch_size=1)
    assert out_track is track
    assert plan == _count_plan(total=4, num_updates=2, micro_batch_size=1)


def test_arrange_packed_picks_up_prompt_from_conditions_dict():
    # review #42 B2: conditions is a Dict, so prompt lengths must be read via dict
    # access — otherwise the budget counts response tokens only. With prompt=50,
    # resp=100 the 2D cost is (50+100)*count<=300 -> <=2 per micro; if the prompt
    # were ignored it'd be 100*count<=300 -> 3 per micro. The cap distinguishes them.
    seg = types.SimpleNamespace(lengths=torch.tensor([100, 100, 100, 100], dtype=torch.long))
    prompt = types.SimpleNamespace(attention_mask=torch.ones(4, 50, dtype=torch.long))
    track = types.SimpleNamespace(batch_size=4, segment=seg, conditions={"prompt": prompt})
    samples = _extract_samples(track)
    assert samples is not None
    assert all(s.prompt == 50 for s in samples)  # prompt read via the dict accessor
    perm, plan = _arrange_packed(samples, num_updates=1, token_budget=300, cost_model="dense")
    assert sorted(perm) == [0, 1, 2, 3]
    assert all((e - s) <= 2 for s, e in plan[0])  # prompt counted -> 2D cap


# --------------------------------------------------------------------------- #
# TokenBudgetPlanner seq-mean guard
# --------------------------------------------------------------------------- #
def _algo(mode):
    return types.SimpleNamespace(loss_agg_mode=mode)


def _planner():
    return TokenBudgetPlanner(token_budget=1024)


@pytest.mark.parametrize("mode", ["seq-mean-token-sum-norm", "seq-mean-token-mean"])
def test_guard_allows_seq_mean(mode):
    _planner().validate(_algo(mode))  # must not raise


@pytest.mark.parametrize("mode", ["token-mean", "something-else", None])
def test_guard_rejects_non_seq_mean(mode):
    with pytest.raises(ValueError):
        _planner().validate(_algo(mode))


def test_guard_rejects_algo_without_agg_mode():
    with pytest.raises(ValueError):
        _planner().validate(types.SimpleNamespace())  # no loss_agg_mode attr


# --------------------------------------------------------------------------- #
# direct runner (no pytest required)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        marks = getattr(fn, "pytestmark", [])
        param_sets = None
        for m in marks:
            if m.name == "parametrize":
                param_sets = m.args[1]
        cases = [(v,) for v in param_sets] if param_sets is not None else [()]
        for args in cases:
            try:
                fn(*args)
                print(f"PASS {name}{args if args else ''}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}{args if args else ''}: {type(exc).__name__}: {exc}")
    print(f"\n{'OK' if failures == 0 else f'{failures} FAILURE(S)'}")
    raise SystemExit(1 if failures else 0)
