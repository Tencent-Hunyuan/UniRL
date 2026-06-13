"""Unit tests for micro-batch planning (unirl/train/stack.py) + the
TokenBudgetPlanner seq-mean guard. Pure / CPU-only — no GPU, no FSDP backend.

Run with ``pytest tests/train/test_packing.py`` or directly:
``python tests/train/test_packing.py``.
"""

from __future__ import annotations

import types

import pytest
import torch

from unirl.train.stack import (
    TokenBudgetPlanner,
    _arrange_packed,
    _build_micro_batch_slices,
    _count_plan,
    _pack_micros,
    _pack_micros_2d,
    _pack_micros_sum,
    _partition_into_k,
    _sync_micro_count,
    _update_ranges,
)


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


def _fake_track(lengths):
    """Minimal RolloutTrack stand-in: just batch_size + segment.lengths."""
    seg = types.SimpleNamespace(lengths=torch.tensor(lengths, dtype=torch.long))
    return types.SimpleNamespace(batch_size=len(lengths), segment=seg, conditions={})


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
# token-budget packing — coverage + budget invariants
# --------------------------------------------------------------------------- #
DENSE_LENGTHS = [4000, 3500, 300, 250, 200, 180, 150, 120]


def test_pack_micros_dense_covers_and_respects_budget():
    bins = _pack_micros(indices=list(range(8)), lengths=DENSE_LENGTHS, token_budget=10240)
    assert _covers(bins, list(range(8)))
    for b in bins:
        cost = max(DENSE_LENGTHS[i] for i in b) * len(b)
        assert cost <= 10240 or len(b) == 1  # single oversize seq allowed its own bin
    # the two long seqs cannot share with the shorts under 10240
    assert all(b for b in bins)  # no empty bins


def test_pack_micros_sum_covers_and_respects_budget():
    bins = _pack_micros_sum(indices=list(range(8)), lengths=DENSE_LENGTHS, token_budget=8000)
    assert _covers(bins, list(range(8)))
    for b in bins:
        cost = sum(DENSE_LENGTHS[i] for i in b)
        assert cost <= 8000 or len(b) == 1


def test_pack_micros_2d_covers_and_respects_budget():
    prompt = [1000, 50, 900, 40, 30, 20, 10, 5]
    resp = [50, 1000, 40, 900, 200, 180, 150, 120]
    bins = _pack_micros_2d(indices=list(range(8)), prompt_lens=prompt, resp_lens=resp, token_budget=4096)
    assert _covers(bins, list(range(8)))
    for b in bins:
        cost = (max(prompt[i] for i in b) + max(resp[i] for i in b)) * len(b)
        assert cost <= 4096 or len(b) == 1


def test_oversize_sequence_gets_its_own_bin():
    # one seq longer than the whole budget must still be placed (never dropped)
    bins = _pack_micros(indices=[0, 1, 2], lengths=[50, 99999, 60], token_budget=1024)
    assert _covers(bins, [0, 1, 2])
    big = next(b for b in bins if 1 in b)
    assert big == [1]


def test_pack_micros_rejects_nonpositive_budget():
    with pytest.raises(ValueError):
        _pack_micros(indices=[0, 1], lengths=[10, 20], token_budget=0)


# --------------------------------------------------------------------------- #
# exact-K re-partition (NCCL micro-count parity)
# --------------------------------------------------------------------------- #
def test_partition_into_k_exact_and_covers():
    idx = list(range(8))
    for k in (1, 2, 3, 5, 8):
        bins = _partition_into_k(indices=idx, lengths=DENSE_LENGTHS, k=k)
        assert len(bins) == k
        assert all(len(b) >= 1 for b in bins)  # every bin non-empty
        assert _covers(bins, idx)


def test_partition_into_k_out_of_range():
    with pytest.raises(ValueError):
        _partition_into_k(indices=[0, 1, 2], lengths=[1, 2, 3], k=0)
    with pytest.raises(ValueError):
        _partition_into_k(indices=[0, 1, 2], lengths=[1, 2, 3], k=4)  # k > n


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
    track = _fake_track(lengths)
    out = _arrange_packed(track, num_updates=2, token_budget=10240, cost_model="dense")
    assert out is not None
    perm, plan = out
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
    track = _fake_track(lengths)
    _, plan = _arrange_packed(track, num_updates=2, token_budget=10240, cost_model="dense")
    for update in plan:
        update_total = sum(e - s for s, e in update)
        weights = [(e - s) / update_total for s, e in update]
        assert update_total == 4
        assert abs(sum(weights) - 1.0) < 1e-12


def test_arrange_packed_falls_back_when_no_lengths():
    track = types.SimpleNamespace(batch_size=4, segment=None, conditions={})
    assert _arrange_packed(track, num_updates=2, token_budget=1024, cost_model="dense") is None


def test_arrange_packed_picks_up_prompt_from_conditions_dict():
    # review #42 B2: conditions is a Dict, so prompt lengths must be read via dict
    # access — otherwise the budget counts response tokens only. With prompt=50,
    # resp=100 the 2D cost is (50+100)*count<=300 -> <=2 per micro; if the prompt
    # were ignored it'd be 100*count<=300 -> 3 per micro. The cap distinguishes them.
    seg = types.SimpleNamespace(lengths=torch.tensor([100, 100, 100, 100], dtype=torch.long))
    prompt = types.SimpleNamespace(attention_mask=torch.ones(4, 50, dtype=torch.long))
    track = types.SimpleNamespace(batch_size=4, segment=seg, conditions={"prompt": prompt})
    out = _arrange_packed(track, num_updates=1, token_budget=300, cost_model="dense")
    assert out is not None
    perm, plan = out
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
