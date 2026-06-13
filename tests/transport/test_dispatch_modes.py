"""Unit tests for dispatch/collect functions, the registry, and @distributed.

The dispatch layer (``unirl/distributed/group/dispatch.py``) is the single source
of truth for how a controller-side call's ``(args, kwargs)`` fans out to SPMD
workers and how their results fold back. Each :class:`Dispatch` mode pairs a
``dispatch_fn`` (``wg, args, kwargs, batch_size -> List[Shard]``) with a
``collect_fn`` (``wg, results -> collected``) in ``DISPATCH_MODE_REGISTRY``.

Pure-CPU: the only thing these functions read off ``wg`` is ``.world_size`` /
``.dp_size`` / ``.rank_infos`` (a list of real :class:`RankInfo`), so a tiny fake
``wg`` (``types.SimpleNamespace``) stands in for the whole Handle — no Ray, no
workers, no GPU. ``pytree_chunk`` / ``pytree_cat`` do the actual splitting/merging
along dim 0, so plain ``torch`` tensors exercise the per-worker shard math.
"""

import types

import pytest
import torch

from unirl.distributed.group.dispatch import (
    DISPATCH_MODE_REGISTRY,
    DISTRIBUTED_CONFIG_ATTR,
    Dispatch,
    Execute,
    _collect_dp_merge,
    _collect_passthrough,
    _dispatch_broadcast,
    _dispatch_dp_scatter,
    _dispatch_dp_scatter_head,
    _dispatch_scatter,
    _is_dp_head,
    distributed,
    resolve_backward_dispatch_mode,
)
from unirl.distributed.group.remote import RankInfo
from unirl.distributed.utils import Broadcast

pytestmark = pytest.mark.cpu


def _wg(rank_infos):
    """Minimal Handle-like object exposing exactly what dispatch/collect read.

    ``world_size`` / ``dp_size`` are derived from the supplied RankInfos so the
    three never drift apart in a test.
    """
    world_size = len(rank_infos)
    dp_size = max((ri.dp_size for ri in rank_infos), default=1)
    return types.SimpleNamespace(world_size=world_size, dp_size=dp_size, rank_infos=rank_infos)


def _ranks_dp_only(dp_size):
    """world == dp_size, every other parallel axis trivial (the plain-DP topology)."""
    return [RankInfo(rank=i, world_size=dp_size, dp_rank=i, dp_size=dp_size) for i in range(dp_size)]


# ── _dispatch_broadcast ──────────────────────────────────────────────────────


def test_dispatch_broadcast_replicates_to_every_worker():
    wg = _wg(_ranks_dp_only(3))
    x = torch.arange(6).reshape(3, 2)
    shards = _dispatch_broadcast(wg, (x,), {"k": 5}, batch_size=3)
    assert len(shards) == 3
    # every worker gets the identical (args, kwargs) payload
    for args, kwargs in shards:
        assert args[0] is x and kwargs == {"k": 5}


def test_dispatch_broadcast_unwraps_broadcast_values():
    # Broadcast is a controller-side annotation: it must be consumed here and the
    # raw .value handed to workers, never the wrapper.
    wg = _wg(_ranks_dp_only(2))
    cfg = {"lr": 0.1}
    shards = _dispatch_broadcast(wg, (Broadcast(7),), {"cfg": Broadcast(cfg)}, batch_size=None)
    for args, kwargs in shards:
        assert args == (7,)
        assert kwargs["cfg"] is cfg and not isinstance(kwargs["cfg"], Broadcast)


# ── _dispatch_scatter ────────────────────────────────────────────────────────


def test_dispatch_scatter_none_batch_size_broadcasts():
    wg = _wg(_ranks_dp_only(2))
    shards = _dispatch_scatter(wg, (Broadcast(9),), {}, batch_size=None)
    assert len(shards) == 2
    for args, kwargs in shards:
        assert args == (9,)  # Broadcast unwrapped on the broadcast fallback


def test_dispatch_scatter_splits_by_world_size():
    # SCATTER treats each worker as its own DP rank: chunk by world_size.
    wg = _wg(_ranks_dp_only(2))
    x = torch.arange(8).reshape(4, 2)
    shards = _dispatch_scatter(wg, (x,), {}, batch_size=4)
    assert len(shards) == 2
    assert torch.equal(shards[0][0][0], x[:2])
    assert torch.equal(shards[1][0][0], x[2:])


def test_dispatch_scatter_splits_kwargs_too():
    wg = _wg(_ranks_dp_only(2))
    y = torch.arange(4)
    shards = _dispatch_scatter(wg, (), {"y": y}, batch_size=4)
    assert torch.equal(shards[0][1]["y"], y[:2])
    assert torch.equal(shards[1][1]["y"], y[2:])


def test_dispatch_scatter_indivisible_raises():
    # pytree_chunk enforces divisibility of batch_size by the chunk count.
    wg = _wg(_ranks_dp_only(3))
    x = torch.arange(8).reshape(4, 2)
    with pytest.raises(ValueError):
        _dispatch_scatter(wg, (x,), {}, batch_size=4)


# ── _dispatch_dp_scatter ─────────────────────────────────────────────────────


def test_dispatch_dp_scatter_same_dp_group_gets_identical_shard():
    # 2 DP groups x 2 TP ranks = 4 workers; workers sharing a dp_rank get the
    # SAME shard (TP slicing is the worker's own job).
    rank_infos = [
        RankInfo(rank=0, world_size=4, dp_rank=0, dp_size=2, tp_rank=0, tp_size=2),
        RankInfo(rank=1, world_size=4, dp_rank=0, dp_size=2, tp_rank=1, tp_size=2),
        RankInfo(rank=2, world_size=4, dp_rank=1, dp_size=2, tp_rank=0, tp_size=2),
        RankInfo(rank=3, world_size=4, dp_rank=1, dp_size=2, tp_rank=1, tp_size=2),
    ]
    wg = _wg(rank_infos)
    x = torch.arange(8).reshape(4, 2)
    shards = _dispatch_dp_scatter(wg, (x,), {}, batch_size=4)
    assert len(shards) == 4
    # dp_rank 0 → first half; both its workers identical
    assert torch.equal(shards[0][0][0], x[:2])
    assert torch.equal(shards[1][0][0], x[:2])
    # dp_rank 1 → second half
    assert torch.equal(shards[2][0][0], x[2:])
    assert torch.equal(shards[3][0][0], x[2:])


def test_dispatch_dp_scatter_none_batch_size_broadcasts_to_all():
    rank_infos = [
        RankInfo(rank=0, world_size=2, dp_rank=0, dp_size=2),
        RankInfo(rank=1, world_size=2, dp_rank=1, dp_size=2),
    ]
    wg = _wg(rank_infos)
    shards = _dispatch_dp_scatter(wg, (Broadcast(3),), {}, batch_size=None)
    assert len(shards) == 2  # one per worker, not per dp group
    for args, _ in shards:
        assert args == (3,)


# ── _dispatch_dp_scatter_head ────────────────────────────────────────────────


def test_dispatch_dp_scatter_head_only_head_gets_shard():
    # head = tp_rank==0 & pp_rank==0 & sp_rank==0; non-heads get empty ((), {}).
    rank_infos = [
        RankInfo(rank=0, world_size=2, dp_rank=0, dp_size=1, tp_rank=0, tp_size=2),  # head
        RankInfo(rank=1, world_size=2, dp_rank=0, dp_size=1, tp_rank=1, tp_size=2),  # non-head
    ]
    wg = _wg(rank_infos)
    x = torch.arange(4).reshape(2, 2)
    shards = _dispatch_dp_scatter_head(wg, (x,), {}, batch_size=2)
    assert torch.equal(shards[0][0][0], x)  # whole dp_rank-0 shard to the head
    assert shards[1] == ((), {})  # non-head gets nothing


def test_dispatch_dp_scatter_head_none_batch_size_head_only():
    rank_infos = [
        RankInfo(rank=0, world_size=2, dp_rank=0, dp_size=1, tp_rank=0, tp_size=2),  # head
        RankInfo(rank=1, world_size=2, dp_rank=0, dp_size=1, tp_rank=1, tp_size=2),  # non-head
    ]
    wg = _wg(rank_infos)
    shards = _dispatch_dp_scatter_head(wg, (Broadcast(5),), {}, batch_size=None)
    assert shards[0][0] == (5,)  # head gets the (unwrapped) broadcast value
    assert shards[1] == ((), {})  # non-head still empty


def test_is_dp_head_predicate():
    assert _is_dp_head(RankInfo(tp_rank=0, pp_rank=0, sp_rank=0))
    assert not _is_dp_head(RankInfo(tp_rank=1, pp_rank=0, sp_rank=0))
    assert not _is_dp_head(RankInfo(tp_rank=0, pp_rank=1, sp_rank=0))
    assert not _is_dp_head(RankInfo(tp_rank=0, pp_rank=0, sp_rank=1))


# ── _collect_passthrough ─────────────────────────────────────────────────────


def test_collect_passthrough_preserves_order():
    wg = _wg(_ranks_dp_only(3))
    results = ["a", "b", "c"]
    out = _collect_passthrough(wg, results)
    assert out == ["a", "b", "c"]
    assert out is results  # returned raw


# ── _collect_dp_merge ────────────────────────────────────────────────────────


def test_collect_dp_merge_keeps_only_dp_heads_and_cats():
    # 2 DP groups x 2 TP ranks; only tp_rank==0 (and pp-last, sp_rank==0) results
    # are merged. Non-head results are dropped (workers replicate internally).
    rank_infos = [
        RankInfo(rank=0, world_size=4, dp_rank=0, dp_size=2, tp_rank=0, tp_size=2),  # keep
        RankInfo(rank=1, world_size=4, dp_rank=0, dp_size=2, tp_rank=1, tp_size=2),  # drop
        RankInfo(rank=2, world_size=4, dp_rank=1, dp_size=2, tp_rank=0, tp_size=2),  # keep
        RankInfo(rank=3, world_size=4, dp_rank=1, dp_size=2, tp_rank=1, tp_size=2),  # drop
    ]
    wg = _wg(rank_infos)
    a = torch.arange(2).reshape(2, 1)
    poison = torch.full((2, 1), -1)
    b = torch.arange(2, 4).reshape(2, 1)
    out = _collect_dp_merge(wg, [a, poison, b, poison])
    # only a and b survive, cat'd in worker order
    assert torch.equal(out, torch.cat([a, b], dim=0))


def test_collect_dp_merge_pipeline_last_stage_only():
    # is_pipeline_last_stage gate: a non-last PP stage's result is dropped even at
    # tp_rank==0 (its tensor is an intermediate, not the final output).
    rank_infos = [
        RankInfo(rank=0, world_size=2, pp_rank=0, pp_size=2, tp_rank=0),  # not last → drop
        RankInfo(rank=1, world_size=2, pp_rank=1, pp_size=2, tp_rank=0),  # last → keep
    ]
    wg = _wg(rank_infos)
    intermediate = torch.full((2, 1), 9)
    final = torch.arange(2).reshape(2, 1)
    out = _collect_dp_merge(wg, [intermediate, final])
    assert torch.equal(out, final)  # single survivor returned as-is


def test_collect_dp_merge_rank_zero_fewer_results_than_world():
    # Execute.RANK_ZERO → len(results) < world_size; the loop is bounded by
    # len(results), so the extra rank_infos are simply never indexed.
    wg = _wg(_ranks_dp_only(4))
    only = torch.arange(2).reshape(2, 1)
    out = _collect_dp_merge(wg, [only])  # one result, four rank_infos
    assert torch.equal(out, only)  # single survivor returned as-is (no pytree_cat)


def test_collect_dp_merge_empty_returns_none():
    wg = _wg(_ranks_dp_only(2))
    assert _collect_dp_merge(wg, []) is None


def test_collect_dp_merge_no_heads_returns_none():
    # all results are non-heads (tp_rank!=0) → nothing kept → None.
    rank_infos = [
        RankInfo(rank=0, world_size=2, tp_rank=1, tp_size=2),
        RankInfo(rank=1, world_size=2, tp_rank=1, tp_size=2),
    ]
    wg = _wg(rank_infos)
    assert _collect_dp_merge(wg, ["x", "y"]) is None


# ── DISPATCH_MODE_REGISTRY ───────────────────────────────────────────────────


def test_registry_maps_every_dispatch_to_its_fn_pair():
    # The registry is the single binding of mode → (dispatch_fn, collect_fn);
    # assert each entry by identity against the real functions.
    assert set(DISPATCH_MODE_REGISTRY) == set(Dispatch)
    assert DISPATCH_MODE_REGISTRY[Dispatch.BROADCAST]["dispatch_fn"] is _dispatch_broadcast
    assert DISPATCH_MODE_REGISTRY[Dispatch.BROADCAST]["collect_fn"] is _collect_passthrough
    assert DISPATCH_MODE_REGISTRY[Dispatch.SCATTER]["dispatch_fn"] is _dispatch_scatter
    assert DISPATCH_MODE_REGISTRY[Dispatch.SCATTER]["collect_fn"] is _collect_passthrough
    assert DISPATCH_MODE_REGISTRY[Dispatch.DP_SCATTER]["dispatch_fn"] is _dispatch_dp_scatter
    assert DISPATCH_MODE_REGISTRY[Dispatch.DP_SCATTER]["collect_fn"] is _collect_dp_merge
    assert DISPATCH_MODE_REGISTRY[Dispatch.DP_SCATTER_HEAD]["dispatch_fn"] is _dispatch_dp_scatter_head
    assert DISPATCH_MODE_REGISTRY[Dispatch.DP_SCATTER_HEAD]["collect_fn"] is _collect_dp_merge


# ── resolve_backward_dispatch_mode ───────────────────────────────────────────


def test_resolve_backward_dp_scatter_pp1_returns_dp_scatter():
    rank_infos = _ranks_dp_only(2)  # pp_size==1 on all
    out = resolve_backward_dispatch_mode("rollout", Dispatch.DP_SCATTER, rank_infos)
    assert out is Dispatch.DP_SCATTER


def test_resolve_backward_dp_scatter_head_pp1_becomes_dp_scatter():
    # DP_SCATTER_HEAD forward → DP_SCATTER backward (all ranks must participate).
    rank_infos = _ranks_dp_only(2)
    out = resolve_backward_dispatch_mode("forward", Dispatch.DP_SCATTER_HEAD, rank_infos)
    assert out is Dispatch.DP_SCATTER


def test_resolve_backward_broadcast_raises():
    rank_infos = _ranks_dp_only(1)
    with pytest.raises(ValueError):
        resolve_backward_dispatch_mode("get_metrics", Dispatch.BROADCAST, rank_infos)


def test_resolve_backward_scatter_raises():
    rank_infos = _ranks_dp_only(2)
    with pytest.raises(ValueError):
        resolve_backward_dispatch_mode("scatter_call", Dispatch.SCATTER, rank_infos)


def test_resolve_backward_pp_gt_1_raises():
    # autograd graph cannot cross pipeline stages → hard error even for DP_SCATTER.
    rank_infos = [
        RankInfo(rank=0, world_size=2, pp_rank=0, pp_size=2),
        RankInfo(rank=1, world_size=2, pp_rank=1, pp_size=2),
    ]
    with pytest.raises(ValueError):
        resolve_backward_dispatch_mode("forward", Dispatch.DP_SCATTER, rank_infos)


# ── @distributed decorator ───────────────────────────────────────────────────


def test_distributed_defaults_dp_scatter_all():
    @distributed
    def rollout(self):
        return "ran"

    cfg = getattr(rollout, DISTRIBUTED_CONFIG_ATTR)
    assert cfg == {"dispatch_mode": Dispatch.DP_SCATTER, "execute_mode": Execute.ALL}
    assert rollout(None) == "ran"  # wrapper still forwards the call


def test_distributed_with_explicit_modes():
    @distributed(dispatch_mode=Dispatch.BROADCAST, execute_mode=Execute.RANK_ZERO)
    def get_metrics(self):
        return 42

    cfg = getattr(get_metrics, DISTRIBUTED_CONFIG_ATTR)
    assert cfg["dispatch_mode"] is Dispatch.BROADCAST
    assert cfg["execute_mode"] is Execute.RANK_ZERO
    assert get_metrics(None) == 42
