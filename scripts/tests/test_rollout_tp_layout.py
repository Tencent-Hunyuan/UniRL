"""Tier 0 smoke tests — rollout TP/EP/PP rank layout (pure Python, no GPU).

Covers the ``handle.py`` rank-layout core that carves a rollout Handle's
``world_size`` into ``(dp, tp, pp, ep)`` groups and populates ``RankInfo``:

  - ``_build_rank_infos`` mapping (flat, TP, PP, SP)
  - ``_parallel_shape_from_init_kwargs`` resolution (config, sibling HandleRef,
    non-SGLang isolation, sp/tp mutual exclusion, divisibility)
  - the per-engine device-slice + NCCL ``rank_offset`` math the trainer wiring
    and ``NCCLWeightSync.connect`` rely on
  - the ``tp_size=1`` baseline is bit-identical to the pre-change flat layout

Run:  pytest scripts/tests/test_rollout_tp_layout.py
"""

from __future__ import annotations

import pytest

from unirl.distributed.group.handle import (
    HandleRef,
    _build_rank_infos,
    _parallel_shape_from_init_kwargs,
)
from unirl.rollout.engine.sglang.engine import SGLangRolloutEngine


class _NonSGLangRole:
    """Stand-in for a non-SGLang role (must ignore config tp_size)."""


# --------------------------------------------------------------------------- #
# _build_rank_infos
# --------------------------------------------------------------------------- #


def test_flat_layout_tp1_is_baseline():
    r = _build_rank_infos(4)
    assert [x.dp_rank for x in r] == [0, 1, 2, 3]
    assert [x.dp_size for x in r] == [4, 4, 4, 4]
    assert all(x.tp_size == 1 and x.tp_rank == 0 for x in r)
    assert all(x.pp_size == 1 and x.sp_size == 1 for x in r)


def test_tp2_world4_two_engines():
    r = _build_rank_infos(4, tp_size=2)
    # Two DP groups, each spanning a contiguous TP pair.
    assert [x.dp_rank for x in r] == [0, 0, 1, 1]
    assert [x.tp_rank for x in r] == [0, 1, 0, 1]
    assert [x.dp_size for x in r] == [2, 2, 2, 2]
    assert all(x.tp_size == 2 for x in r)


def test_tp2_world8_dp4():
    r = _build_rank_infos(8, tp_size=2)
    assert [x.dp_rank for x in r] == [0, 0, 1, 1, 2, 2, 3, 3]
    assert [x.tp_rank for x in r] == [0, 1, 0, 1, 0, 1, 0, 1]
    assert all(x.dp_size == 4 for x in r)


def test_pp2_layout_marks_last_stage():
    r = _build_rank_infos(4, pp_size=2)
    # tp fastest (=1 here), then pp: ranks -> pp_rank [0,1,0,1], dp [0,0,1,1]
    assert [x.pp_rank for x in r] == [0, 1, 0, 1]
    assert [x.dp_rank for x in r] == [0, 0, 1, 1]
    assert [x.is_pipeline_last_stage for x in r] == [False, True, False, True]


def test_sp_layout_unchanged():
    r = _build_rank_infos(4, sp_size=2)
    assert [x.dp_rank for x in r] == [0, 0, 1, 1]
    assert [x.sp_rank for x in r] == [0, 1, 0, 1]
    assert all(x.tp_size == 1 for x in r)


def test_ep_size_is_recorded_but_layout_is_sglang_internal():
    r = _build_rank_infos(4, tp_size=2, ep_size=2)
    # EP is sharded inside SGLang; UniRL only records the requested size.
    assert all(x.ep_size == 2 and x.ep_rank == 0 for x in r)


# --------------------------------------------------------------------------- #
# _parallel_shape_from_init_kwargs
# --------------------------------------------------------------------------- #


def test_shape_reads_tp_from_sglang_config():
    sp, tp, pp, ep = _parallel_shape_from_init_kwargs({"config": {"tp_size": 2}}, 4, SGLangRolloutEngine)
    assert (sp, tp, pp, ep) == (1, 2, 1, 1)


def test_shape_reads_pp_ep_from_sglang_config():
    sp, tp, pp, ep = _parallel_shape_from_init_kwargs(
        {"config": {"tp_size": 2, "ep_size": 2}}, 8, SGLangRolloutEngine
    )
    assert (sp, tp, pp, ep) == (1, 2, 1, 2)


def test_non_sglang_role_ignores_config_tp():
    # A diffusion engine carries its own tp_size config that must NOT drive the
    # AR rollout Handle layout.
    sp, tp, pp, ep = _parallel_shape_from_init_kwargs({"config": {"tp_size": 8}}, 4, _NonSGLangRole)
    assert tp == 1


def test_sibling_handleref_inherits_tp():
    # Colocated weight sync gets rollout=<HandleRef tp_size=2> and must adopt it.
    ref = HandleRef(role_name="rollout", tp_size=2)
    sp, tp, pp, ep = _parallel_shape_from_init_kwargs({"rollout": ref}, 4, _NonSGLangRole)
    assert tp == 2


def test_sp_and_tp_mutually_exclusive():
    with pytest.raises(ValueError):
        _parallel_shape_from_init_kwargs({"sp_size": 2, "config": {"tp_size": 2}}, 4, SGLangRolloutEngine)


def test_world_must_divide_tp_times_pp():
    with pytest.raises(ValueError):
        _parallel_shape_from_init_kwargs({"config": {"tp_size": 3}}, 4, SGLangRolloutEngine)


def test_empty_init_kwargs_is_flat():
    assert _parallel_shape_from_init_kwargs(None, 4, SGLangRolloutEngine) == (1, 1, 1, 1)
    assert _parallel_shape_from_init_kwargs({}, 4, _NonSGLangRole) == (1, 1, 1, 1)


# --------------------------------------------------------------------------- #
# Derived math the trainer wiring + NCCLWeightSync.connect depend on
# --------------------------------------------------------------------------- #


def test_engine_device_slices_are_contiguous_and_disjoint():
    device_ids = [0, 1, 2, 3]
    r = _build_rank_infos(4, tp_size=2)
    slices = {}
    for i, ri in enumerate(r):
        if ri.tp_rank != 0:
            continue
        engine_index = ri.dp_rank * ri.pp_size + ri.pp_rank
        start = engine_index * ri.tp_size
        slices[engine_index] = device_ids[start : start + ri.tp_size]
    assert slices == {0: [0, 1], 1: [2, 3]}


def test_tp_zero_worker_indices():
    r = _build_rank_infos(8, tp_size=2)
    tp_zero = [i for i, ri in enumerate(r) if ri.tp_rank == 0]
    assert tp_zero == [0, 2, 4, 6]


@pytest.mark.parametrize(
    "num_engines,tp_size,expected_offsets,expected_world",
    [
        (4, 1, [1, 2, 3, 4], 5),  # baseline: rank_offset = i + 1
        (2, 2, [1, 3], 5),  # 2 engines x tp2 => offsets 1,3 ; world 2*2+1
        (4, 2, [1, 3, 5, 7], 9),  # 4 engines x tp2 => world 4*2+1
    ],
)
def test_nccl_rank_offset_formula(num_engines, tp_size, expected_offsets, expected_world):
    offsets = [i * tp_size + 1 for i in range(num_engines)]
    world = num_engines * tp_size + 1
    assert offsets == expected_offsets
    assert world == expected_world
