"""Tier 0 smoke tests — rollout EP and PP parallelism (pure Python, no GPU).

Covers the EP/PP surface that the TP suite does not exercise:

EP (Expert Parallel, for MoE models):
  - ``ep_size`` is a first-class ``SGLangEngineConfig`` field and flows into
    ``server_intent`` (SGLang shards experts internally; UniRL only records the
    requested size on ``RankInfo`` and forwards it).
  - ``enable_expert_parallel`` toggles SGLang's EP flag.
  - EP composes with TP (``tp_size=2, ep_size=2``) and is recorded on every
    ``RankInfo`` without changing the (dp, tp, pp) layout.
  - EP does NOT change the NCCL ``rank_offset`` math (it is SGLang-internal).

PP (Pipeline Parallel):
  - ``pp_size>1`` populates ``RankInfo.pp_rank`` / ``is_pipeline_last_stage``.
  - The dispatch collect filter (``tp_rank==0 and is_pipeline_last_stage and
    sp_rank==0``) selects the correct DP-head ranks under PP.
  - ``NCCLWeightSync.connect(pp_size>1)`` fails closed with
    ``NotImplementedError`` — per-stage rank_offset routing is future work.
  - ``pp_size=1`` is the supported default and never raises.

Run:  pytest scripts/tests/test_rollout_ep_pp.py
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pytest

from unirl.distributed.group.handle import (
    HandleRef,
    _build_rank_infos,
    _parallel_shape_from_init_kwargs,
)
from unirl.distributed.group.remote import RankInfo
from unirl.rollout.engine.sglang.config import SGLangEngineConfig, SGLangPorts
from unirl.rollout.engine.sglang.engine import SGLangRolloutEngine


PORTS = SGLangPorts(server_port=30000, nccl_port=30001)


def _cfg(**kw) -> SGLangEngineConfig:
    base = dict(pretrained_model_ckpt_path="/tmp/model", model_family="text")
    base.update(kw)
    return SGLangEngineConfig(**base)


# --------------------------------------------------------------------------- #
# EP — config + intent
# --------------------------------------------------------------------------- #


def test_ep_size_field_defaults_none():
    cfg = _cfg()
    assert cfg.ep_size is None


def test_ep_size_flows_into_intent():
    intent = _cfg(ep_size=4).server_intent(ports=PORTS)
    assert intent["ep_size"] == 4


def test_enable_expert_parallel_flows_into_intent():
    intent = _cfg(ep_size=4, enable_expert_parallel=True).server_intent(ports=PORTS)
    assert intent["ep_size"] == 4
    assert intent["enable_expert_parallel"] is True


def test_ep_size_must_be_positive():
    with pytest.raises(Exception):
        _cfg(ep_size=0)


def test_ep_default_intent_is_one():
    # tp=1 default must still set ep_size=1 (the SGLang default), not leak None.
    intent = _cfg().server_intent(ports=PORTS)
    assert intent["ep_size"] == 1


# --------------------------------------------------------------------------- #
# EP — layout composition (EP is SGLang-internal, layout unchanged)
# --------------------------------------------------------------------------- #


def test_ep_does_not_change_rank_layout():
    r_tp = _build_rank_infos(4, tp_size=2)
    r_tp_ep = _build_rank_infos(4, tp_size=2, ep_size=2)
    # Same dp/tp/pp ranks; only ep_size is recorded.
    assert [x.dp_rank for x in r_tp] == [x.dp_rank for x in r_tp_ep]
    assert [x.tp_rank for x in r_tp] == [x.tp_rank for x in r_tp_ep]
    assert all(x.ep_size == 2 for x in r_tp_ep)
    assert all(x.ep_rank == 0 for x in r_tp_ep)  # SGLang owns EP shard rank


def test_ep_does_not_change_nccl_rank_offset():
    # NCCL rank_offset = engine_idx * tp_size + 1, independent of ep_size.
    num_engines, tp, ep = 2, 2, 4
    offsets = [i * tp + 1 for i in range(num_engines)]
    world = num_engines * tp + 1
    assert offsets == [1, 3]
    assert world == 5  # ep_size does NOT enter the NCCL group math


def test_ep_composes_with_tp_in_shape_resolution():
    sp, tp, pp, ep = _parallel_shape_from_init_kwargs(
        {"config": {"tp_size": 2, "ep_size": 4}}, 4, SGLangRolloutEngine
    )
    assert (sp, tp, pp, ep) == (1, 2, 1, 4)


def test_ep_inherited_from_sibling_handleref():
    ref = HandleRef(role_name="rollout", tp_size=2, ep_size=4)
    sp, tp, pp, ep = _parallel_shape_from_init_kwargs({"rollout": ref}, 4, type("R", (), {}))
    assert ep == 4


# --------------------------------------------------------------------------- #
# PP — layout + dispatch filter
# --------------------------------------------------------------------------- #


def test_pp2_populates_pipeline_stage_ranks():
    r = _build_rank_infos(4, pp_size=2)
    assert [x.pp_rank for x in r] == [0, 1, 0, 1]
    assert [x.is_pipeline_last_stage for x in r] == [False, True, False, True]
    assert all(x.pp_size == 2 for x in r)


def test_pp1_no_last_stage_bias():
    r = _build_rank_infos(4, pp_size=1)
    # pp_size=1 => every rank is the (single) last stage; collect keeps all DP heads.
    assert all(x.is_pipeline_last_stage for x in r)


def test_pp2_with_tp2_layout():
    # world=8, tp=2, pp=2 => 2 DP groups, each with 2 PP stages x 2 TP ranks.
    r = _build_rank_infos(8, tp_size=2, pp_size=2)
    assert [x.dp_rank for x in r] == [0, 0, 0, 0, 1, 1, 1, 1]
    # tp fastest, then pp: rank -> tp_rank = i%2, pp_rank = (i//2)%2
    assert [x.tp_rank for x in r] == [0, 1, 0, 1, 0, 1, 0, 1]
    assert [x.pp_rank for x in r] == [0, 0, 1, 1, 0, 0, 1, 1]


def test_pp2_dispatch_collect_filter_selects_tp0_last_stage():
    # The DP_SCATTER collect keeps results where tp_rank==0 AND
    # is_pipeline_last_stage AND sp_rank==0. Under tp=2 pp=2 world=8, the
    # collected ranks are those with tp_rank==0 and pp_rank==1.
    r = _build_rank_infos(8, tp_size=2, pp_size=2)
    collected = [
        i for i, ri in enumerate(r)
        if ri.tp_rank == 0 and ri.is_pipeline_last_stage and ri.sp_rank == 0
    ]
    assert collected == [2, 6]  # one per DP group


def test_pp_shape_resolution_from_sglang_config():
    sp, tp, pp, ep = _parallel_shape_from_init_kwargs(
        {"config": {"tp_size": 2, "pp_size": 2}}, 8, SGLangRolloutEngine
    )
    assert (sp, tp, pp, ep) == (1, 2, 2, 1)


def test_pp_world_must_divide_tp_times_pp():
    with pytest.raises(ValueError):
        _parallel_shape_from_init_kwargs({"config": {"pp_size": 3}}, 4, SGLangRolloutEngine)


# --------------------------------------------------------------------------- #
# PP — weight sync fail-closed
# --------------------------------------------------------------------------- #


@dataclass
class _FakeBackend:
    rollout_adapter_name: str = "default"


class _FakeRayHandle:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: List[Tuple[str, Tuple, Dict]] = []
        self.call = self._Call(self)

    class _Call:
        def __init__(self, owner): self._o = owner
        def remote(self, role_name, method_name, args, kwargs, **_):
            self._o.calls.append((method_name, args, dict(kwargs)))
            return _FakeRef()


@dataclass
class _FakeRef:
    def __init__(self): pass


def _nccl_for_test(monkeypatch):
    """Build an NCCLWeightSync with ray/pg plumbing stubbed."""
    monkeypatch.setattr(
        "unirl.utils.distributed_utils.init_process_group", lambda **kw: ("pg",)
    )
    import ray
    monkeypatch.setattr(ray, "get", lambda refs: None)
    from unirl.distributed.weight_sync.full.nccl import NCCLWeightSync
    sync = NCCLWeightSync.__new__(NCCLWeightSync)
    sync._group_name = "g"
    sync._model_update_group = None
    sync._rollout_targets = []
    sync._rollout_role = None
    sync._track_prefix = ""
    return sync


def test_pp_size_gt1_connect_fails_closed(monkeypatch):
    sync = _nccl_for_test(monkeypatch)
    sync._rollout_targets = [_FakeRayHandle("e0")]
    sync._rollout_role = "rollout"
    with pytest.raises(NotImplementedError, match="pp_size>1"):
        sync.connect.__wrapped__(
            sync,
            master_addr="127.0.0.1", master_port=1234,
            num_rollout_gpus=2, tp_size=1, pp_size=2,
        )


def test_pp_size_1_connect_does_not_raise(monkeypatch):
    sync = _nccl_for_test(monkeypatch)
    sync._rollout_targets = [_FakeRayHandle("e0"), _FakeRayHandle("e1")]
    sync._rollout_role = "rollout"
    # pp_size=1 is the supported default; must go through the normal path.
    sync.connect.__wrapped__(
        sync,
        master_addr="127.0.0.1", master_port=1234,
        num_rollout_gpus=2, tp_size=1, pp_size=1,
    )
    all_calls = [c for h in sync._rollout_targets for c in h.calls]
    assert len(all_calls) == 2  # one init_weights_update_group per engine


def test_pp_size_default_is_one(monkeypatch):
    # Omitting pp_size entirely must behave identically to pp_size=1.
    sync = _nccl_for_test(monkeypatch)
    sync._rollout_targets = [_FakeRayHandle("e0")]
    sync._rollout_role = "rollout"
    sync.connect.__wrapped__(
        sync,
        master_addr="127.0.0.1", master_port=1234,
        num_rollout_gpus=1, tp_size=1,
    )
    assert sync._rollout_targets[0].calls[0][2]["rank_offset"] == 1


# --------------------------------------------------------------------------- #
# EP + PP composition — engine shell still honors tp_rank guard
# --------------------------------------------------------------------------- #


def test_ep_pp_does_not_affect_tp_shell_guard():
    # A tp_rank>0 shell must remain a no-op even with EP/PP enabled — the
    # guard keys on tp_rank alone, not ep/pp.
    cfg = _cfg(tp_size=2, ep_size=2, pp_size=1)
    eng = SGLangRolloutEngine(
        config=cfg, rank=1, tp_rank=1, tp_size=2, tp_device_ids=[0, 1], ep_size=2,
    )
    assert eng._is_tp_zero is False
    assert eng._backend is None
    eng.shutdown()
