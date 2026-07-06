"""Tier 0 smoke tests — TP-aware weight sync (NCCL + Tensor), no GPU.

Validates the weight-sync changes without a live NCCL group or SGLang server by
driving the pure logic:

  - ``NCCLWeightSync.connect`` computes ``rank_offset = i*tp_size + 1`` per
    engine and dispatches exactly one ``init_weights_update_group`` per
    tp_rank==0 target (SGLang fans out to its own TP ranks internally).
  - ``TensorWeightSync.sync`` ships ``tp_size`` payload copies on tp_rank==0 and
    pushes nothing on a tp_rank>0 shell (while still draining the all-gather
    generator in lockstep).
  - Both are bit-identical to the pre-change path when ``tp_size == 1``.

The transports are exercised with lightweight fakes for the Ray handle / rollout
sibling and the FSDP backend, so no GPU / torch.distributed is needed.

Run:  pytest scripts/tests/test_rollout_tp_weight_sync.py
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pytest

from unirl.distributed.group.remote import RankInfo
from unirl.distributed.weight_sync.full.nccl import NCCLWeightSync
from unirl.distributed.weight_sync.full.tensor import TensorWeightSync


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


@dataclass
class _FakeBackend:
    """FSDP backend stand-in: ``rollout_adapter_name`` is all the base reads."""
    rollout_adapter_name: str = "default"


class _FakeRayHandle:
    """Stand-in for a Ray Worker actor handle recording dispatched RPCs."""
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
    """A picklable stand-in for a Ray ObjectRef: ``ray.get`` returns None."""
    def __init__(self): pass


# --------------------------------------------------------------------------- #
# NCCLWeightSync.connect rank_offset + world_size math
# --------------------------------------------------------------------------- #


def _nccl_for_test(monkeypatch):
    """Build an NCCLWeightSync whose rank-0-only / ray / pg plumbing is stubbed.

    We bypass the real ``init_process_group`` and ``ray.get`` (the fake handles
    return non-Ray refs) and seed ``_rollout_targets`` / ``_rollout_role``
    directly so the ``connect`` body runs without a driver dispatch decorator.
    """
    monkeypatch.setattr(
        "unirl.utils.distributed_utils.init_process_group", lambda **kw: ("pg",)
    )
    import ray
    monkeypatch.setattr(ray, "get", lambda refs: None)
    sync = NCCLWeightSync.__new__(NCCLWeightSync)
    sync._group_name = "g"
    sync._model_update_group = None
    sync._rollout_targets = []
    sync._rollout_role = None
    sync._track_prefix = ""
    return sync


def test_nccl_connect_tp1_matches_baseline(monkeypatch):
    sync = _nccl_for_test(monkeypatch)
    targets = [_FakeRayHandle(f"e{i}") for i in range(4)]
    sync._rollout_targets = targets
    sync._rollout_role = "rollout"

    sync.connect.__wrapped__(sync, master_addr="127.0.0.1", master_port=1234, num_rollout_gpus=4, tp_size=1)

    all_calls = [c for h in targets for c in h.calls]
    offsets = [c[2]["rank_offset"] for c in all_calls]
    worlds = [c[2]["world_size"] for c in all_calls]
    assert offsets == [1, 2, 3, 4]  # i + 1
    assert worlds == [5, 5, 5, 5]
    # Each target gets exactly one init_weights_update_group call.
    assert all(len(h.calls) == 1 for h in targets)


def test_nccl_connect_tp2_scales_rank_offset(monkeypatch):
    sync = _nccl_for_test(monkeypatch)
    # Two engines (tp_rank==0 workers), each occupying tp_size=2 NCCL ranks.
    targets = [_FakeRayHandle("e0"), _FakeRayHandle("e1")]
    sync._rollout_targets = targets
    sync._rollout_role = "rollout"

    sync.connect.__wrapped__(
        sync, master_addr="127.0.0.1", master_port=1234, num_rollout_gpus=4, tp_size=2
    )

    e0_offset = targets[0].calls[0][2]["rank_offset"]
    e1_offset = targets[1].calls[0][2]["rank_offset"]
    world = targets[0].calls[0][2]["world_size"]
    assert e0_offset == 1  # 0*2 + 1
    assert e1_offset == 3  # 1*2 + 1
    assert world == 5  # 2 engines * 2 tp + 1


# --------------------------------------------------------------------------- #
# TensorWeightSync.sync payload count + tp_rank guard
# --------------------------------------------------------------------------- #


class _FakeRollout:
    """Records update_weights_from_tensor calls; used as the colocate sibling."""
    def __init__(self): self.pushes: List[List[str]] = []
    def update_weights_from_tensor(self, *, serialized_named_tensors, load_format, flush_cache, track_prefix):
        self.pushes.append(list(serialized_named_tensors))


def _tensor_sync_for_test(monkeypatch, rank_info: Optional[RankInfo], rollout: _FakeRollout):
    """Construct a TensorWeightSync with the FSDP walk and serializer stubbed."""
    sync = TensorWeightSync.__new__(TensorWeightSync)
    sync._backend = _FakeBackend()
    sync._bucket_bytes = 1 << 30
    sync._flush_cache = True
    sync._lora_merged = False
    sync._adapter_name = "default"
    sync._name_remap = []
    sync._track_prefix = ""
    sync._wire_dtype = None
    sync.weight_version = 0
    sync._rollout = rollout
    sync.rank_info = rank_info

    # Stub the FSDP all-gather walk with two tiny fake buckets. Each yield is
    # ``(bucket, is_last)`` where ``bucket`` is a list of ``(name, tensor)``.
    fake_buckets = [
        ([("w0", _FakeTensor())], False),
        ([("w1", _FakeTensor())], True),
    ]

    def _iter_buckets():
        for b in fake_buckets:
            yield b
    monkeypatch.setattr(sync, "_iter_buckets", _iter_buckets)

    # Stub the SGLang serializer path: one payload per dtype bucket. The sync
    # body imports these locally (from sgl_compat when the rollout sibling is
    # not SGLang), so patch the source module the local import reads from.
    from unirl.distributed.weight_sync.transfer import sgl_compat
    monkeypatch.setattr(sgl_compat, "FlattenedTensorBucket", _FakeFlatBucket)
    monkeypatch.setattr(sgl_compat, "MultiprocessingSerializer", _FakeSerializer)
    monkeypatch.setattr(sgl_compat, "monkey_patch_torch_reductions", lambda: None)
    return sync


@dataclass
class _FakeTensor:
    dtype: Any = "bfloat16"  # hashable dtype key for by_dtype grouping


class _FakeFlatBucket:
    def __init__(self, *, named_tensors): self._nt = named_tensors
    def get_flattened_tensor(self): return _FakeTensor()
    def get_metadata(self): return {"names": [n for n, _ in self._nt]}


class _FakeSerializer:
    @staticmethod
    def serialize(payload, output_str=True): return "PAYLOAD"


def test_tensor_sync_tp1_ships_single_payload(monkeypatch):
    ri = RankInfo(tp_rank=0, tp_size=1)
    rollout = _FakeRollout()
    sync = _tensor_sync_for_test(monkeypatch, ri, rollout)

    sync.sync()

    # Two buckets, one payload each.
    assert len(rollout.pushes) == 2
    assert all(len(p) == 1 for p in rollout.pushes)


def test_tensor_sync_tp2_replicates_payload(monkeypatch):
    ri = RankInfo(tp_rank=0, tp_size=2)
    rollout = _FakeRollout()
    sync = _tensor_sync_for_test(monkeypatch, ri, rollout)

    sync.sync()

    assert len(rollout.pushes) == 2
    assert all(len(p) == 2 for p in rollout.pushes), rollout.pushes


def test_tensor_sync_non_tp_zero_pushes_nothing(monkeypatch):
    ri = RankInfo(tp_rank=1, tp_size=2)
    rollout = _FakeRollout()
    sync = _tensor_sync_for_test(monkeypatch, ri, rollout)

    sync.sync()

    # tp_rank>0 still drains the generator (all-gather lockstep) but never pushes.
    assert rollout.pushes == []
    # weight_version still advances (the train-mesh collective completed).
    assert sync.weight_version == 1
