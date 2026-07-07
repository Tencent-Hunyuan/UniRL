"""Tier 0 smoke tests — TP-aware weight sync (NCCL + Tensor + IPC + LoRA), no GPU.

Validates the weight-sync changes without a live NCCL group or SGLang server by
driving the pure logic:

  - ``NCCLWeightSync.connect`` computes ``rank_offset = i*tp_size + 1`` per
    engine and dispatches exactly one ``init_weights_update_group`` per
    tp_rank==0 target (SGLang fans out to its own TP ranks internally).
  - ``TensorWeightSync.sync`` ships ``tp_size`` payload copies on tp_rank==0 and
    pushes nothing on a tp_rank>0 shell (while still draining the all-gather
    generator in lockstep).
  - ``IPCWeightSync.sync`` starts the rollout receiver once and pumps each stage.
  - ``LocalLoraWeightSync.sync`` extracts on every rank but only pushes from
    tp_rank==0, leaving SGLang to fan the adapter out to its TP workers.
  - ``RemoteLoraWeightSync.sync`` extracts on every rank but only Ray-pushes from
    train rank 0, including the copy receiver branch.
  - The full-weight paths are bit-identical to the pre-change path when
    ``tp_size == 1``.

The transports are exercised with lightweight fakes for the Ray handle / rollout
sibling and the FSDP backend, so no GPU / torch.distributed is needed.

Run:  pytest scripts/tests/rollout/test_tp_weight_sync_on_cpu.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from unirl.distributed.group.remote import RankInfo
from unirl.distributed.weight_sync.full.ipc import IPCWeightSync
from unirl.distributed.weight_sync.full.tensor import TensorWeightSync
from unirl.distributed.weight_sync.lora.local import LocalLoraWeightSync
from unirl.distributed.weight_sync.lora.remote import RemoteLoraWeightSync

from ..conftest import FakeBackend, FakeRayHandle, make_nccl_sync

# --------------------------------------------------------------------------- #
# NCCLWeightSync.connect rank_offset + world_size math
# --------------------------------------------------------------------------- #


def test_nccl_connect_tp1_matches_baseline(monkeypatch):
    sync = make_nccl_sync(monkeypatch)
    targets = [FakeRayHandle(f"e{i}") for i in range(4)]
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
    sync = make_nccl_sync(monkeypatch)
    # Two engines (tp_rank==0 workers), each occupying tp_size=2 NCCL ranks.
    targets = [FakeRayHandle("e0"), FakeRayHandle("e1")]
    sync._rollout_targets = targets
    sync._rollout_role = "rollout"

    sync.connect.__wrapped__(sync, master_addr="127.0.0.1", master_port=1234, num_rollout_gpus=4, tp_size=2)

    e0_offset = targets[0].calls[0][2]["rank_offset"]
    e1_offset = targets[1].calls[0][2]["rank_offset"]
    world = targets[0].calls[0][2]["world_size"]
    assert e0_offset == 1  # 0*2 + 1
    assert e1_offset == 3  # 1*2 + 1
    assert world == 5  # 2 engines * 2 tp + 1


# --------------------------------------------------------------------------- #
# NCCLWeightSync.sync rank-zero broadcast + non-rank-zero drain
# --------------------------------------------------------------------------- #


@dataclass
class _FakeBroadcastTensor:
    dtype: Any = "bfloat16"
    shape: tuple = (2, 3)

    @property
    def data(self):
        return self

    def contiguous(self):
        return self


def _install_fake_nccl_buckets(monkeypatch, sync, *, bucket_counter=None):
    fake_buckets = [
        ([("w0", _FakeBroadcastTensor())], False),
        ([("w1", _FakeBroadcastTensor())], True),
    ]

    def _iter_buckets():
        for bucket in fake_buckets:
            if bucket_counter is not None:
                bucket_counter.append(bucket)
            yield bucket

    monkeypatch.setattr(sync, "_iter_buckets", _iter_buckets)


def test_nccl_sync_rank_zero_posts_recvs_and_broadcasts(monkeypatch):
    sync = make_nccl_sync(monkeypatch)
    targets = [FakeRayHandle("e0"), FakeRayHandle("e1")]
    sync._rollout_targets = targets
    sync._rollout_role = "rollout"
    sync._flush_cache = True
    sync._track_prefix = "ar"
    sync._model_update_group = "pg"
    sync.weight_version = 0
    sync.rank_info = RankInfo(rank=0)
    _install_fake_nccl_buckets(monkeypatch, sync)

    broadcasts = []

    import torch.distributed as dist

    monkeypatch.setattr(dist, "broadcast", lambda tensor, src, group: broadcasts.append((tensor, src, group)))

    sync.sync()

    assert len(broadcasts) == 2
    assert all(src == 0 and group == "pg" for _, src, group in broadcasts)
    assert [c[0] for h in targets for c in h.calls] == [
        "update_weights_from_distributed",
        "update_weights_from_distributed",
        "update_weights_from_distributed",
        "update_weights_from_distributed",
    ]
    assert targets[0].calls[0][2]["flush_cache"] is False
    assert targets[0].calls[1][2]["flush_cache"] is True
    assert targets[0].calls[0][2]["track_prefix"] == "ar"
    assert sync.weight_version == 1


def test_nccl_sync_non_rank_zero_drains_but_does_not_push(monkeypatch):
    sync = make_nccl_sync(monkeypatch)
    targets = [FakeRayHandle("e0")]
    sync._rollout_targets = targets
    sync._rollout_role = "rollout"
    sync._flush_cache = True
    sync._track_prefix = ""
    sync._model_update_group = "pg"
    sync.weight_version = 0
    sync.rank_info = RankInfo(rank=1)
    drained = []
    _install_fake_nccl_buckets(monkeypatch, sync, bucket_counter=drained)

    import torch.distributed as dist

    broadcasts = []
    monkeypatch.setattr(dist, "broadcast", lambda tensor, src, group: broadcasts.append((tensor, src, group)))

    sync.sync()

    assert len(drained) == 2
    assert targets[0].calls == []
    assert broadcasts == []
    assert sync.weight_version == 1


# --------------------------------------------------------------------------- #
# TensorWeightSync.sync payload count + tp_rank guard
# --------------------------------------------------------------------------- #


class _FakeRollout:
    """Records update_weights_from_tensor calls; used as the colocate sibling."""

    def __init__(self):
        self.pushes: List[List[str]] = []

    def update_weights_from_tensor(self, *, serialized_named_tensors, load_format, flush_cache, track_prefix):
        self.pushes.append(list(serialized_named_tensors))


def _tensor_sync_for_test(monkeypatch, rank_info: Optional[RankInfo], rollout: _FakeRollout):
    """Construct a TensorWeightSync with the FSDP walk and serializer stubbed."""
    sync = TensorWeightSync.__new__(TensorWeightSync)
    sync._backend = FakeBackend()
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
    def __init__(self, *, named_tensors):
        self._nt = named_tensors

    def get_flattened_tensor(self):
        return _FakeTensor()

    def get_metadata(self):
        return {"names": [n for n, _ in self._nt]}


class _FakeSerializer:
    @staticmethod
    def serialize(payload, output_str=True):
        return "PAYLOAD"


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


# --------------------------------------------------------------------------- #
# LocalLoraWeightSync.sync extract-on-all-ranks + tp_rank guard
# --------------------------------------------------------------------------- #


class _FakeLoraRollout:
    """Records set_lora_from_tensors calls; used as the colocate sibling."""

    def __init__(self):
        self.pushes: List[tuple[str, dict, dict]] = []

    def set_lora_from_tensors(self, adapter_name, lora_tensors, *, peft_config=None):
        self.pushes.append((adapter_name, dict(lora_tensors), dict(peft_config or {})))


class _ExtractCounter:
    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return {"model.layers.0.q_proj.lora_A.weight": "A"}, {"r": 16, "lora_alpha": 16}


def _lora_sync_for_test(monkeypatch, rank_info: Optional[RankInfo], rollout: _FakeLoraRollout):
    """Construct a LocalLoraWeightSync with the FSDP extraction stubbed."""
    sync = LocalLoraWeightSync.__new__(LocalLoraWeightSync)
    sync._backend = FakeBackend()
    sync._param_prefix = ""
    sync._adapter_name = "default"
    sync._verify = False
    sync._track_prefix = ""
    sync._rollout = rollout
    sync.rank_info = rank_info

    extract = _ExtractCounter()
    monkeypatch.setattr(sync, "_extract", extract)
    return sync, extract


def test_lora_sync_tp_zero_pushes_after_extract(monkeypatch):
    ri = RankInfo(rank=0, tp_rank=0, tp_size=2)
    rollout = _FakeLoraRollout()
    sync, extract = _lora_sync_for_test(monkeypatch, ri, rollout)

    sync.sync()

    assert extract.calls == 1
    assert rollout.pushes == [
        (
            "default",
            {"model.layers.0.q_proj.lora_A.weight": "A"},
            {"r": 16, "lora_alpha": 16},
        )
    ]


def test_lora_sync_non_tp_zero_extracts_but_pushes_nothing(monkeypatch):
    ri = RankInfo(rank=1, tp_rank=1, tp_size=2)
    rollout = _FakeLoraRollout()
    sync, extract = _lora_sync_for_test(monkeypatch, ri, rollout)

    sync.sync()

    # tp_rank>0 still runs the extraction (FSDP all-gather lockstep) but never
    # calls the rollout shell. The tp_rank==0 SGLang server handles TP fan-out.
    assert extract.calls == 1
    assert rollout.pushes == []


# --------------------------------------------------------------------------- #
# RemoteLoraWeightSync rank-zero Ray push + non-rank-zero extraction only
# --------------------------------------------------------------------------- #


def _remote_lora_sync_for_test(monkeypatch, rank_info: RankInfo, *, copy: bool = False):
    sync = RemoteLoraWeightSync.__new__(RemoteLoraWeightSync)
    sync._backend = FakeBackend()
    sync._param_prefix = ""
    sync._adapter_name = "default"
    sync._verify = False
    sync._track_prefix = ""
    sync._copy = bool(copy)
    sync._targets = []
    sync._cached = None
    sync.rank_info = rank_info

    extract = _ExtractCounter()
    monkeypatch.setattr(sync, "_extract", extract)

    import ray

    monkeypatch.setattr(ray, "get", lambda refs: None)
    return sync, extract


def test_remote_lora_rank_zero_pushes_to_each_target(monkeypatch):
    sync, extract = _remote_lora_sync_for_test(monkeypatch, RankInfo(rank=0))
    workers = [FakeRayHandle("w0"), FakeRayHandle("w1")]
    sync._targets = [("rollout", workers)]

    sync.sync()

    assert extract.calls == 1
    for worker in workers:
        assert worker.calls == [
            (
                "set_lora_from_tensors",
                ("default", {"model.layers.0.q_proj.lora_A.weight": "A"}),
                {"peft_config": {"r": 16, "lora_alpha": 16}},
            )
        ]
    assert sync._cached is None


def test_remote_lora_copy_mode_uses_copy_receiver(monkeypatch):
    sync, _ = _remote_lora_sync_for_test(monkeypatch, RankInfo(rank=0), copy=True)
    worker = FakeRayHandle("w0")
    sync._targets = [("rollout", [worker])]

    sync.sync()

    assert worker.calls[0][0] == "set_lora_from_tensors_copy"


def test_remote_lora_non_rank_zero_extracts_but_does_not_push(monkeypatch):
    sync, extract = _remote_lora_sync_for_test(monkeypatch, RankInfo(rank=1))
    worker = FakeRayHandle("w0")
    sync._targets = [("rollout", [worker])]

    sync.sync()

    assert extract.calls == 1
    assert worker.calls == []
    assert sync._cached is None


# --------------------------------------------------------------------------- #
# IPCWeightSync receiver thread + per-stage sender pump
# --------------------------------------------------------------------------- #


class _FakeIPCRollout:
    def __init__(self, stages=None):
        self.stages = stages or {0: 1}
        self.receiver_calls = []

    def tp_per_stage(self):
        return self.stages

    def update_weights_from_ipc(self, **kwargs):
        self.receiver_calls.append(dict(kwargs))


def _ipc_sync_for_test(monkeypatch, rollout: _FakeIPCRollout):
    sync = IPCWeightSync.__new__(IPCWeightSync)
    sync._backend = FakeBackend()
    sync._bucket_bytes = 16 << 20
    sync._flush_cache = True
    sync._lora_merged = False
    sync._adapter_name = "default"
    sync._name_remap = []
    sync._track_prefix = "ar"
    sync._wire_dtype = None
    sync.weight_version = 0
    sync._rollout = rollout
    sync._use_shm = True
    sync.rank_info = RankInfo(rank=3)

    full_walks = []

    def _iter_full_tensors():
        full_walks.append("walk")
        yield "w0", _FakeBroadcastTensor()

    monkeypatch.setattr(sync, "_iter_full_tensors", _iter_full_tensors)
    return sync, full_walks


def test_ipc_sync_spawns_receiver_and_sends_each_stage(monkeypatch):
    rollout = _FakeIPCRollout(stages={0: 1, 2: 1})
    sync, full_walks = _ipc_sync_for_test(monkeypatch, rollout)
    sender_inits = []
    sent_weights = []

    class _FakeSender:
        def __init__(self, *, zmq_handle, bucket_size_mb, use_shm):
            sender_inits.append((zmq_handle, bucket_size_mb, use_shm))

        async def async_send_weights(self, weights):
            sent_weights.append(list(weights))

    monkeypatch.setattr("unirl.distributed.weight_sync.transfer.bucketed_transfer.BucketedWeightSender", _FakeSender)
    monkeypatch.setattr(
        "unirl.distributed.weight_sync.transfer.ipc_dispatch.zmq_handle",
        lambda *, replica_rank, stage_id, local_rank: f"ipc://r{replica_rank}-s{stage_id}-l{local_rank}",
    )

    sync.sync()

    assert rollout.receiver_calls == [
        {
            "peft_config": None,
            "base_sync_done": False,
            "use_shm": True,
            "replica_rank": 3,
            "track_prefix": "ar",
        }
    ]
    assert sender_inits == [
        ("ipc://r3-s0-l0", 16, True),
        ("ipc://r3-s2-l0", 16, True),
    ]
    assert len(sent_weights) == 2
    assert full_walks == ["walk", "walk"]
    assert sync.weight_version == 1
