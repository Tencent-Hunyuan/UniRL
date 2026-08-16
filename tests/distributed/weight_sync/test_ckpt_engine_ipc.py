"""CPU tests for checkpoint-engine IPC topology and sender sizing."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from unirl.distributed.group.remote import RankInfo
from unirl.distributed.weight_sync.full.ckpt_engine_ipc import CkptEngineIPCWeightSync
from unirl.distributed.weight_sync.transfer.ckpt_engine_transfer import CkptEngineWeightSender


class _FakeBackend:
    rollout_adapter_name = "default"
    model = None


def _make_sync(*, cfg: SimpleNamespace, tp_size: int = 1, rank_info: RankInfo | None = None) -> CkptEngineIPCWeightSync:
    rollout = SimpleNamespace(cfg=cfg, _tp_size=tp_size, _backend=SimpleNamespace(requires_main_thread_ipc_receiver=False))
    sync = CkptEngineIPCWeightSync(backend=_FakeBackend(), rollout=rollout, timeout_s=5)
    sync.rank_info = rank_info
    return sync


def test_timeout_must_be_positive() -> None:
    with pytest.raises(ValueError, match="timeout_s"):
        CkptEngineIPCWeightSync(backend=_FakeBackend(), rollout=SimpleNamespace(), timeout_s=0)


def test_reject_pipeline_parallel() -> None:
    sync = _make_sync(cfg=SimpleNamespace(dp_size=1, engine_kwargs={}), rank_info=RankInfo(pp_size=2, tp_size=1))
    with pytest.raises(NotImplementedError, match="pp_size"):
        sync._validate_topology(1)


def test_reject_server_dp_on_config() -> None:
    sync = _make_sync(cfg=SimpleNamespace(dp_size=2, engine_kwargs={}))
    with pytest.raises(NotImplementedError, match="dp_size"):
        sync._validate_topology(1)


def test_reject_server_dp_in_engine_kwargs() -> None:
    sync = _make_sync(cfg=SimpleNamespace(dp_size=None, engine_kwargs={"dp_size": 2}))
    with pytest.raises(NotImplementedError, match="dp_size"):
        sync._validate_topology(1)


def test_reject_speculative_kwargs() -> None:
    sync = _make_sync(cfg=SimpleNamespace(dp_size=1, engine_kwargs={"speculative_algorithm": "EAGLE"}))
    with pytest.raises(NotImplementedError, match="speculative"):
        sync._validate_topology(1)


def test_reject_rankinfo_tp_mismatch() -> None:
    sync = _make_sync(cfg=SimpleNamespace(dp_size=1, engine_kwargs={}), tp_size=8, rank_info=RankInfo(tp_size=1))
    with pytest.raises(RuntimeError, match="tp_size"):
        sync._validate_topology(8)


def test_accepts_matching_http_layout() -> None:
    sync = _make_sync(cfg=SimpleNamespace(dp_size=1, engine_kwargs={}), tp_size=8, rank_info=RankInfo(tp_size=8, tp_rank=0))
    sync._validate_topology(8)


def test_sender_bucket_bytes() -> None:
    sender = CkptEngineWeightSender({"GPU-0": "ipc:///tmp/unirl-ckpt-engine-test.sock"}, bucket_size_mb=640)
    assert sender.bucket_size == 640 << 20
    assert sender.timeout_ms == 600_000


def test_oversized_tensor_raises_not_assert() -> None:
    sender = CkptEngineWeightSender({"GPU-0": "ipc:///tmp/unirl-ckpt-engine-test.sock"}, bucket_size_mb=1)
    sender.buffer = type("Buf", (), {})()  # skip CUDA allocate
    sender.sockets = [SimpleNamespace(send_pyobj=lambda *_: None, recv=lambda: b"", close=lambda **_: None)]
    sender._can_send = [True]
    sender._handle = object()

    import torch

    too_big = torch.zeros(sender.bucket_size + 1, dtype=torch.uint8)

    def _no_handshake() -> None:
        return None

    sender._handshake = _no_handshake  # type: ignore[method-assign]
    sender._init_sockets = lambda: None  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="too large"):
        sender.send_weights(iter([("embed.weight", too_big)]))
