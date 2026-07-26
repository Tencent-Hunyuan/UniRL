from dataclasses import dataclass
from typing import Any

import torch

from unirl.distributed.tensor import worker_local
from unirl.distributed.tensor.backend.colocate_store.transport import ColocateStoreTransport
from unirl.distributed.tensor.backend.gpu_store.transport import GPUStoreTransport
from unirl.distributed.tensor.ref import TensorSpan
from unirl.distributed.tensor.worker_local import WorkerLocalTransport


class _Handle:
    def __init__(self, shape: tuple, dtype: torch.dtype) -> None:
        self.shape = shape
        self.dtype = dtype
        self.bound_worker = None

    def rebind(self, worker: Any) -> None:
        self.bound_worker = worker


@dataclass
class _ObjectRef:
    kind: str
    pair: tuple[int, int]
    value: Any


class _TransportOp:
    def __init__(self, pool: "_Pool", device_id: int) -> None:
        self.pool = pool
        self.device_id = device_id

    def remote(self, op: str, peer_device_id: int, *args: Any) -> _ObjectRef:
        if op == "nccl_recv":
            pair = (peer_device_id, self.device_id)
            shapes, dtypes = args
            handles = [_Handle(shape, dtype) for shape, dtype in zip(shapes, dtypes)]
            self.pool.received[pair] = handles
            self.pool.calls.append((op, pair, shapes, dtypes))
            return _ObjectRef(op, pair, handles)

        assert op == "nccl_send"
        pair = (self.device_id, peer_device_id)
        (spans,) = args
        self.pool.calls.append((op, pair, spans))
        return _ObjectRef(op, pair, None)


class _Worker:
    def __init__(self, pool: "_Pool", device_id: int) -> None:
        self.device_id = device_id
        self.transport_op = _TransportOp(pool, device_id)


class _Pool:
    def __init__(self, num_devices: int) -> None:
        self.calls = []
        self.received = {}
        self.workers = [_Worker(self, device_id) for device_id in range(num_devices)]

    def slot0_worker(self, device_id: int) -> _Worker:
        return self.workers[device_id]


def test_move_orders_pairs_recv_first_and_waits_per_pair(monkeypatch) -> None:
    pool = _Pool(3)
    waits = []

    def fake_get(refs: list[_ObjectRef]) -> list[Any]:
        waits.append(tuple((ref.kind, ref.pair) for ref in refs))
        pool.calls.append(("wait", refs[0].pair))
        return [ref.value for ref in refs]

    monkeypatch.setattr(worker_local.ray, "get", fake_get)

    span_21 = TensorSpan(_Handle((8, 5), torch.float16), 1, 6)
    span_02_first = TensorSpan(_Handle((7, 4), torch.float32), 3, 5)
    span_02_second = TensorSpan(_Handle((9, 4), torch.bfloat16), 2, 6)
    key_21 = (2, 1, "late-pair", span_21.start, span_21.stop)
    key_02_first = (0, 2, "first-in-batch", span_02_first.start, span_02_first.stop)
    key_02_second = (0, 2, "second-in-batch", span_02_second.start, span_02_second.stop)

    moved = WorkerLocalTransport._move(
        pool,
        {
            key_21: span_21,
            key_02_first: span_02_first,
            key_02_second: span_02_second,
        },
    )

    assert [(call[0], call[1]) for call in pool.calls] == [
        ("nccl_recv", (0, 2)),
        ("nccl_send", (0, 2)),
        ("wait", (0, 2)),
        ("nccl_recv", (2, 1)),
        ("nccl_send", (2, 1)),
        ("wait", (2, 1)),
    ]
    assert waits == [
        (("nccl_recv", (0, 2)), ("nccl_send", (0, 2))),
        (("nccl_recv", (2, 1)), ("nccl_send", (2, 1))),
    ]

    recv_02, send_02 = pool.calls[:2]
    assert recv_02[2:] == (
        [span_02_first.shape, span_02_second.shape],
        [span_02_first.dtype, span_02_second.dtype],
    )
    assert send_02[2] == [span_02_first, span_02_second]

    assert set(moved) == {key_21, key_02_first, key_02_second}
    for key, span in moved.items():
        src_device_id, dst_device_id = key[:2]
        received_index = [key_02_first, key_02_second].index(key) if (src_device_id, dst_device_id) == (0, 2) else 0
        received_handle = pool.received[(src_device_id, dst_device_id)][received_index]
        assert span.handle is received_handle
        assert span.start == 0
        assert span.stop == received_handle.shape[0]
        assert received_handle.bound_worker is pool.workers[dst_device_id]


def test_worker_local_backends_inherit_ordered_move() -> None:
    assert ColocateStoreTransport._move.__func__ is WorkerLocalTransport._move.__func__
    assert GPUStoreTransport._move.__func__ is WorkerLocalTransport._move.__func__
