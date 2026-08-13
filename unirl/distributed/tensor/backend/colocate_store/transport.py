"""ColocateStoreTransport — TensorTransport over a worker-local TensorStore."""

from __future__ import annotations

from typing import Any, List

import ray
import torch

from unirl.distributed.tensor.backend.colocate_store.handle import ColocateTensorHandle
from unirl.distributed.tensor.ref import TensorRef, TensorSpan
from unirl.distributed.tensor.worker_local import WorkerLocalTransport


class ColocateStoreTransport(WorkerLocalTransport):
    """TensorStore backend — per-tensor put/get, NCCL transfer, ref-counting."""

    def __init__(self, store: Any) -> None:
        self._store = store

    @property
    def store(self) -> Any:
        return self._store

    def _resolve_handles(self, handles: List[ColocateTensorHandle]) -> List[torch.Tensor]:
        out: List[torch.Tensor] = []
        for h in handles:
            if h.object_ref is not None:
                out.append(ray.get(h.object_ref).detach())
            elif h.source_id != self._store.worker_id:
                raise RuntimeError(
                    f"ColocateStoreTransport: handle from '{h.source_id}' is not local to "
                    f"'{self._store.worker_id}'. localize should have transferred it."
                )
            else:
                out.append(self._store.get(h))
        return out

    def put(self, tensor: torch.Tensor) -> Any:
        return self._store.put(tensor)

    def is_ref(self, value: Any) -> bool:
        return isinstance(value, TensorRef)

    def incref(self, key: Any) -> None:
        self._store.incref(key)

    def decref(self, key: Any) -> None:
        self._store.decref(key)

    def setup_transfer(self, global_rank: int, world_size: int) -> None:
        self._store.setup_global_pg(global_rank, world_size)

    def nccl_send(self, dst_rank: int, spans: List[TensorSpan]) -> None:
        items = [(s.handle.store_key, s.start, s.stop) for s in spans]
        self._store._nccl_send(dst_rank, items)

    def nccl_recv(self, src_rank: int, shapes: List[tuple], dtypes: List[torch.dtype]) -> List[Any]:
        return self._store._nccl_recv(src_rank, shapes, dtypes)


TensorStoreTransport = ColocateStoreTransport

__all__ = ["ColocateStoreTransport", "TensorStoreTransport"]
