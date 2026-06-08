"""ColocateStoreTransport — TensorTransport over a worker-local TensorStore.

The store lives in the Worker process, so put/get are per-tensor dict ops (no IPC, no
batching). Cross-device moves use the store's NCCL path; ref-counting delegates to the
store. WORKER_LOCAL, GPU-resident default backend.

Single-slot per device: colocate runs one Worker per GPU (DevicePool enforces
``workers_per_device == 1`` for this backend). Colocated multi-slot is gpu_store's job —
its shared per-GPU TensorWorker handles same-GPU sharing without per-process IPC. So
colocate never needs CUDA-IPC: a ref is either local to this worker or on another
device (→ NCCL). Multi-device (all slot0) is supported and leak-free.

Only the store-specific methods are overridden. Batched resolve/pack (get_batch /
put_batch), the per-call scope (end_call), and the remote-compute helpers (tensor_op /
get_cpu) inherit the ABC defaults — those build on this class's get/put, so the defaults
are already correct (and local).
"""

from __future__ import annotations

from typing import Any, List

import ray
import torch

from unirl.distributed.tensor.backend.colocate_store.handle import TensorHandle
from unirl.distributed.tensor.transport import TensorMeta, WorkerLocalTransport


class ColocateStoreTransport(WorkerLocalTransport):
    """TensorStore backend — per-tensor put/get, NCCL transfer, ref-counting."""

    def __init__(self, store: Any) -> None:
        self._store = store

    @property
    def store(self) -> Any:
        return self._store

    def _resolve_handle(self, handle: TensorHandle) -> torch.Tensor:
        if handle.object_ref is not None:
            return ray.get(handle.object_ref).detach()
        if handle.source_id != self._store.worker_id:
            raise RuntimeError(
                f"ColocateStoreTransport: handle from '{handle.source_id}' is not local to "
                f"'{self._store.worker_id}'. localize should have transferred it."
            )
        return self._store.get(handle)

    def put(self, tensor: torch.Tensor) -> Any:
        return self._store.put(tensor)

    def get(self, refs: List[Any]) -> torch.Tensor:
        if not refs:
            raise ValueError("ColocateStoreTransport.get: empty refs list")
        parts = [self._resolve_handle(h) for h in refs]
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)

    def is_ref(self, value: Any) -> bool:
        return isinstance(value, TensorMeta)

    # ── lifecycle ──

    def incref(self, key: Any) -> None:
        self._store.incref(key)

    def decref(self, key: Any) -> None:
        self._store.decref(key)

    # locality: colocate is single-slot per device, so a ref is local only if produced
    # by the dst worker itself — exactly WorkerLocalTransport's default _is_local. The
    # shared localize skeleton (cross-device → slot0↔slot0 NCCL) is inherited.

    # ── cross-worker transfer ──

    def setup_transfer(self, global_rank: int, world_size: int) -> None:
        self._store.setup_global_pg(global_rank, world_size)

    def nccl_send(self, dst_rank: int, handles: List[Any]) -> None:
        self._store._nccl_send(dst_rank, handles)

    def nccl_recv(self, src_rank: int, shapes: List[tuple], dtypes: List[torch.dtype]) -> List[Any]:
        return self._store._nccl_recv(src_rank, shapes, dtypes)


# Backwards-compatible alias (older name).
TensorStoreTransport = ColocateStoreTransport

__all__ = ["ColocateStoreTransport", "TensorStoreTransport"]
