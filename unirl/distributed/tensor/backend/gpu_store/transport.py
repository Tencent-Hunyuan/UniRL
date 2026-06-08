"""GPUStoreTransport — TensorTransport over a per-GPU TensorWorker actor.

Tensors live in a separate per-GPU ``TensorWorker`` Ray actor; the Worker reaches
it over a handle (``tw``) and moves data via batched CUDA IPC:

  resolve (get_batch):  one ``batch_borrow`` RPC → open zero-copy IPC views.
  pack    (put_batch):  one ``batch_allocate`` RPC → IPC-write → ``batch_write_done``.

Open IPC views are held for the duration of the Worker's ``call()`` and released
in ``end_call()`` (counter 1→0). A WORKER_LOCAL backend: lifecycle / NCCL /
remote-compute delegate to the TensorWorker.
"""

from __future__ import annotations

from typing import Any, Dict, List

import ray
import torch

from unirl.distributed.tensor.backend.gpu_store.handle import TensorHandle
from unirl.distributed.tensor.transport import TensorMeta, WorkerLocalTransport


class GPUStoreTransport(WorkerLocalTransport):
    """TensorWorker backend — batched IPC put/get, NCCL transfer, ref-counting."""

    def __init__(self, worker_id: str, device_id: int, device: str, tw: Any = None) -> None:
        self.worker_id = worker_id
        self.device_id = device_id
        self.device = device
        self._tw = tw
        # Per-call IPC state (cleared in end_call):
        self._open_storages: list = []  # UntypedStorage refs keeping views alive
        self._resolved_map: Dict[str, torch.Tensor] = {}  # store_key → view (dedup)

    def set_tensor_worker(self, tw: Any) -> None:
        """Inject the per-GPU TensorWorker actor handle (called by DevicePool)."""
        self._tw = tw

    def is_ref(self, value: Any) -> bool:
        return isinstance(value, TensorMeta)

    # ── resolve helpers: batched borrow + zero-copy IPC views ──

    def _batch_borrow(self, handles: List[TensorHandle]) -> Dict[str, tuple]:
        """One ``batch_borrow`` RPC for all not-yet-open CUDA store_keys."""
        unique = list(
            dict.fromkeys(
                h.store_key for h in handles if h.object_ref is None and h.store_key not in self._resolved_map
            )
        )
        if not unique:
            return {}
        return dict(zip(unique, ray.get(self._tw.batch_borrow.remote(unique))))

    def _resolve_handle(self, handle: TensorHandle, borrow_map: Dict[str, tuple]) -> torch.Tensor:
        if handle.object_ref is not None:
            return ray.get(handle.object_ref).detach()
        cached = self._resolved_map.get(handle.store_key)
        if cached is not None:
            return cached
        ipc_h, shape, stride = borrow_map[handle.store_key]
        storage = torch.UntypedStorage._new_shared_cuda(*ipc_h)
        view = torch.empty(0, dtype=handle.dtype, device=self.device)
        view.set_(storage, 0, shape, stride)  # offset=0; TW stores contiguous
        self._open_storages.append(storage)
        resolved = view.detach()
        self._resolved_map[handle.store_key] = resolved
        return resolved

    # ── single-tensor primitives (used by transform / dehydrate defaults) ──

    def put(self, tensor: torch.Tensor) -> Any:
        return self.put_batch({"_": tensor})["_"].refs[0]

    def get(self, refs: List[Any]) -> torch.Tensor:
        if not refs:
            raise ValueError("GPUStoreTransport.get: empty refs list")
        borrow_map = self._batch_borrow(refs)
        parts = [self._resolve_handle(h, borrow_map) for h in refs]
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)

    # ── batched resolve / pack (the Worker's path) ──

    def get_batch(self, metas: Dict[str, TensorMeta]) -> Dict[str, torch.Tensor]:
        all_handles = [h for m in metas.values() for h in m.refs]
        borrow_map = self._batch_borrow(all_handles)
        out: Dict[str, torch.Tensor] = {}
        for k, m in metas.items():
            parts = [self._resolve_handle(h, borrow_map) for h in m.refs]
            out[k] = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
        return out

    def put_batch(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, TensorMeta]:
        items = list(tensors.items())

        # Pass 1: one batch_allocate for the unique CUDA tensors (by id).
        cuda_unique: Dict[int, torch.Tensor] = {}
        for _, t in items:
            if isinstance(t, torch.Tensor) and t.is_cuda:
                cuda_unique.setdefault(id(t), t)

        handle_map: Dict[int, TensorHandle] = {}
        if cuda_unique:
            tl = list(cuda_unique.values())
            allocs = ray.get(self._tw.batch_allocate.remote([(tuple(t.shape), t.dtype) for t in tl]))
            views_storages = []
            for t, (_, ipc_h, stride) in zip(tl, allocs):
                storage = torch.UntypedStorage._new_shared_cuda(*ipc_h)
                view = torch.empty(0, dtype=t.dtype, device=self.device)
                view.set_(storage, 0, tuple(t.shape), stride)
                view.copy_(t.contiguous())
                views_storages.append((view, storage))
            torch.cuda.current_stream().synchronize()
            for v, s in views_storages:
                del v, s  # counter 1→0
            ray.get(self._tw.batch_write_done.remote([sk for sk, _, _ in allocs]))
            for t, (sk, _, _) in zip(tl, allocs):
                handle_map[id(t)] = TensorHandle(
                    store_key=sk, source_id=self.worker_id, shape=tuple(t.shape), dtype=t.dtype, device=self.device
                )

        # Pass 2: one TensorMeta per key; duplicate output tensors share a
        # store_key but get distinct handles (independent rebind/GC) + batch_incref.
        emitted: set = set()
        extra_incref: list = []
        result: Dict[str, TensorMeta] = {}
        for k, t in items:
            if isinstance(t, torch.Tensor) and t.is_cuda:
                h = handle_map[id(t)]
                if id(t) in emitted:
                    extra_incref.append(h.store_key)
                    handle = TensorHandle(
                        store_key=h.store_key, source_id=h.source_id, shape=h.shape, dtype=h.dtype, device=h.device
                    )
                else:
                    emitted.add(id(t))
                    handle = h
            elif isinstance(t, torch.Tensor):  # CPU tensor → Ray plasma
                handle = TensorHandle(
                    store_key=None,
                    source_id=self.worker_id,
                    shape=tuple(t.shape),
                    dtype=t.dtype,
                    device=str(t.device),
                    object_ref=ray.put(t.detach()),
                )
            else:
                result[k] = t
                continue
            bs = handle.shape[0] if handle.shape else 1
            result[k] = TensorMeta(
                refs=[handle], sizes=[bs], shape=tuple(handle.shape), dtype=handle.dtype, device=str(handle.device)
            )
        if extra_incref:
            self._tw.batch_incref.remote(extra_incref)
        return result

    def end_call(self) -> None:
        # Close borrowed IPC views: drop storages + view refs so counter 1→0.
        self._open_storages.clear()
        self._resolved_map.clear()

    # ── lifecycle (delegate to TensorWorker) ──

    def incref(self, key: Any) -> None:
        self._tw.incref.remote(key)

    def decref(self, key: Any) -> None:
        self._tw.decref.remote(key)

    # ── locality (gpu: same device is local via the shared per-GPU TensorWorker) ──

    @classmethod
    def _is_local(cls, ref: Any, dst_worker_id: str, dst_device_id: int, pool: Any) -> bool:
        # gpu's per-GPU TensorWorker is shared by all slots → a same-device ref (any
        # slot) is already resolvable on the dst worker; only cross-device needs NCCL.
        # The shared localize skeleton handles the transfer.
        return ref.source_id == dst_worker_id or pool.device_id_of(ref.source_id) == dst_device_id

    # ── cross-worker transfer (delegate to TensorWorker) ──

    def setup_transfer(self, global_rank: int, world_size: int) -> None:
        ray.get(self._tw.setup_global_pg.remote(global_rank, world_size))

    def nccl_send(self, dst_rank: int, handles: List[Any]) -> None:
        ray.get(self._tw._nccl_send.remote(dst_rank, [h.store_key for h in handles]))

    def nccl_recv(self, src_rank: int, shapes: List[tuple], dtypes: List[torch.dtype]) -> List[Any]:
        return ray.get(self._tw._nccl_recv.remote(src_rank, shapes, dtypes))

    # ── remote compute (on-worker kernels on the TensorWorker) ──

    def tensor_op(self, handle: Any, op: str, *op_args) -> Any:
        return ray.get(self._tw.tensor_op.remote(handle, op, *op_args))

    def get_cpu(self, handle: Any) -> torch.Tensor:
        return ray.get(self._tw.get_tensor_cpu.remote(handle))


__all__ = ["GPUStoreTransport"]
