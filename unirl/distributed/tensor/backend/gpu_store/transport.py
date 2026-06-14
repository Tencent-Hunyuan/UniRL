"""GPUStoreTransport — TensorTransport over a per-GPU TensorWorker actor.

Tensors live in a separate per-GPU ``TensorWorker`` Ray actor; the Worker reaches
it over a handle (``tw``) and moves data via batched CUDA IPC:

  resolve (_resolve_handles):  one ``batch_borrow`` RPC → open zero-copy IPC views.
  pack    (put_batch):         one ``batch_allocate`` RPC → IPC-write → ``batch_write_done``.

A borrowed IPC view is kept alive by the resolved tensor that aliases it (PyTorch
storage refcount): the mapping closes itself once the last slice off it is dropped,
so there is no per-call release step. A WORKER_LOCAL backend: lifecycle / NCCL /
remote-compute delegate to the TensorWorker.
"""

from __future__ import annotations

from typing import Any, Dict, List

import ray
import torch

from unirl.distributed.tensor.backend.gpu_store.handle import GPUTensorHandle
from unirl.distributed.tensor.ref import TensorRef, TensorSpan
from unirl.distributed.tensor.worker_local import WorkerLocalTransport


class GPUStoreTransport(WorkerLocalTransport):
    """TensorWorker backend — batched IPC put/get, NCCL transfer, ref-counting."""

    def __init__(self, worker_id: str, device_id: int, device: str, tw: Any = None) -> None:
        self.worker_id = worker_id
        self.device_id = device_id
        self.device = device
        self._tw = tw

    def set_tensor_worker(self, tw: Any) -> None:
        """Inject the per-GPU TensorWorker actor handle (called by DevicePool)."""
        self._tw = tw

    def is_ref(self, value: Any) -> bool:
        return isinstance(value, TensorRef)

    # ── resolve helpers: batched borrow + zero-copy IPC views ──

    def _batch_borrow(self, handles: List[GPUTensorHandle]) -> Dict[str, tuple]:
        """One ``batch_borrow`` RPC for all not-yet-open CUDA store_keys.

        Borrowing is block-level: callers pass span handles, so N spans of one
        block dedup (``dict.fromkeys``) to a single IPC open.
        """
        unique = list(dict.fromkeys(h.store_key for h in handles if h.object_ref is None))
        if not unique:
            return {}
        return dict(zip(unique, ray.get(self._tw.batch_borrow.remote(unique))))

    def _resolve_handles(self, handles: List[GPUTensorHandle]) -> List[torch.Tensor]:
        # One batched borrow for the CUDA blocks, then resolve each handle to its
        # full base. The base's get/get_batch slice the spans' rows (a zero-copy
        # view of the open IPC mapping). ``resolved`` dedups repeat store_keys
        # within this one resolve; it is dropped when the resolve returns.
        borrow_map = self._batch_borrow(handles)
        resolved: Dict[str, torch.Tensor] = {}
        return [self._resolve_one(h, borrow_map, resolved) for h in handles]

    def _resolve_one(
        self, h: GPUTensorHandle, borrow_map: Dict[str, tuple], resolved: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        # Resolve one handle to its full base, deduped per store_key within this
        # resolve. The base owns the open IPC mapping via storage refcount, so the
        # view stays mapped exactly as long as a slice off it is alive — the mapping
        # closes itself once the caller drops the resolved tensor (no release step).
        if h.object_ref is not None:
            return ray.get(h.object_ref).detach()
        base = resolved.get(h.store_key)
        if base is not None:
            return base
        ipc_h, shape, stride = borrow_map[h.store_key]
        storage = torch.UntypedStorage._new_shared_cuda(*ipc_h)
        view = torch.empty(0, dtype=h.dtype, device=self.device)
        view.set_(storage, 0, shape, stride)  # offset=0; TW stores contiguous; view holds storage
        base = view.detach()
        resolved[h.store_key] = base
        return base

    # ── single-tensor primitives (used by transform / dehydrate defaults) ──

    def put(self, tensor: torch.Tensor) -> Any:
        return self.put_batch({"_": tensor})["_"].spans[0].handle

    # ── batched pack (the Worker's path); resolve + get/get_batch come from the base ──

    def put_batch(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, TensorRef]:
        items = list(tensors.items())

        # Pass 1: one batch_allocate for the unique CUDA tensors (by id).
        cuda_unique: Dict[int, torch.Tensor] = {}
        for _, t in items:
            if isinstance(t, torch.Tensor) and t.is_cuda:
                cuda_unique.setdefault(id(t), t)

        handle_map: Dict[int, GPUTensorHandle] = {}
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
                handle_map[id(t)] = GPUTensorHandle(
                    store_key=sk, source_id=self.worker_id, shape=tuple(t.shape), dtype=t.dtype, device=self.device
                )

        # Pass 2: one TensorRef per key; duplicate output tensors share a
        # store_key but get distinct handles (independent rebind/GC) + batch_incref.
        emitted: set = set()
        extra_incref: list = []
        result: Dict[str, TensorRef] = {}
        for k, t in items:
            if isinstance(t, torch.Tensor) and t.is_cuda:
                h = handle_map[id(t)]
                if id(t) in emitted:
                    extra_incref.append(h.store_key)
                    handle = GPUTensorHandle(
                        store_key=h.store_key, source_id=h.source_id, shape=h.shape, dtype=h.dtype, device=h.device
                    )
                else:
                    emitted.add(id(t))
                    handle = h
            elif isinstance(t, torch.Tensor):  # CPU tensor → Ray plasma
                handle = GPUTensorHandle(
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
            result[k] = TensorRef(
                spans=[TensorSpan(handle, 0, bs)],
                shape=tuple(handle.shape),
                dtype=handle.dtype,
                device=str(handle.device),
            )
        if extra_incref:
            self._tw.batch_incref.remote(extra_incref)
        return result

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

    def nccl_send(self, dst_rank: int, spans: List[TensorSpan]) -> None:
        # Each ref is a span → send ONLY its [start:stop) rows (exact-row routing).
        items = [(s.handle.store_key, s.start, s.stop) for s in spans]
        ray.get(self._tw._nccl_send.remote(dst_rank, items))

    def nccl_recv(self, src_rank: int, shapes: List[tuple], dtypes: List[torch.dtype]) -> List[Any]:
        return ray.get(self._tw._nccl_recv.remote(src_rank, shapes, dtypes))

    # ── remote compute (on-worker kernels on the TensorWorker) ──

    def tensor_op(self, handle: Any, op: str, *op_args) -> Any:
        return ray.get(self._tw.tensor_op.remote(handle, op, *op_args))

    def get_cpu(self, handle: Any) -> torch.Tensor:
        return ray.get(self._tw.get_tensor_cpu.remote(handle))


__all__ = ["GPUStoreTransport"]
