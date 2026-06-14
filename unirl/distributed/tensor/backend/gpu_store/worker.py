"""TensorWorker — per-GPU tensor storage, IPC, and NCCL actor.

Hidden Ray actor: one per GPU, created by DevicePool, never exposed to user code.
All role Workers on the same GPU share this single TensorWorker.

Responsibilities:
  - Tensor storage with reference counting (_store, _ref_counts)
  - IPC handle management (_share_cuda_() only called here, never in Worker)
  - NCCL cross-GPU transfers (global ProcessGroup)
  - tensor_op / get_tensor_cpu for GPUTensorHandle remote operations

Put path (Worker → TW):
  1. Worker calls batch_allocate([(shape, dtype), ...]) → [(store_key, ipc_h, stride), ...]
  2. Worker IPC-writes tensors into TW buffers via the returned ipc handles
  3. Worker calls batch_write_done([store_key, ...]) → buf moves _pending → _store

Borrow path (TW → Worker):
  1. Worker calls batch_borrow([store_key, ...]) → [(ipc_h, shape, stride), ...]
  2. Worker opens IPC views (zero-copy read), offset always 0 (TW stores contiguous)
  3. IPC views release by refcount when the resolved tensors aliasing them drop

GC:
  controller weakref.finalize → tw.decref.remote(store_key)
  decref: ref_count → 0 → del buf → ipc_collect() cleans Limbo (counter=0 ≤ 0)
"""

from __future__ import annotations

import os
import threading
from datetime import timedelta
from typing import Dict, List, Tuple

import ray
import torch
import torch.distributed as dist
from torch import Tensor

from unirl.distributed.tensor.backend.gpu_store.handle import GPUTensorHandle


class TensorWorker:
    """Per-GPU tensor storage, IPC, and NCCL Ray actor.

    Hidden: created by DevicePool, device_id maps to global GPU index.
    Workers on the same GPU hold a Ray handle to this actor.

    _share_cuda_() is called once per batch_allocate (for Worker to write into)
    and once per batch_borrow (for Worker to read). Each call produces an
    independent counter slot. ipc_collect() in decref cleans all counter=0 Limbo
    entries — no _release_ipc_counter_cuda needed.

    All allocations use expandable_segments:False so _share_cuda_() always
    produces a portable cudaMalloc-backed IPC handle.
    """

    def __init__(self, device_id: int):
        self.device_id = device_id
        # Handles produced by this TW are owned by the slot0 worker on this GPU.
        # source_id is a worker_id string so controller-side locality (handle._unwrap,
        # pool.device_id_of) resolves it; all slots on this GPU share this TW.
        self.source_id = f"dw{device_id}"
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            self.device = "cuda:0"
            torch.cuda.memory._set_allocator_settings("expandable_segments:False")
        else:
            self.device = "cpu"

        self._store: Dict[str, Tensor] = {}
        self._pending: Dict[str, Tensor] = {}  # allocated but not yet written
        self._ref_counts: Dict[str, int] = {}
        self._lock = threading.Lock()
        self._counter = 0
        self._global_pg = None

        # ipc_collect is deferred: called only when cumulative freed count OR bytes
        # hits a threshold, amortizing the ~620 µs cost across many decref→0 events.
        #
        # In the normal flow (Worker closes IPC view BEFORE controller GC fires decref),
        # del buf + empty_cache() already returns memory to CUDA without needing
        # ipc_collect(). The thresholds here are a safety net for edge cases where
        # consumers close late (e.g., slow GC, explicit copy.copy handles).
        #
        # Tunable via env vars forwarded by DevicePool:
        #   MMRL_IPC_COLLECT_COUNT  (default 128)
        #   MMRL_IPC_COLLECT_BYTES  (default 1 GB)
        self._limbo_count: int = 0
        self._limbo_bytes: int = 0
        self._ipc_collect_count: int = int(os.environ.get("MMRL_IPC_COLLECT_COUNT", "128"))
        self._ipc_collect_bytes: int = int(os.environ.get("MMRL_IPC_COLLECT_BYTES", str(1 << 30)))

    # ── Allocation: put path ──

    def batch_allocate(self, requests: List[Tuple[tuple, torch.dtype]]) -> List[Tuple[str, tuple, tuple]]:
        """Batch-allocate buffers, return [(store_key, ipc_h, stride), ...].

        stride: contiguous stride of the buf — Worker uses it directly, no
        computation needed. offset is always 0 (torch.empty, contiguous).
        Bufs go into _pending; visible for borrow only after batch_write_done.

        _share_cuda_() called inside lock so buf hold and IPC handle creation
        are atomic (TW is Ray single-threaded, lock is a safety net).
        """
        results = []
        with self._lock:
            for shape, dtype in requests:
                buf = torch.empty(shape, dtype=dtype, device=self.device)
                ipc = buf.untyped_storage()._share_cuda_()
                key = f"tw{self.device_id}_{self._counter}"
                self._counter += 1
                self._pending[key] = buf
                results.append((key, ipc, buf.stride()))
        return results

    def batch_write_done(self, store_keys: List[str]) -> None:
        """Move bufs from _pending into _store after Worker finishes writing.

        Must be called synchronously (ray.get) before returning GPUTensorHandle to
        controller — otherwise a concurrent borrow would KeyError on the key.
        """
        with self._lock:
            for key in store_keys:
                buf = self._pending.pop(key)
                self._store[key] = buf
                self._ref_counts[key] = 1

    # ── Borrow: read path ──

    def batch_borrow(self, store_keys: List[str]) -> List[Tuple[tuple, tuple, tuple]]:
        """Batch-create IPC handles for reading, return [(ipc_h, shape, stride), ...].

        shape/stride come from _store buf — Worker uses them directly.
        offset is always 0. Each key gets an independent counter slot.
        Dedup is the caller's responsibility (same key must appear only once).

        _share_cuda_() inside lock: atomic with respect to concurrent decref
        (TW is single-threaded but lock documents the invariant).
        """
        with self._lock:
            return [
                (self._store[k].untyped_storage()._share_cuda_(), tuple(self._store[k].shape), self._store[k].stride())
                for k in store_keys
            ]

    # ── Reference counting ──

    def incref(self, key: str) -> None:
        """Increment reference count. Called by GPUTensorHandle __copy__ on controller."""
        with self._lock:
            if key not in self._ref_counts:
                raise KeyError(f"TensorWorker: cannot incref unknown key '{key}'")
            self._ref_counts[key] += 1

    def batch_incref(self, keys: List[str]) -> None:
        """Batch-increment reference counts for multiple keys (1 RPC instead of N).

        Used by _pack_outputs when the same tensor object appears multiple times
        in the output tree — collects all extra-occurrence keys and fires one RPC.
        """
        with self._lock:
            for key in keys:
                if key not in self._ref_counts:
                    raise KeyError(f"TensorWorker: cannot incref unknown key '{key}'")
                self._ref_counts[key] += 1

    def decref(self, key: str) -> None:
        """Decrement reference count. If zero, release the storage.

        Called by GPUTensorHandle._release (GC finalizer on controller side).

        ipc_collect() is triggered lazily: only after _ipc_collect_count tensors
        freed OR _ipc_collect_bytes of cumulative VRAM freed, whichever comes first.
        In the normal flow the consumer (Worker) closes its IPC view before the
        controller GC fires decref, so del buf alone is sufficient and ipc_collect
        is a safety net for late-closing consumers.
        buf.nbytes is recorded before del buf — pure Python, no synchronize needed.
        """
        do_collect = False
        with self._lock:
            if key not in self._ref_counts:
                raise KeyError(f"TensorWorker: cannot decref unknown key '{key}'")
            self._ref_counts[key] -= 1
            if self._ref_counts[key] <= 0:
                buf = self._store.pop(key)
                del self._ref_counts[key]
                self._limbo_count += 1
                self._limbo_bytes += buf.nbytes  # accumulate before del
                del buf  # IPC counter goes 1→0, storage enters Limbo
                if self._limbo_count >= self._ipc_collect_count or self._limbo_bytes >= self._ipc_collect_bytes:
                    do_collect = True
        if do_collect:
            torch.cuda.ipc_collect()
            with self._lock:
                self._limbo_count = 0
                self._limbo_bytes = 0

    def ref_count(self, key: str) -> int:
        with self._lock:
            if key not in self._ref_counts:
                raise KeyError(f"TensorWorker: unknown key '{key}'")
            return self._ref_counts[key]

    # ── Tensor operations (called by GPUTensorHandle.remote_op / TensorRef.materialize) ──

    def tensor_op(self, handle: GPUTensorHandle, op: str, *op_args) -> GPUTensorHandle:
        """Execute a tensor operation directly in _store (no IPC needed).

        Returns a new unbound GPUTensorHandle (caller must rebind).
        """
        if handle.object_ref is not None:
            t = ray.get(handle.object_ref)
        else:
            with self._lock:
                t = self._store[handle.store_key]

        if op == "getitem":
            result = t[op_args[0]]
        elif op == "reshape":
            result = t.reshape(op_args[0])
        elif op == "permute":
            result = t.permute(op_args[0])
        else:
            raise ValueError(f"Unknown tensor_op: '{op}'")

        result = result.contiguous()
        with self._lock:
            key = f"tw{self.device_id}_{self._counter}"
            self._counter += 1
            self._store[key] = result
            self._ref_counts[key] = 1
        return GPUTensorHandle(
            store_key=key, source_id=self.source_id, shape=tuple(result.shape), dtype=result.dtype, device=self.device
        )

    def get_tensor_cpu(self, handle: GPUTensorHandle) -> Tensor:
        """Return tensor as CPU tensor (for TensorRef.materialize())."""
        if handle.object_ref is not None:
            return ray.get(handle.object_ref)
        with self._lock:
            t = self._store[handle.store_key]
        return t.cpu()

    def get_store_size(self) -> int:
        with self._lock:
            return len(self._store)

    def memory_allocated(self) -> int:
        """Return torch.cuda.memory_allocated() from inside the TW process."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            return int(torch.cuda.memory_allocated())
        return 0

    def empty_cache(self) -> None:
        """Clean up remaining IPC Limbo entries and release PyTorch allocator cache.

        Handles any Limbo entries that haven't yet hit the decref thresholds,
        then returns the allocator's cached blocks to CUDA.
        """
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            with self._lock:
                self._limbo_count = 0
                self._limbo_bytes = 0

    # ── Cross-worker NCCL transfer ──

    def setup_global_pg(self, global_rank: int, global_world_size: int) -> None:
        """Initialize the global ProcessGroup for cross-GPU NCCL transfers.

        Uses ProcessGroupNCCL via TCPStore. MASTER_ADDR and MASTER_PORT must be
        set in the environment (injected by DevicePool via runtime_env).
        """
        store = dist.TCPStore(
            host_name=os.environ["MASTER_ADDR"],
            port=int(os.environ["MASTER_PORT"]),
            world_size=global_world_size,
            is_master=(global_rank == 0),
            timeout=timedelta(seconds=30),
        )
        self._global_pg = dist.ProcessGroupNCCL(store, global_rank, global_world_size)

    def _nccl_send(self, dst_rank: int, items: List) -> None:
        """Send stored tensors (or row ranges of them) to dst_rank via NCCL.

        Each item is ``(store_key, start, end)`` — a ``TensorSpan`` routing copy
        ships only its ``[start:end)`` rows; ``(key, None, None)`` sends the whole
        block. Bare ``str`` items are accepted for backward compatibility.
        """
        assert self._global_pg is not None, "Global PG not initialized."
        for item in items:
            key, start, end = (item, None, None) if isinstance(item, str) else item
            with self._lock:
                tensor = self._store[key]
            if start is not None:
                tensor = tensor[start:end]
            self._global_pg.send([tensor.contiguous()], dst_rank, 0).wait()

    def _nccl_recv(self, src_rank: int, shapes: List[tuple], dtypes: List[torch.dtype]) -> List[GPUTensorHandle]:
        """Receive tensors from src_rank via NCCL, store in _store.

        Returns unbound TensorHandles (caller must rebind to this TW).
        """
        assert self._global_pg is not None, "Global PG not initialized."
        handles = []
        for shape, dtype in zip(shapes, dtypes):
            buf = torch.empty(shape, dtype=dtype, device=self.device)
            self._global_pg.recv([buf], src_rank, 0).wait()
            with self._lock:
                key = f"tw{self.device_id}_{self._counter}"
                self._counter += 1
                self._store[key] = buf
                self._ref_counts[key] = 1
            handles.append(
                GPUTensorHandle(store_key=key, source_id=self.source_id, shape=shape, dtype=dtype, device=self.device)
            )
        return handles
