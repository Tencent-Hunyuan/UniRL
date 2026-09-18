"""TensorWorker — per-GPU tensor storage, IPC, and NCCL actor."""

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
    """Per-GPU tensor storage, IPC, and NCCL Ray actor."""

    def __init__(self, device_id: int):
        self.device_id = device_id
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

        self._limbo_count: int = 0
        self._limbo_bytes: int = 0
        self._ipc_collect_count: int = int(os.environ.get("MMRL_IPC_COLLECT_COUNT", "128"))
        self._ipc_collect_bytes: int = int(os.environ.get("MMRL_IPC_COLLECT_BYTES", str(1 << 30)))

    def batch_allocate(self, requests: List[Tuple[tuple, torch.dtype]]) -> List[Tuple[str, tuple, tuple]]:
        """Batch-allocate buffers, return [(store_key, ipc_h, stride), ...]."""
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
        """Move bufs from _pending into _store after Worker finishes writing."""
        with self._lock:
            for key in store_keys:
                buf = self._pending.pop(key)
                self._store[key] = buf
                self._ref_counts[key] = 1

    def batch_borrow(self, store_keys: List[str]) -> List[Tuple[tuple, tuple, tuple]]:
        """Batch-create IPC handles for reading, return [(ipc_h, shape, stride), ...]."""
        with self._lock:
            return [
                (self._store[k].untyped_storage()._share_cuda_(), tuple(self._store[k].shape), self._store[k].stride())
                for k in store_keys
            ]

    def incref(self, key: str) -> None:
        """Increment reference count. Called by GPUTensorHandle __copy__ on controller."""
        with self._lock:
            if key not in self._ref_counts:
                raise KeyError(f"TensorWorker: cannot incref unknown key '{key}'")
            self._ref_counts[key] += 1

    def batch_incref(self, keys: List[str]) -> None:
        """Batch-increment reference counts for multiple keys (1 RPC instead of N)."""
        with self._lock:
            for key in keys:
                if key not in self._ref_counts:
                    raise KeyError(f"TensorWorker: cannot incref unknown key '{key}'")
                self._ref_counts[key] += 1

    def decref(self, key: str) -> None:
        """Decrement reference count. If zero, release the storage."""
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

    def tensor_op(self, handle: GPUTensorHandle, op: str, *op_args) -> GPUTensorHandle:
        """Execute a tensor operation directly in _store (no IPC needed)."""
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
        """Clean up remaining IPC Limbo entries and release PyTorch allocator cache."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            with self._lock:
                self._limbo_count = 0
                self._limbo_bytes = 0

    def setup_global_pg(self, global_rank: int, global_world_size: int) -> None:
        """Initialize the global ProcessGroup for cross-GPU NCCL transfers."""
        store = dist.TCPStore(
            host_name=os.environ["MASTER_ADDR"],
            port=int(os.environ["MASTER_PORT"]),
            world_size=global_world_size,
            is_master=(global_rank == 0),
            timeout=timedelta(seconds=30),
        )
        self._global_pg = dist.ProcessGroupNCCL(store, global_rank, global_world_size)
        # Reserve NCCL communicators before models consume device memory.
        self._global_pg.eager_connect_single_device(torch.device(self.device))

    def _nccl_send(self, dst_rank: int, items: List) -> None:
        """Send stored tensors (or row ranges of them) to dst_rank via NCCL."""
        assert self._global_pg is not None, "Global PG not initialized."
        for item in items:
            key, start, end = (item, None, None) if isinstance(item, str) else item
            with self._lock:
                tensor = self._store[key]
            if start is not None:
                tensor = tensor[start:end]
            self._global_pg.send([tensor.contiguous()], dst_rank, 0).wait()

    def _nccl_recv(self, src_rank: int, shapes: List[tuple], dtypes: List[torch.dtype]) -> List[GPUTensorHandle]:
        """Receive tensors from src_rank via NCCL, store in _store."""
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
