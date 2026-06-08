"""TensorStore — worker-local tensor registry with ref-counting.

Each worker holds one TensorStore instance. Tensors stay on the worker's
device; only lightweight TensorHandle refs cross the Ray RPC boundary.

Single-slot per device (DevicePool enforces ``workers_per_device == 1`` for the
colocate backend), so the store never shares tensors across processes — no CUDA-IPC,
no IPC reclaim. Cross-device moves go through NCCL. Colocated multi-slot is gpu_store's
job.

Lifecycle:
  put(tensor) → TensorHandle (ref_count=1)
  incref(key) → ref_count += 1  (called by TensorHandle.__copy__ on controller)
  decref(key) → ref_count -= 1  (called by TensorHandle._release on controller GC)
                if ref_count == 0 → storage freed
"""

from __future__ import annotations

import os
import threading
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import ray
import torch
import torch.distributed as dist
from torch import Tensor

from unirl.distributed.tensor.backend.colocate_store.handle import TensorHandle


class TensorStore:
    """Worker-local GPU tensor registry with reference counting.

    Thread-safe: all mutations protected by a lock for concurrent
    put/get/incref/decref from Ray async calls.
    """

    def __init__(
        self,
        worker_id: str,
        device: str = "cuda:0",
        global_rank: Optional[int] = None,
        global_world_size: Optional[int] = None,
    ):
        self.worker_id = worker_id
        self.device = device
        self.global_rank = global_rank
        self.global_world_size = global_world_size

        # store_key → tensor (holds storage alive)
        self._store: Dict[str, Tensor] = {}
        # store_key → ref count (number of live TensorHandles referencing this key)
        self._ref_counts: Dict[str, int] = {}
        self._lock = threading.Lock()
        self._counter = 0

        # Global ProcessGroup for cross-worker NCCL (initialized lazily)
        self._global_pg = None

    def put(self, tensor: Tensor) -> TensorHandle:
        """Store a tensor and return a lightweight TensorHandle.

        CUDA tensors: a contiguous copy is stored under a fresh key.
        CPU tensors: stored in Ray plasma store via ray.put(); not tracked here.
          Lifecycle managed by ObjectRef Python refcount (no decref RPC needed).
        """
        if not tensor.is_cuda:
            return TensorHandle(
                store_key=None,
                source_id=self.worker_id,
                shape=tuple(tensor.shape),
                dtype=tensor.dtype,
                device=str(tensor.device),
                object_ref=ray.put(tensor.detach()),
            )

        t = tensor.detach().contiguous()
        with self._lock:
            key = f"{self.worker_id}_{self._counter}"
            self._counter += 1
            self._store[key] = t
            self._ref_counts[key] = 1

        return TensorHandle(
            store_key=key,
            source_id=self.worker_id,
            shape=tuple(t.shape),
            dtype=t.dtype,
            device=str(t.device),
        )

    def get(self, handle: TensorHandle) -> Tensor:
        """Return the stored tensor for this handle.

        This is the only safe public API for retrieving a stored tensor.
        Always use this — never access the store by key directly.
        """
        with self._lock:
            if handle.store_key not in self._store:
                raise KeyError(f"TensorStore: key '{handle.store_key}' not found")
            return self._store[handle.store_key].detach()

    def ref_count(self, key: str) -> int:
        """Return current reference count for a key."""
        with self._lock:
            if key not in self._ref_counts:
                raise KeyError(f"TensorStore: key '{key}' not found")
            return self._ref_counts[key]

    def incref(self, key: str) -> None:
        """Increment reference count. Called by TensorHandle copy on controller."""
        with self._lock:
            if key not in self._ref_counts:
                raise KeyError(f"TensorStore: cannot incref unknown key '{key}'")
            self._ref_counts[key] += 1

    def decref(self, key: str) -> None:
        """Decrement reference count. If zero, release the storage.

        Called by TensorHandle._release (GC finalizer on controller side).
        """
        with self._lock:
            if key not in self._ref_counts:
                raise KeyError(f"TensorStore: cannot decref unknown key '{key}'")
            self._ref_counts[key] -= 1
            if self._ref_counts[key] <= 0:
                del self._store[key]
                del self._ref_counts[key]
                # tensor goes out of scope here → Python GC → GPU memory freed

    # ── Cross-worker NCCL transfer ──

    def setup_global_pg(self, global_rank: int, global_world_size: int) -> None:
        """Initialize the global ProcessGroup for cross-worker NCCL transfers.

        Always uses ProcessGroupNCCL directly via TCPStore so the global dist
        state (init_process_group) is never touched — each Role group manages
        its own dist process group without interference.

        MASTER_ADDR and MASTER_PORT must be set in the environment (injected by
        DevicePool via Worker runtime_env).
        """
        self.global_rank = global_rank
        self.global_world_size = global_world_size

        store = dist.TCPStore(
            host_name=os.environ["MASTER_ADDR"],
            port=int(os.environ["MASTER_PORT"]),
            world_size=global_world_size,
            is_master=(global_rank == 0),
            timeout=timedelta(seconds=30),
        )
        self._global_pg = dist.ProcessGroupNCCL(store, global_rank, global_world_size)

    def _nccl_send(self, dst_rank: int, handles: List[TensorHandle]) -> None:
        """Send tensors to dst_rank via NCCL.

        Uses ProcessGroupNCCL.send() natively so that privately-created PG
        (not registered in the global dist world) works correctly.
        Each tensor is sent separately so send and recv always stay in sync.
        Single-slot per device, so every handle reads from this worker's own store.
        """
        assert self._global_pg is not None, "Global PG not initialized. Call setup_global_pg first."

        for h in handles:
            assert h.object_ref is None, "CPU tensor (object_ref set) must not go through NCCL. Check localize routing."
            tensor = self.get(h).contiguous()
            self._global_pg.send([tensor], dst_rank, 0).wait()

    def _nccl_recv(
        self,
        src_global_rank: int,
        shapes: List[Tuple[int, ...]],
        dtypes: List[torch.dtype],
    ) -> List[TensorHandle]:
        """Receive tensors from another worker via NCCL p2p."""
        assert self._global_pg is not None, "Global PG not initialized. Call setup_global_pg first."

        handles = []
        for shape, dtype in zip(shapes, dtypes):
            buf = torch.empty(shape, dtype=dtype, device=self.device)
            self._global_pg.recv([buf], src_global_rank, 0).wait()
            handles.append(self.put(buf))
        return handles

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def __repr__(self) -> str:
        return f"TensorStore(worker={self.worker_id}, items={len(self)})"
