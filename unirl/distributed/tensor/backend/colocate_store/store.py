"""TensorStore — worker-local tensor registry with ref-counting."""

from __future__ import annotations

import os
import threading
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import ray
import torch
import torch.distributed as dist
from torch import Tensor

from unirl.distributed.tensor.backend.colocate_store.handle import ColocateTensorHandle


class TensorStore:
    """Worker-local GPU tensor registry with reference counting."""

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

        self._store: Dict[str, Tensor] = {}
        self._ref_counts: Dict[str, int] = {}
        self._lock = threading.Lock()
        self._counter = 0

        self._global_pg = None

    def put(self, tensor: Tensor) -> ColocateTensorHandle:
        """Store a tensor and return a lightweight ColocateTensorHandle."""
        if not tensor.is_cuda:
            return ColocateTensorHandle(
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

        return ColocateTensorHandle(
            store_key=key,
            source_id=self.worker_id,
            shape=tuple(t.shape),
            dtype=t.dtype,
            device=str(t.device),
        )

    def get(self, handle: ColocateTensorHandle) -> Tensor:
        """Return the stored tensor for this handle."""
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
        """Increment reference count. Called by ColocateTensorHandle copy on controller."""
        with self._lock:
            if key not in self._ref_counts:
                raise KeyError(f"TensorStore: cannot incref unknown key '{key}'")
            self._ref_counts[key] += 1

    def decref(self, key: str) -> None:
        """Decrement reference count. If zero, release the storage."""
        with self._lock:
            if key not in self._ref_counts:
                raise KeyError(f"TensorStore: cannot decref unknown key '{key}'")
            self._ref_counts[key] -= 1
            if self._ref_counts[key] <= 0:
                del self._store[key]
                del self._ref_counts[key]

    def setup_global_pg(self, global_rank: int, global_world_size: int) -> None:
        """Initialize the global ProcessGroup for cross-worker NCCL transfers."""
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
        self._global_pg.eager_connect_single_device(torch.device(self.device))

    def _nccl_send(self, dst_rank: int, items: List) -> None:
        """Send stored tensors (or row ranges of them) to dst_rank via NCCL."""
        assert self._global_pg is not None, "Global PG not initialized. Call setup_global_pg first."
        for item in items:
            key, start, stop = (item, None, None) if isinstance(item, str) else item
            tensor = self._store[key].detach()
            if start is not None:
                tensor = tensor[start:stop]
            self._global_pg.send([tensor.contiguous()], dst_rank, 0).wait()

    def _nccl_recv(
        self,
        src_global_rank: int,
        shapes: List[Tuple[int, ...]],
        dtypes: List[torch.dtype],
    ) -> List[ColocateTensorHandle]:
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
