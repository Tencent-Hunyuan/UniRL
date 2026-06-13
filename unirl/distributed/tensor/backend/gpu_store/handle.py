"""GPUTensorHandle — tensor reference for the Single Controller.

A reference-counted handle to a single tensor stored in a worker. It manages its
own GC via weakref.finalize and flows between worker and controller via Ray RPC
(pickle). This is the canonical handle shared by the gpu_store and colocate_store
backends.
"""

from __future__ import annotations

import logging
import weakref
from typing import Any

import ray
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class GPUTensorHandle:
    """Handle to a single tensor stored in a worker.

    Two phases of life:
      1. Worker side (after put()): pure data, NO worker_handle.
      2. Controller side (after rebind()): worker_handle set, finalize registered.

    source_id identifies the worker/device that owns the tensor. worker_handle
    points to the worker actor; the GC finalizer relays a decref into the
    worker's transport via worker_handle.transport_op.remote("decref", store_key).
    """

    __slots__ = (
        "store_key",
        "source_id",
        "shape",
        "dtype",
        "device",
        "worker_handle",
        "_finalized",
        "__weakref__",
        "object_ref",
    )

    def __init__(self, store_key: str, source_id, shape: tuple, dtype: torch.dtype, device: str, object_ref=None):
        self.store_key = store_key
        self.source_id = source_id
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.worker_handle = None
        self._finalized = False
        self.object_ref = object_ref  # Ray ObjectRef for CPU tensors in plasma store

    # ── Controller-side activation ──

    def rebind(self, worker_handle) -> None:
        """Attach worker actor handle and register release finalizer.

        Must only be called once. Calling rebind on an already-bound handle
        indicates a bug (e.g. double-processing in _rebind_tree).

        CPU tensors (object_ref is not None): worker_handle is recorded but no
        finalizer registered — lifecycle managed by ObjectRef Python refcount,
        no decref RPC needed.
        """
        assert not self._finalized, (
            f"GPUTensorHandle {self.store_key!r} is already bound. rebind() must only be called once."
        )
        self._finalized = True
        self.worker_handle = worker_handle
        if self.object_ref is None:
            # CUDA tensor: register finalizer for decref RPC
            weakref.finalize(self, GPUTensorHandle._release, worker_handle, self.store_key)

    # ── Remote operations ──

    def remote_op_async(self, op: str, *args) -> Any:
        """Fire remote tensor_op on the worker, return ObjectRef (does not wait)."""
        assert self.worker_handle is not None, "GPUTensorHandle not bound"
        return self.worker_handle.transport_op.remote("tensor_op", self, op, *args)

    def remote_op(self, op: str, *args) -> GPUTensorHandle:
        """Execute remote tensor_op synchronously, return new bound GPUTensorHandle."""
        new_h = ray.get(self.remote_op_async(op, *args))
        new_h.rebind(self.worker_handle)
        return new_h

    def local(self) -> Tensor:
        """Fetch the actual tensor to controller CPU."""
        if self.object_ref is not None:
            return ray.get(self.object_ref)
        assert self.worker_handle is not None, "GPUTensorHandle not bound to worker"
        return ray.get(self.worker_handle.transport_op.remote("get_cpu", self))

    # ── Copy protocols ──

    def __copy__(self) -> GPUTensorHandle:
        if self.worker_handle is not None and self.object_ref is None:
            # CUDA tensor: explicit incref to keep tensor alive in the worker
            self.worker_handle.transport_op.remote("incref", self.store_key)
        clone = GPUTensorHandle(
            self.store_key, self.source_id, self.shape, self.dtype, self.device, object_ref=self.object_ref
        )
        if self.worker_handle is not None:
            clone.rebind(self.worker_handle)
        return clone

    def __deepcopy__(self, memo) -> GPUTensorHandle:
        clone = self.__copy__()
        memo[id(self)] = clone
        return clone

    # ── Pickle protocol (for Ray RPC) ──

    def __getstate__(self) -> dict:
        return {
            "store_key": self.store_key,
            "source_id": self.source_id,
            "shape": self.shape,
            "dtype": self.dtype,
            "device": self.device,
            "object_ref": self.object_ref,
        }

    def __setstate__(self, state: dict) -> None:
        self.store_key = state["store_key"]
        self.source_id = state["source_id"]
        self.shape = state["shape"]
        self.dtype = state["dtype"]
        self.device = state["device"]
        self.worker_handle = None
        self._finalized = False
        self.object_ref = state.get("object_ref")  # None for CUDA handles

    # ── Release callback ──

    @staticmethod
    def _release(worker_handle, store_key: str) -> None:
        """GC callback: tell the worker to decref this tensor.

        Skipped if Ray is not initialized — happens during interpreter shutdown
        when module-scoped fixtures have already called ray.shutdown().
        """
        try:
            if not ray.is_initialized():
                return
            worker_handle.transport_op.remote("decref", store_key)
        except Exception:
            logger.debug("Failed to release remote tensor %s.", store_key, exc_info=True)

    def __repr__(self) -> str:
        bound = "bound" if self.worker_handle else "unbound"
        return f"GPUTensorHandle({self.shape}, {self.dtype}, source_id={self.source_id}, {bound})"
