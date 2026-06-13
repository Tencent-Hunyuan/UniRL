"""Bucketed weight transfer via ZMQ + IPC (or shared memory fallback).

Lifted from upstream ``verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py``
with two adjustments to drop verl-internal dependencies:

- Replace ``verl.utils.device.{get_device_id,get_device_name,get_torch_device}``
  with plain ``torch.cuda`` calls. The original abstraction layer existed to
  support NPU/CPU backends; we run CUDA-only here.
- Inline ``ensure_async_iterator`` (was ``verl.workers.rollout.utils``).

Same algorithm verified by verl-omni in production: one pre-allocated
fixed-size buffer (default 2 GB) shared via CUDA IPC, tensors copied
into the buffer in sender-side chunks, per-bucket metadata sent over a
ZMQ REQ/REP socket. Receiver reconstructs tensor views into the shared
buffer and hands each bucket to a callback.
"""

from __future__ import annotations

import gc
import logging
import os
from multiprocessing import shared_memory
from typing import Any, Callable, TypedDict

import torch
import zmq
from torch.multiprocessing.reductions import reduce_tensor

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class TensorMetadata(TypedDict):
    name: str
    shape: torch.Size
    dtype: torch.dtype
    offset: int


# copy from https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/rlhf_utils.py
def rebuild_ipc(handle: tuple[Callable, tuple], device_id: int | None = None) -> torch.Tensor:
    """Rebuild a CUDA tensor from an IPC handle, optionally rewriting the device id.

    The trainer and worker may have different ``CUDA_VISIBLE_DEVICES``, so
    the same logical tensor lives at a different ``cuda:i`` on each side.
    Pass the receiver-side ``device_id`` to swap the encoded id out.
    """
    func, args = handle
    list_args = list(args)
    if device_id is not None:
        # the key is to change device id to the current device id
        # in case two processes have different CUDA_VISIBLE_DEVICES
        list_args[6] = device_id
    buffer = func(*list_args)
    return buffer


def create_shared_memory(size: int, name: str) -> shared_memory.SharedMemory:
    """Create shared memory for weight transfer. If already exists, attach to it."""
    try:
        shm = shared_memory.SharedMemory(name=name, create=True, size=size)
    except FileExistsError:
        shm = shared_memory.SharedMemory(name=name)
        assert shm.size >= size, f"Stale shm segment '{name}': expected {size} bytes, got {shm.size}"
    return shm


def rebuild_shared_memory(name: str, size: int, dtype: torch.dtype = torch.uint8):
    """Rebuild tensor from shared memory."""
    shm = shared_memory.SharedMemory(name=name)
    tensor = torch.frombuffer(shm.buf[:size], dtype=dtype)
    return tensor, shm


async def _ensure_async_iterator(iterable: Any):
    """Convert an iterable to an async iterator. Inlined from verl.workers.rollout.utils."""
    if hasattr(iterable, "__aiter__"):
        async for item in iterable:
            yield item
    else:
        for item in iterable:
            yield item


class BucketedWeightSender:
    """Send model weights via bucketed IPC transfer over ZMQ.

    Packs weight tensors into a fixed-size communication buffer and sends them
    in buckets to the receiver. Supports CUDA IPC and shared memory fallback.

    Args:
        zmq_handle: ZMQ IPC socket path (e.g., "ipc:///tmp/diffrl-zmq-...sock")
        bucket_size_mb: Communication buffer size in MB
        use_shm: Use shared memory instead of CUDA IPC (for NPU compatibility)
    """

    def __init__(
        self,
        zmq_handle: str,
        bucket_size_mb: int = 2048,
        use_shm: bool = False,
    ) -> None:
        # 2048 MB so the largest single tensor in HI3 (``lm_head.weight``,
        # ~1 GiB at bf16) fits in one bucket. The upstream default of 512 MB
        # would trip the per-tensor assertion below for that param.
        self.zmq_handle = zmq_handle
        self.bucket_size_mb = int(bucket_size_mb)
        self.bucket_size = self.bucket_size_mb << 20
        self.use_shm = bool(use_shm)

        self.zmq_context = zmq.Context.instance()
        self.socket = None
        self.buffer = None
        self.shm = None

    async def async_send_weights(self, weights) -> None:
        """Send weights to the receiver. Accepts a sync generator or async iterator.

        Args:
            weights: Generator or async iterator yielding (name, tensor) pairs
        """
        try:
            self._init_socket()
            self._init_buffer()

            offset = 0
            bucket_meta: dict[str, TensorMetadata] = {}
            async for name, weight in _ensure_async_iterator(weights):
                # model parameters are in fp32 full precision; preserve their
                # dtype rather than force-casting (some — e.g. moe gates — must
                # stay fp32). Receiver will cast on demand if it wants.
                if offset + weight.nbytes > self.bucket_size:
                    torch.cuda.synchronize()
                    self.socket.send_pyobj({"bucket_meta": bucket_meta, "is_last": False})
                    self.socket.recv()
                    bucket_meta = {}
                    offset = 0

                # TODO: slice embedding-layer weight into chunks
                assert offset + weight.nbytes <= self.bucket_size, (
                    f"Weight {name}({weight.shape}, {weight.dtype}) is too large to fit in the bucket. "
                    f"Please increase bucket_size_mb (currently {self.bucket_size_mb} MB)."
                )
                bucket_meta[name] = {
                    "name": name,
                    "shape": weight.shape,
                    "dtype": weight.dtype,
                    "offset": offset,
                }
                self.buffer[offset : offset + weight.nbytes].copy_(weight.view(-1).view(torch.uint8), non_blocking=True)
                offset += weight.nbytes

            torch.cuda.synchronize()
            self.socket.send_pyobj({"bucket_meta": bucket_meta, "is_last": True})
            self.socket.recv()
        finally:
            self._cleanup()

    def _init_socket(self) -> None:
        """Initialize ZMQ REQ socket and bind."""
        if self.zmq_handle.startswith("ipc://"):
            ipc_path = self.zmq_handle[len("ipc://") :]
            try:
                os.remove(ipc_path)
            except OSError:
                pass
        self.socket = self.zmq_context.socket(zmq.REQ)
        self.socket.bind(self.zmq_handle)

    def _init_buffer(self) -> None:
        """Build communication buffer + share its handle with receiver."""
        buffer, shm = None, None
        if not self.use_shm:
            buffer = torch.empty(
                self.bucket_size,
                dtype=torch.uint8,
                device=f"cuda:{torch.cuda.current_device()}",
            )
            handle = reduce_tensor(buffer)
            self.socket.send_pyobj(handle)
        else:
            import uuid

            shm_name = f"diffrl_weights_{uuid.uuid4().hex}"
            shm = create_shared_memory(self.bucket_size, shm_name)
            buffer = torch.frombuffer(shm.buf, dtype=torch.uint8)

            comm_metadata = {"name": shm_name, "size": self.bucket_size}
            self.socket.send_pyobj(comm_metadata)

        self.socket.recv()
        self.buffer = buffer
        self.shm = shm

    def _cleanup(self) -> None:
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        if self.zmq_handle.startswith("ipc://"):
            ipc_path = self.zmq_handle[len("ipc://") :]
            try:
                os.remove(ipc_path)
            except OSError:
                pass
        del self.buffer
        self.buffer = None
        if self.shm is not None:
            self.shm.close()
            self.shm.unlink()
            del self.shm
            self.shm = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()


class BucketedWeightReceiver:
    """Receive model weights via bucketed IPC transfer over ZMQ.

    Receives weight tensors from BucketedWeightSender and passes each
    bucket to a callback for processing (e.g., loading into the model).

    Args:
        zmq_handle: ZMQ IPC socket path (must match sender)
        device: Target device for received tensors
        use_shm: Use shared memory instead of CUDA IPC
    """

    def __init__(
        self,
        zmq_handle: str,
        device: torch.device,
        use_shm: bool = False,
    ) -> None:
        self.zmq_handle = zmq_handle
        self.device = device
        self.use_shm = bool(use_shm)

        self.zmq_context = zmq.Context.instance()
        self.socket = None
        self.buffer = None
        self.shm = None

    def receive_weights(self, on_bucket_received: Callable[[list], None]) -> None:
        """Receive weights from sender and process each bucket via callback.

        Args:
            on_bucket_received: Callback called per bucket with a list of
                ``(name, tensor)`` tuples. Tensors are views into the shared
                buffer; consume immediately (next bucket overwrites).
        """
        try:
            self._init_socket()
            self._init_buffer()

            while True:
                metadata = self.socket.recv_pyobj()
                weights, tensor = [], None
                for name, meta in metadata["bucket_meta"].items():
                    shape, dtype, offset = meta["shape"], meta["dtype"], meta["offset"]
                    size = dtype.itemsize * shape.numel()
                    tensor = self.buffer[offset : offset + size].view(dtype=dtype).view(shape)
                    if self.use_shm:
                        tensor = tensor.to(self.device)
                    weights.append((name, tensor))
                on_bucket_received(weights)
                torch.cuda.synchronize()
                self.socket.send(b"")
                del weights, tensor
                if metadata["is_last"]:
                    break
        finally:
            self._cleanup()

    def _init_socket(self) -> None:
        """Initialize ZMQ REP socket and connect."""
        self.socket = self.zmq_context.socket(zmq.REP)
        self.socket.connect(self.zmq_handle)

    def _init_buffer(self) -> None:
        """Receive and rebuild communication buffer from sender."""
        comm_metadata = self.socket.recv_pyobj()
        buffer, shm = None, None
        if not self.use_shm:
            handle = comm_metadata
            buffer = rebuild_ipc(handle, self.device.index)
            assert buffer.dtype == torch.uint8
        else:
            shm_name = comm_metadata["name"]
            shm_size = comm_metadata["size"]
            buffer, shm = rebuild_shared_memory(shm_name, shm_size, dtype=torch.uint8)
        self.socket.send(b"")
        self.buffer = buffer
        self.shm = shm

    def _cleanup(self) -> None:
        if self.socket is not None:
            self.socket.close()
            self.socket = None
        # Synchronize before releasing the buffer to ensure all async ops
        # referencing it (e.g. clone, .to()) have completed.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        del self.buffer
        self.buffer = None
        if self.shm is not None:
            self.shm.close()
            del self.shm
            self.shm = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()


__all__ = [
    "BucketedWeightSender",
    "BucketedWeightReceiver",
    "TensorMetadata",
    "rebuild_ipc",
    "create_shared_memory",
    "rebuild_shared_memory",
]
