"""Bucketed weight transfer via ZMQ + IPC (or shared memory fallback)."""

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

_ZMQ_TIMEOUT_S = 600
_ZMQ_TIMEOUT_MS = _ZMQ_TIMEOUT_S * 1000


class TensorMetadata(TypedDict):
    name: str
    shape: torch.Size
    dtype: torch.dtype
    offset: int


# From https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/rlhf_utils.py
def rebuild_ipc(handle: tuple[Callable, tuple], device_id: int | None = None) -> torch.Tensor:
    """Rebuild a CUDA tensor from an IPC handle, optionally rewriting the device id."""
    func, args = handle
    list_args = list(args)
    if device_id is not None:
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


def _zmq_call(operation: str, func: Callable, *args):
    try:
        return func(*args)
    except zmq.Again as exc:
        raise TimeoutError(f"ZeroMQ {operation} timed out after {_ZMQ_TIMEOUT_S}s") from exc


class BucketedWeightSender:
    """Send model weights via bucketed IPC transfer over ZMQ."""

    def __init__(
        self,
        zmq_handle: str,
        bucket_size_mb: int = 2048,
        use_shm: bool = False,
    ) -> None:
        self.zmq_handle = zmq_handle
        self.bucket_size_mb = int(bucket_size_mb)
        self.bucket_size = self.bucket_size_mb << 20
        self.use_shm = bool(use_shm)

        self.zmq_context = zmq.Context.instance()
        self.socket = None
        self.buffer = None
        self.shm = None

    async def async_send_weights(self, weights) -> None:
        """Send weights to the receiver. Accepts a sync generator or async iterator."""
        try:
            self._init_socket()
            self._init_buffer()

            offset = 0
            bucket_meta: dict[str, TensorMetadata] = {}
            async for name, weight in _ensure_async_iterator(weights):
                if offset + weight.nbytes > self.bucket_size:
                    torch.cuda.synchronize()
                    _zmq_call(
                        "bucket metadata send", self.socket.send_pyobj, {"bucket_meta": bucket_meta, "is_last": False}
                    )
                    _zmq_call("bucket acknowledgement receive", self.socket.recv)
                    bucket_meta = {}
                    offset = 0

                # TODO: Chunk embedding weights before transfer.
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
            _zmq_call(
                "final bucket metadata send", self.socket.send_pyobj, {"bucket_meta": bucket_meta, "is_last": True}
            )
            _zmq_call("final bucket acknowledgement receive", self.socket.recv)
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
        self.socket.setsockopt(zmq.RCVTIMEO, _ZMQ_TIMEOUT_MS)
        self.socket.setsockopt(zmq.SNDTIMEO, _ZMQ_TIMEOUT_MS)
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
            self.buffer = buffer
            handle = reduce_tensor(buffer)
            _zmq_call("CUDA IPC metadata send", self.socket.send_pyobj, handle)
        else:
            import uuid

            shm_name = f"diffrl_weights_{uuid.uuid4().hex}"
            shm = create_shared_memory(self.bucket_size, shm_name)
            buffer = torch.frombuffer(shm.buf, dtype=torch.uint8)
            self.buffer = buffer
            self.shm = shm

            comm_metadata = {"name": shm_name, "size": self.bucket_size}
            _zmq_call("shared-memory metadata send", self.socket.send_pyobj, comm_metadata)

        _zmq_call("buffer initialization acknowledgement receive", self.socket.recv)

    def _cleanup(self) -> None:
        if self.socket is not None:
            self.socket.close(linger=0)
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
    """Receive model weights via bucketed IPC transfer over ZMQ."""

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
        self._completed = False

    def receive_weights(self, on_bucket_received: Callable[[list], None]) -> None:
        """Receive weights from sender and process each bucket via callback."""
        try:
            self._init_socket()
            self._init_buffer()

            while True:
                metadata = _zmq_call("bucket metadata receive", self.socket.recv_pyobj)
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
                _zmq_call("bucket acknowledgement send", self.socket.send, b"")
                del weights, tensor
                if metadata["is_last"]:
                    self._completed = True
                    break
        finally:
            self._cleanup()

    def _init_socket(self) -> None:
        """Initialize ZMQ REP socket and connect."""
        self.socket = self.zmq_context.socket(zmq.REP)
        self.socket.setsockopt(zmq.RCVTIMEO, _ZMQ_TIMEOUT_MS)
        self.socket.setsockopt(zmq.SNDTIMEO, _ZMQ_TIMEOUT_MS)
        self.socket.connect(self.zmq_handle)

    def _init_buffer(self) -> None:
        """Receive and rebuild communication buffer from sender."""
        comm_metadata = _zmq_call("buffer initialization metadata receive", self.socket.recv_pyobj)
        buffer, shm = None, None
        if not self.use_shm:
            handle = comm_metadata
            buffer = rebuild_ipc(handle, self.device.index)
            assert buffer.dtype == torch.uint8
        else:
            shm_name = comm_metadata["name"]
            shm_size = comm_metadata["size"]
            buffer, shm = rebuild_shared_memory(shm_name, shm_size, dtype=torch.uint8)
        self.buffer = buffer
        self.shm = shm
        _zmq_call("buffer initialization acknowledgement send", self.socket.send, b"")

    def _cleanup(self) -> None:
        if self.socket is not None:
            self.socket.close(linger=_ZMQ_TIMEOUT_MS if self._completed else 0)
            self.socket = None
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
