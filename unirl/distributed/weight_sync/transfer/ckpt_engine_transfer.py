"""Weight transfer via ZMQ + CUDA IPC using the ``checkpoint_engine`` protocol.

This sender is compatible with SGLang's ``update_weights_from_ipc`` path, which
delegates to ``checkpoint_engine.worker.update_weights_from_ipc``. The protocol
differs from verl's ``BucketedWeightSender`` in several ways (see plan):

- **Single reusable buffer**: REQ/REP waits for each receiver before reuse
- **Metadata as ``list[dict]``**: NOT ``dict[str, dict]`` with ``is_last``
- **Termination**: two ``None`` signals (release + post_hook), not ``is_last``
- **No per-tensor IPC handle**: all tensors must fit in the bucket buffer
- **Per-GPU sender**: normally one local buffer and one REQ socket per train rank

The sender creates one or more REQ sockets, allocates one reusable bucket
buffer on the trainer's GPU, and sends the same IPC handle + bucket
metadata to its receiver. Each TP rank's scheduler subprocess creates a REP
socket, rebuilds the buffer view via IPC, and calls ``model.load_weights()``
(which does TP sharding internally).
"""

from __future__ import annotations

import gc
import os
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import torch
import zmq
from torch.multiprocessing.reductions import reduce_tensor


class CkptEngineWeightSender:
    """Send model weights via the checkpoint_engine ZMQ + CUDA IPC protocol.

    Creates one REQ socket per supplied handle, allocates a reusable bucket,
    and sends its IPC handle + bucket metadata to the receiver. In UniRL each
    TP train rank supplies only its colocated rollout GPU, so transfer memory
    and PCIe traffic are distributed instead of concentrated on TP rank zero.

    Args:
        zmq_handles: Dict mapping device UUID to ZMQ socket path.
        bucket_size_mb: Communication buffer size in MB.
    """

    def __init__(
        self,
        zmq_handles: Dict[str, str],
        bucket_size_mb: int = 2048,
        timeout_s: int = 600,
    ) -> None:
        self.socket_paths = list(zmq_handles.values())
        self.bucket_size_mb = int(bucket_size_mb)
        self.bucket_size = self.bucket_size_mb << 20
        self.timeout_ms = int(timeout_s) * 1000

        self.zmq_context = zmq.Context.instance()
        self.sockets: List[zmq.Socket] = []
        self._can_send: List[bool] = []
        self.buffer = None
        self._handle = None
        self._abort_sent = False

    def prepare(self) -> None:
        """Allocate/export the CUDA buffer before receivers start.

        Socket creation is deliberately NOT done here: pyzmq sockets are not
        thread-safe, and the NativeBackend path runs ``send_weights`` on a
        daemon thread while ``prepare`` runs on the engine-owning thread.
        Sockets are created lazily at the top of :meth:`send_weights` so each
        socket's full lifecycle (create/bind/send/recv/close) is confined to
        the one thread that sends.
        """
        self._allocate_buffer()

    def send_weights(
        self,
        weights: Iterator[Tuple[str, "object"]],
        consensus: Callable[[Optional[BaseException], str], None] | None = None,
    ) -> None:
        """Send weights to all TP rank receivers.

        Args:
            weights: Generator yielding (name, tensor) pairs.
        """
        weight = None
        try:
            if self.buffer is None:
                self.prepare()
            # Create/bind sockets in THIS thread (see ``prepare`` docstring):
            # under NativeBackend this is a daemon thread, and pyzmq sockets
            # must live and die on the thread that uses them.
            if not self.sockets:
                self._init_sockets()
            self._exchange(self._handshake, consensus, "handshake")

            offset = 0
            bucket_index = 0
            bucket_meta: List[Dict[str, Any]] = []

            for name, weight in weights:
                weight_nbytes = weight.nbytes

                # Flush current bucket if this tensor doesn't fit
                if offset + weight_nbytes > self.bucket_size and bucket_meta:
                    torch.cuda.synchronize()
                    self._exchange(
                        lambda: self._send_bucket(bucket_meta),
                        consensus,
                        f"bucket-{bucket_index}",
                    )
                    bucket_index += 1
                    offset = 0
                    bucket_meta = []

                # ``raise`` (not ``assert``): the check must survive ``python -O``,
                # otherwise an oversized tensor silently truncates into the next
                # bucket slot and surfaces later as an opaque CUDA copy/shape error.
                if offset + weight_nbytes > self.bucket_size:
                    raise ValueError(
                        f"Weight {name}({weight.shape}, {weight.dtype}) is too large "
                        f"to fit in the bucket ({weight_nbytes} > {self.bucket_size}). "
                        f"Please increase bucket_size_mb (currently {self.bucket_size_mb} MB)."
                    )

                bucket_meta.append(
                    {
                        "name": name,
                        "shape": weight.shape,
                        "dtype": weight.dtype,
                        "offset": offset,
                    }
                )
                self.buffer[offset : offset + weight_nbytes].copy_(weight.view(-1).view(torch.uint8), non_blocking=True)
                offset += weight_nbytes
                weight = None

            # Send the last bucket
            if bucket_meta:
                torch.cuda.synchronize()
                self._exchange(
                    lambda: self._send_bucket(bucket_meta),
                    consensus,
                    f"bucket-{bucket_index}",
                )

            # Match checkpoint-engine's lifecycle: release both sides of the IPC
            # allocation before running a potentially memory-heavy post-hook.
            self._exchange(self._send_none_to_all, consensus, "release")
            weight = None
            self._release_buffer()
            self._exchange(self._send_none_to_all, consensus, "post-hook")

        except BaseException as exc:
            self._abort_receivers(exc)
            raise
        finally:
            close = getattr(weights, "close", None)
            if callable(close):
                close()
            weight = None
            self._cleanup()

    @staticmethod
    def _exchange(
        operation: Callable[[], None],
        consensus: Callable[[Optional[BaseException], str], None] | None,
        phase: str,
    ) -> None:
        error = None
        try:
            operation()
        except BaseException as exc:
            error = exc
        if consensus is not None:
            consensus(error, phase)
        if error is not None:
            raise error

    def _init_sockets(self) -> None:
        """Create one REQ socket per supplied device handle."""
        for path in self.socket_paths:
            if path.startswith("ipc://"):
                ipc_path = path[len("ipc://") :]
                try:
                    os.remove(ipc_path)
                except OSError:
                    pass
            sock = self.zmq_context.socket(zmq.REQ)
            sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
            sock.bind(path)
            self.sockets.append(sock)
            self._can_send.append(True)

    def _allocate_buffer(self) -> None:
        """Allocate and export the reusable CUDA buffer."""
        # _send_bucket waits for every receiver's ACK before this buffer is reused,
        # so a second half cannot overlap useful work and only increases peak VRAM.
        self.buffer = torch.empty(
            self.bucket_size,
            dtype=torch.uint8,
            device=f"cuda:{torch.cuda.current_device()}",
        )
        self._handle = reduce_tensor(self.buffer)

    def _handshake(self) -> None:
        """Send the prepared IPC handle to every receiver."""
        # Send the IPC handle to every supplied receiver, then collect acks.
        for i, sock in enumerate(self.sockets):
            sock.send_pyobj(self._handle)
            self._can_send[i] = False
        errors = []
        for i, sock in enumerate(self.sockets):
            ack = sock.recv()
            self._can_send[i] = True
            if ack != b"":
                errors.append(ack.decode("utf-8", errors="replace"))
                # The worker's handshake error path waits for one raw ACK before
                # raising and closing; it has not entered the payload state machine.
                sock.send(b"")
                self._can_send[i] = False
        if errors:
            raise RuntimeError(f"CkptEngineWeightSender: receiver handshake failed: {errors[0]}")

    def _send_bucket(self, metadata: List[Dict[str, Any]]) -> None:
        """Send bucket metadata to every supplied receiver and collect acks."""
        # Send to all sockets
        for i, sock in enumerate(self.sockets):
            sock.send_pyobj(metadata)
            self._can_send[i] = False
        # Collect acks from all sockets
        errors = []
        for i, sock in enumerate(self.sockets):
            ack = sock.recv()
            self._can_send[i] = True
            if ack != b"":
                errors.append(f"receiver {i}: {ack.decode('utf-8', errors='replace')}")
        if errors:
            raise RuntimeError(f"CkptEngineWeightSender: bucket load failed: {errors[0]}")

    def _send_none_to_all(self) -> None:
        """Send None to every supplied receiver and collect acks."""
        for i, sock in enumerate(self.sockets):
            sock.send_pyobj(None)
            self._can_send[i] = False
        for i, sock in enumerate(self.sockets):
            sock.recv()
            self._can_send[i] = True

    def _abort_receivers(self, error: BaseException) -> None:
        """Release workers that entered checkpoint-engine's payload loop."""
        abort = RuntimeError(f"CkptEngineWeightSender aborted: {error}")
        for i, sock in enumerate(self.sockets):
            if not self._can_send[i]:
                continue
            try:
                # checkpoint_engine.worker raises this payload without replying.
                sock.send_pyobj(abort)
                self._can_send[i] = False
                self._abort_sent = True
            except Exception:
                pass

    def close(self) -> None:
        """Release a prepared sender that did not enter ``send_weights``."""
        self._cleanup()

    def _cleanup(self) -> None:
        """Close all sockets and release the buffer."""
        for sock in self.sockets:
            try:
                sock.close(linger=5000 if self._abort_sent else 0)
            except Exception:
                pass
        self.sockets = []
        self._can_send = []

        for path in self.socket_paths:
            if path.startswith("ipc://"):
                ipc_path = path[len("ipc://") :]
                try:
                    os.remove(ipc_path)
                except OSError:
                    pass

        self._release_buffer()

    def _release_buffer(self) -> None:
        """Drop producer IPC storage after the receiver's release ACK."""
        self.buffer = None
        self._handle = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()


__all__ = ["CkptEngineWeightSender"]
