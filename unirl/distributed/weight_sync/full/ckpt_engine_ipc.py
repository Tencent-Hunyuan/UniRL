"""Full-weight IPC sync for SGLang via the ``checkpoint_engine`` protocol.

Colocate-only: the trainer and SGLang engine share the same node. Uses
``checkpoint_engine.worker.update_weights_from_ipc`` (ZMQ REQ/REP + CUDA IPC
shared buffer) for zero-copy weight transfer — the receiver reconstructs
tensor **views** into the shared buffer and calls ``model.load_weights()``
in-place, so no extra GPU memory is allocated on the rollout side.

This is the SGLang analogue of :class:`~unirl.distributed.weight_sync.full.ipc.IPCWeightSync`
(which is vLLM-Omni only). The protocols differ fundamentally — see
:class:`~unirl.distributed.weight_sync.transfer.ckpt_engine_transfer.CkptEngineWeightSender`.

Thread discipline:
    - **NativeBackend**: ``engine.update_weights_from_ipc()`` calls
      ``loop.run_until_complete()`` which MUST run on the engine's thread.
      So the receiver runs in the **main thread** and the sender in a
      **daemon thread** (inverted from IPCWeightSync).
    - **HTTPBackend**: ``update_from_ipc`` is a blocking HTTP POST (thread-safe).
      The receiver runs in a **daemon thread** and the sender in the
      **main thread** (same as IPCWeightSync).
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, Optional

from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.distributed.weight_sync.full.base import FullWeightSync

logger = logging.getLogger(__name__)


class CkptEngineIPCWeightSync(FullWeightSync):
    """Colocate full-weight sync for SGLang via checkpoint_engine IPC.

    Uses SGLang's ``update_weights_from_ipc`` API (ZMQ + CUDA IPC shared buffer)
    for zero-copy weight transfer. Every trainer rank allocates one reusable
    bucket on its GPU, shares it via CUDA IPC, and sends bucket metadata over
    ZMQ. Each SGLang
    scheduler subprocess (one per TP rank) creates a REP socket, reconstructs
    tensor views into the shared buffer, and calls ``load_weights`` (which
    handles TP sharding internally).
    """

    def __init__(
        self,
        *,
        backend: Any,
        rollout: Any,
        bucket_size_mb: int = 2048,
        flush_cache: bool = True,
        lora_merged: bool = False,
        adapter_name: Optional[str] = None,
        name_remap: Optional[Dict[str, Optional[str]]] = None,
        track_prefix: str = "",
        wire_dtype: Any = None,
        timeout_s: int = 600,
    ) -> None:
        super().__init__(
            backend=backend,
            bucket_size_mb=bucket_size_mb,
            flush_cache=flush_cache,
            lora_merged=lora_merged,
            adapter_name=adapter_name,
            name_remap=name_remap,
            track_prefix=track_prefix,
            wire_dtype=wire_dtype,
        )
        self._rollout = rollout
        self._timeout_s = int(timeout_s)
        if self._timeout_s <= 0:
            raise ValueError(f"timeout_s must be positive; got {timeout_s}")

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def sync(self) -> None:
        """Push full weights to the SGLang engine via checkpoint_engine IPC.

        Runs on every train rank (BROADCAST); the ``_iter_full_tensors`` walk
        all-gathers each FSDP shard in lockstep. Every TP rank sends through a
        buffer allocated on its own GPU to the colocated SGLang scheduler. The
        TP-zero rank additionally starts the engine-side receiver fan-out.
        """
        ri = self.rank_info
        rank = ri.rank if ri is not None else 0
        is_tp_zero = ri is None or ri.tp_rank == 0

        tp_size = self._get_tp_size()
        self._validate_topology(tp_size)
        zmq_handles = self._build_zmq_handles(tp_size)
        sender, local_uuid, tp_rank = self._prepare_local_sender(zmq_handles, tp_size)

        if not is_tp_zero:
            self._run_sender(sender)
            logger.debug(
                "[CkptEngine-IPC] rank %s: pushed weights from local TP GPU %s (tp_rank=%s/%s)",
                rank,
                local_uuid,
                tp_rank,
                ri.tp_size if ri else 1,
            )
            return

        logger.info(
            "[CkptEngine-IPC] rank %s: pushing full weights to %d TP rank(s) via checkpoint_engine IPC",
            rank,
            tp_size,
        )
        self._run_exchange(sender, zmq_handles)
        self.weight_version += 1
        logger.info("[CkptEngine-IPC] rank %s: full weight sync completed", rank)

    def _validate_topology(self, tp_size: int) -> None:
        """Reject SGLang layouts that the checkpoint-engine route cannot map."""
        ri = self.rank_info
        if ri is not None and ri.pp_size > 1:
            raise NotImplementedError(
                "CkptEngineIPCWeightSync: rollout pp_size>1 is not implemented; "
                "stage-local socket routing and parameter filtering are required."
            )
        if ri is not None and int(ri.tp_size) != tp_size:
            raise RuntimeError(
                f"CkptEngineIPCWeightSync: RankInfo tp_size={ri.tp_size} does not match "
                f"the colocated rollout tp_size={tp_size}."
            )
        cfg = getattr(self._rollout, "cfg", None)
        if int(getattr(cfg, "dp_size", None) or 1) != 1:
            raise NotImplementedError("CkptEngineIPCWeightSync does not support SGLang server-level dp_size>1")
        engine_kwargs = dict(getattr(cfg, "engine_kwargs", None) or {})
        if any(key.startswith("speculative") for key in engine_kwargs):
            raise NotImplementedError("CkptEngineIPCWeightSync does not support SGLang speculative decoding workers")

    def _prepare_local_sender(self, zmq_handles: Dict[str, str], tp_size: int):
        """Select this TP rank's colocated endpoint and allocate its sender."""
        ri = self.rank_info
        tp_rank = int(ri.tp_rank) if ri is not None else 0
        try:
            local_uuid, local_path = list(zmq_handles.items())[tp_rank]
        except IndexError as exc:
            raise RuntimeError(
                f"CkptEngineIPCWeightSync: tp_rank={tp_rank} has no IPC handle in a tp_size={tp_size} rollout group"
            ) from exc
        return self._prepare_sender({local_uuid: local_path}), local_uuid, tp_rank

    def _run_exchange(self, sender, zmq_handles: Dict[str, str]) -> None:
        """Run receiver and sender with the backend-required thread placement."""
        recv_error: dict = {}
        sender_error: dict = {}

        def _spawn_receiver() -> None:
            """Trigger the SGLang engine to connect its REP sockets."""
            try:
                self._rollout.update_weights_from_ipc(
                    zmq_handles=zmq_handles,
                    flush_cache=self._flush_cache,
                )
            except Exception as exc:
                recv_error["exc"] = exc

        def _spawn_sender() -> None:
            try:
                self._run_sender(sender)
            except Exception as exc:
                sender_error["exc"] = exc

        if self._receiver_must_run_on_main_thread():
            self._run_native_exchange(_spawn_receiver, _spawn_sender)
        else:
            self._run_http_exchange(_spawn_receiver, sender)

        if "exc" in recv_error:
            raise RuntimeError("CkptEngineIPCWeightSync: rollout receiver failed") from recv_error["exc"]
        if "exc" in sender_error:
            raise RuntimeError("CkptEngineIPCWeightSync: trainer sender failed") from sender_error["exc"]

    def _run_native_exchange(self, receive, send) -> None:
        """Run native receiver on the engine-owning thread."""
        sender_thread = threading.Thread(target=send, daemon=True)
        sender_thread.start()
        try:
            receive()
        finally:
            sender_thread.join(timeout=self._timeout_s + 30)
        if sender_thread.is_alive():
            raise TimeoutError("CkptEngineIPCWeightSync: sender thread did not stop")

    def _run_http_exchange(self, receive, sender) -> None:
        """Run the thread-safe HTTP receiver beside the main-thread sender."""
        recv_thread = threading.Thread(target=receive, daemon=True)
        recv_thread.start()
        try:
            self._run_sender(sender)
        finally:
            recv_thread.join(timeout=self._timeout_s + 30)
        if recv_thread.is_alive():
            raise TimeoutError("CkptEngineIPCWeightSync: receiver thread did not stop")

    def _prepare_sender(self, zmq_handles: Dict[str, str]):
        """Allocate every rank's IPC buffer before starting any receiver."""
        from unirl.distributed.weight_sync.transfer.ckpt_engine_transfer import (
            CkptEngineWeightSender,
        )

        sender = CkptEngineWeightSender(
            zmq_handles=zmq_handles,
            bucket_size_mb=self._bucket_bytes // (1024 * 1024),
            timeout_s=self._timeout_s,
        )
        prepare_error = None
        try:
            sender.prepare()
        except BaseException as exc:
            prepare_error = exc
            sender.close()

        import torch
        import torch.distributed as dist

        if dist.is_initialized() and dist.get_world_size() > 1:
            status = torch.tensor(
                0 if prepare_error is not None else 1,
                dtype=torch.int32,
                device=f"cuda:{torch.cuda.current_device()}",
            )
            dist.all_reduce(status, op=dist.ReduceOp.MIN)
            all_prepared = bool(status.item())
        else:
            all_prepared = prepare_error is None

        if not all_prepared:
            if prepare_error is not None:
                raise RuntimeError("CkptEngineIPCWeightSync: failed to prepare local IPC sender") from prepare_error
            sender.close()
            raise RuntimeError("CkptEngineIPCWeightSync: another train rank failed to prepare its IPC sender")
        return sender

    def _run_sender(self, sender) -> None:
        """Stream weights through a prepared CkptEngineWeightSender."""
        sender.send_weights(self._iter_full_tensors(), consensus=self._sender_consensus)

    @staticmethod
    def _sender_consensus(error: Optional[BaseException], phase: str) -> None:
        """Keep every train rank on the same transport/FSDP boundary."""
        import torch
        import torch.distributed as dist

        if not dist.is_initialized() or dist.get_world_size() == 1:
            if error is not None:
                raise error
            return
        status = torch.tensor(
            0 if error is not None else 1,
            dtype=torch.int32,
            device=f"cuda:{torch.cuda.current_device()}",
        )
        dist.all_reduce(status, op=dist.ReduceOp.MIN)
        if error is not None:
            raise error
        if not bool(status.item()):
            raise RuntimeError(f"CkptEngineIPCWeightSync: a peer sender failed during {phase}")

    def _get_tp_size(self) -> int:
        """Get the SGLang engine's TP size."""
        tp_size = getattr(self._rollout, "_tp_size", 1)
        return int(tp_size) if tp_size else 1

    @staticmethod
    def _get_current_gpu_uuid() -> str:
        """Return the UUID of this Ray worker's CUDA device."""
        import torch

        uuid = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
        return uuid if uuid.startswith("GPU-") else f"GPU-{uuid}"

    def _build_zmq_handles(self, tp_size: int) -> Dict[str, str]:
        """Build the ``{device_uuid: zmq_socket_path}`` dict for all TP ranks.

        Distributed workers exchange their current CUDA UUID and hostname, so
        cluster-global DevicePool IDs are never mistaken for node-local GPU
        ordinals.
        """
        import socket

        import torch.distributed as dist

        ri = self.rank_info
        if ri is not None and dist.is_initialized() and dist.get_world_size() > 1:
            local = {
                "dp_rank": int(ri.dp_rank),
                "pp_rank": int(ri.pp_rank),
                "tp_rank": int(ri.tp_rank),
                "host": socket.gethostname(),
                "uuid": self._get_current_gpu_uuid(),
            }
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, local)
            groups = {}
            for item in gathered:
                groups.setdefault((item["dp_rank"], item["pp_rank"]), []).append(item)
            for key, items in groups.items():
                items.sort(key=lambda item: item["tp_rank"])
                if len(items) != tp_size or [item["tp_rank"] for item in items] != list(range(tp_size)):
                    raise RuntimeError(f"CkptEngineIPCWeightSync: incomplete TP group metadata for {key}: {items}")
                hosts = {item["host"] for item in items}
                if len(hosts) != 1:
                    raise NotImplementedError(
                        "CkptEngineIPCWeightSync requires every TP group to be colocated on one node; "
                        f"group={key}, hosts={sorted(hosts)}"
                    )
            group = groups[(local["dp_rank"], local["pp_rank"])]
            job_id = os.environ.get("RAY_JOB_ID", "default")
            return {item["uuid"]: f"ipc:///tmp/unirl-ckpt-engine-{job_id}-{item['uuid']}.sock" for item in group}

        if tp_size != 1:
            raise RuntimeError("CkptEngineIPCWeightSync requires an initialized distributed group when tp_size>1")
        uuid = self._get_current_gpu_uuid()
        job_id = os.environ.get("RAY_JOB_ID", "default")
        return {uuid: f"ipc:///tmp/unirl-ckpt-engine-{job_id}-{uuid}.sock"}

    def _receiver_must_run_on_main_thread(self) -> bool:
        """Whether IPC receive must run on the rollout engine's owning thread."""
        backend = getattr(self._rollout, "_backend", None)
        return bool(getattr(backend, "requires_main_thread_ipc_receiver", False))


__all__ = ["CkptEngineIPCWeightSync"]
