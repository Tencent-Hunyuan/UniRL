"""Colocate SGLang full-weight sync via checkpoint-engine ZMQ + CUDA IPC."""

from __future__ import annotations

import logging
import os
import secrets
import threading
import zlib
from typing import Any, Dict, Optional

from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.distributed.weight_sync.full.base import FullWeightSync
from unirl.distributed.weight_sync.transfer.ckpt_engine_transfer import (
    CkptEngineWeightSender,
    CoordinatedWeightSyncError,
)

logger = logging.getLogger(__name__)


class CkptEngineIPCWeightSync(FullWeightSync):
    """Colocate full-weight sync for SGLang via checkpoint-engine IPC."""

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
        """Push full weights to every colocated SGLang TP scheduler."""
        ri = self.rank_info
        rank = ri.rank if ri is not None else 0
        is_tp_zero = ri is None or ri.tp_rank == 0

        preflight_error: Optional[BaseException] = None
        tp_size = 1
        local_uuid = ""
        try:
            tp_size = self._get_tp_size()
            self._validate_topology(tp_size)
            self._validate_rollout_capability()
            local_uuid = self._get_current_gpu_uuid()
        except BaseException as exc:
            preflight_error = exc
        self._sender_consensus(preflight_error, "preflight")

        topology_error: Optional[BaseException] = None
        zmq_handles: Dict[str, str] = {}
        try:
            zmq_handles = self._build_zmq_handles(tp_size, local_uuid)
        except BaseException as exc:
            topology_error = exc
        self._sender_consensus(topology_error, "topology")

        sender, tp_rank = self._prepare_local_sender(zmq_handles, tp_size, local_uuid)

        operation_error: Optional[BaseException] = None
        try:
            if not is_tp_zero:
                self._run_sender(sender)
                logger.debug(
                    "[CkptEngine-IPC] rank %s: pushed weights from local TP GPU %s (tp_rank=%s/%s)",
                    rank,
                    local_uuid,
                    tp_rank,
                    ri.tp_size if ri else 1,
                )
            else:
                logger.info(
                    "[CkptEngine-IPC] rank %s: pushing full weights to %d TP rank(s) via checkpoint_engine IPC",
                    rank,
                    tp_size,
                )
                self._run_exchange(sender, zmq_handles)
                logger.info("[CkptEngine-IPC] rank %s: full weight sync completed", rank)
        except CoordinatedWeightSyncError as exc:
            self._poison_rollout(exc)
            raise
        except BaseException as exc:
            operation_error = exc

        try:
            self._sender_consensus(operation_error, "complete")
        except BaseException as exc:
            self._poison_rollout(exc)
            raise

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
        engine_kwargs = dict(getattr(cfg, "engine_kwargs", None) or {})
        server_dp = getattr(cfg, "dp_size", None)
        if server_dp is None:
            server_dp = engine_kwargs.get("dp_size")
        if int(server_dp or 1) != 1:
            raise NotImplementedError("CkptEngineIPCWeightSync does not support SGLang server-level dp_size>1")
        if any(key.startswith("speculative") for key in engine_kwargs):
            raise NotImplementedError("CkptEngineIPCWeightSync does not support SGLang speculative decoding workers")

    def _validate_rollout_capability(self) -> None:
        """Require the dedicated checkpoint-engine rollout contract."""
        update = getattr(self._rollout, "update_weights_from_checkpoint_engine_ipc", None)
        poison = getattr(self._rollout, "mark_checkpoint_engine_sync_failed", None)
        if not callable(update) or not callable(poison):
            raise TypeError("CkptEngineIPCWeightSync requires a checkpoint-engine-capable rollout engine")
        ri = self.rank_info
        if ri is None or ri.tp_rank == 0:
            backend = getattr(self._rollout, "_backend", None)
            if backend is None or not callable(getattr(backend, "update_from_ipc", None)):
                raise TypeError("CkptEngineIPCWeightSync currently supports only the SGLang HTTP backend")

    def _prepare_local_sender(
        self,
        zmq_handles: Dict[str, str],
        tp_size: int,
        local_uuid: str,
    ):
        """Select this TP rank's colocated endpoint and allocate its sender."""
        ri = self.rank_info
        tp_rank = int(ri.tp_rank) if ri is not None else 0
        local_path = zmq_handles.get(local_uuid)
        if local_path is None:
            raise RuntimeError(
                "CkptEngineIPCWeightSync: the current GPU has no matching IPC endpoint; "
                f"uuid={local_uuid!r}, tp_rank={tp_rank}, tp_size={tp_size}, "
                f"available_uuids={sorted(zmq_handles)!r}"
            )
        return self._prepare_sender({local_uuid: local_path}), tp_rank

    def _run_exchange(self, sender, zmq_handles: Dict[str, str]) -> None:
        """Run the HTTP receiver beside the main-thread sender."""
        recv_error: dict = {}

        def _spawn_receiver() -> None:
            """Trigger the SGLang engine to connect its REP sockets."""
            try:
                self._rollout.update_weights_from_checkpoint_engine_ipc(
                    zmq_handles=zmq_handles,
                    flush_cache=self._flush_cache,
                    track_prefix=self._track_prefix,
                    timeout_s=self._timeout_s + 30,
                )
            except Exception as exc:
                recv_error["exc"] = exc

        self._run_http_exchange(_spawn_receiver, sender)
        if "exc" in recv_error:
            raise RuntimeError("CkptEngineIPCWeightSync: rollout receiver failed") from recv_error["exc"]

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

        try:
            self._sender_consensus(prepare_error, "prepare")
        except BaseException:
            sender.close()
            raise
        return sender

    def _run_sender(self, sender) -> None:
        """Stream weights through a prepared CkptEngineWeightSender."""
        sender.send_weights(self._iter_full_tensors(), consensus=self._sender_consensus)

    @staticmethod
    def _sender_consensus(error: Optional[BaseException], phase: str) -> None:
        """Fail every train rank together and reject mismatched phases."""
        import torch
        import torch.distributed as dist

        if not dist.is_initialized() or dist.get_world_size() == 1:
            if error is not None:
                raise CoordinatedWeightSyncError(f"CkptEngineIPCWeightSync failed during {phase}") from error
            return

        phase_id = zlib.crc32(phase.encode("utf-8"))
        state = torch.tensor(
            [phase_id, -phase_id, int(error is None)],
            dtype=torch.int64,
            device=f"cuda:{torch.cuda.current_device()}",
        )
        dist.all_reduce(state, op=dist.ReduceOp.MIN)
        min_phase, neg_max_phase, all_succeeded = (int(value) for value in state.cpu().tolist())

        if min_phase != -neg_max_phase:
            raise CoordinatedWeightSyncError(
                "CkptEngineIPCWeightSync ranks reached different protocol phases; "
                f"local_phase={phase!r}, observed_phase_id_range=({min_phase}, {-neg_max_phase})"
            ) from error
        if not all_succeeded:
            raise CoordinatedWeightSyncError(f"CkptEngineIPCWeightSync failed during {phase}") from error

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

    @staticmethod
    def _new_zmq_endpoint() -> str:
        """Return a process- and update-unique Linux abstract IPC endpoint."""
        return f"ipc://@unirl-ce-{os.getpid()}-{secrets.token_hex(6)}"

    def _build_zmq_handles(self, tp_size: int, local_uuid: str) -> Dict[str, str]:
        """Build ``{device_uuid: zmq_socket_path}`` for this TP group."""
        import socket

        import torch.distributed as dist

        ri = self.rank_info
        local_endpoint = self._new_zmq_endpoint()
        if ri is not None and dist.is_initialized() and dist.get_world_size() > 1:
            local = {
                "dp_rank": int(ri.dp_rank),
                "pp_rank": int(ri.pp_rank),
                "tp_rank": int(ri.tp_rank),
                "host": socket.gethostname(),
                "uuid": local_uuid,
                "endpoint": local_endpoint,
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
                if len({item["uuid"] for item in items}) != tp_size:
                    raise RuntimeError(f"CkptEngineIPCWeightSync: duplicate GPU UUIDs in TP group {key}: {items}")
                if len({item["endpoint"] for item in items}) != tp_size:
                    raise RuntimeError(f"CkptEngineIPCWeightSync: duplicate IPC endpoints in TP group {key}: {items}")
                hosts = {item["host"] for item in items}
                if len(hosts) != 1:
                    raise NotImplementedError(
                        "CkptEngineIPCWeightSync requires every TP group to be colocated on one node; "
                        f"group={key}, hosts={sorted(hosts)}"
                    )
            group = groups[(local["dp_rank"], local["pp_rank"])]
            return {item["uuid"]: item["endpoint"] for item in group}

        if tp_size != 1:
            raise RuntimeError("CkptEngineIPCWeightSync requires an initialized distributed group when tp_size>1")
        return {local_uuid: local_endpoint}

    def _poison_rollout(self, error: BaseException) -> None:
        """Prevent generation after a possibly partial live-weight update."""
        try:
            self._rollout.mark_checkpoint_engine_sync_failed(str(error))
        except Exception:
            logger.exception("Failed to mark rollout unhealthy after checkpoint-engine sync failure")


__all__ = ["CkptEngineIPCWeightSync"]
