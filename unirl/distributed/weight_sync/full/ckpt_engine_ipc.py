"""Colocate SGLang full-weight sync via checkpoint-engine ZMQ + CUDA IPC."""

from __future__ import annotations

import logging
import os
import secrets
import threading
import zlib
from typing import Any, Dict, Optional

from unirl.config.contracts import validate_checkpoint_engine_ipc_options
from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.distributed.weight_sync.full.base import FullWeightSync
from unirl.distributed.weight_sync.transfer.ckpt_engine_transfer import (
    CkptEngineWeightSender,
    CoordinatedWeightSyncError,
)

logger = logging.getLogger(__name__)
_RECEIVER_TIMEOUT_GRACE_S = 30


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
            tp_size = self._rollout._tp_size
            if tp_size < 1:
                raise ValueError(f"CkptEngineIPCWeightSync requires tp_size>=1; got {tp_size}")
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

        sender = self._prepare_sender(zmq_handles, tp_size, local_uuid)

        try:
            operation_error: Optional[BaseException] = None
            try:
                if not is_tp_zero:
                    self._run_sender(sender)
                    logger.debug(
                        "[CkptEngine-IPC] rank %s: pushed weights from local TP GPU %s (tp_rank=%s/%s)",
                        rank,
                        local_uuid,
                        ri.tp_rank,
                        ri.tp_size,
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
        finally:
            sender.close()

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
        if ri is not None and ri.tp_size != tp_size:
            raise RuntimeError(
                f"CkptEngineIPCWeightSync: RankInfo tp_size={ri.tp_size} does not match "
                f"the colocated rollout tp_size={tp_size}."
            )
        cfg = self._rollout.cfg
        engine_kwargs = cfg.engine_kwargs
        validate_checkpoint_engine_ipc_options(
            backend=cfg.backend,
            dp_size=cfg.dp_size,
            engine_kwargs=engine_kwargs,
        )

    def _validate_rollout_capability(self) -> None:
        """Require the dedicated checkpoint-engine rollout contract."""
        required = (
            "update_weights_from_checkpoint_engine_ipc",
            "mark_checkpoint_engine_sync_failed",
            "shutdown",
        )
        missing = [name for name in required if not callable(getattr(self._rollout, name, None))]
        if missing:
            raise TypeError(f"CkptEngineIPCWeightSync rollout is missing required capabilities: {missing}")
        ri = self.rank_info
        if ri is None or ri.tp_rank == 0:
            backend = getattr(self._rollout, "_backend", None)
            if backend is None or not callable(getattr(backend, "update_from_checkpoint_engine_ipc", None)):
                raise TypeError("CkptEngineIPCWeightSync currently supports only the SGLang HTTP backend")

    def _prepare_sender(
        self,
        zmq_handles: Dict[str, str],
        tp_size: int,
        local_uuid: str,
    ):
        """Select this TP rank's colocated endpoint and allocate its sender."""
        sender = None
        prepare_error = None
        try:
            ri = self.rank_info
            tp_rank = ri.tp_rank if ri is not None else 0
            local_path = zmq_handles.get(local_uuid)
            if local_path is None:
                raise RuntimeError(
                    "CkptEngineIPCWeightSync: the current GPU has no matching IPC endpoint; "
                    f"uuid={local_uuid!r}, tp_rank={tp_rank}, tp_size={tp_size}, "
                    f"available_uuids={sorted(zmq_handles)!r}"
                )
            sender = CkptEngineWeightSender(
                socket_path=local_path,
                bucket_size_mb=self._bucket_bytes // (1024 * 1024),
                timeout_s=self._timeout_s,
            )
            sender.prepare()
        except BaseException as exc:
            prepare_error = exc

        try:
            self._sender_consensus(prepare_error, "prepare")
        except BaseException:
            if sender is not None:
                sender.close()
            raise
        return sender

    def _run_exchange(self, sender, zmq_handles: Dict[str, str]) -> None:
        """Run the HTTP receiver beside the main-thread sender."""
        recv_error: dict = {}

        def _spawn_receiver() -> None:
            """Trigger the SGLang engine to connect its REP sockets."""
            try:
                self._rollout.update_weights_from_checkpoint_engine_ipc(
                    zmq_handles=zmq_handles,
                    flush_cache=self._flush_cache,
                    timeout_s=self._timeout_s + _RECEIVER_TIMEOUT_GRACE_S,
                )
            except Exception as exc:
                recv_error["exc"] = exc

        recv_thread = threading.Thread(target=_spawn_receiver, daemon=True)
        recv_thread.start()
        sender_error: Optional[BaseException] = None
        try:
            self._run_sender(sender)
        except BaseException as exc:
            sender_error = exc

        # A checkpoint-engine worker has no receive timeout. If a REQ socket
        # times out while awaiting an ACK, it cannot send the protocol's abort
        # payload, so terminate SRT rather than leave its REP workers wedged.
        recv_thread.join(timeout=5 if sender_error is not None else _RECEIVER_TIMEOUT_GRACE_S)
        if sender_error is not None or recv_thread.is_alive():
            operation_error = sender_error or TimeoutError("CkptEngineIPCWeightSync: receiver thread did not stop")
            try:
                self._rollout.shutdown()
            except Exception:
                logger.exception("Failed to terminate SGLang after a checkpoint-engine transfer failure")
            recv_thread.join(timeout=10)
            if recv_thread.is_alive():
                logger.error("Checkpoint-engine receiver thread remained alive after SGLang shutdown")
            raise operation_error
        if "exc" in recv_error:
            raise RuntimeError("CkptEngineIPCWeightSync: rollout receiver failed") from recv_error["exc"]

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
        try:
            state = torch.tensor(
                [phase_id, -phase_id, error is None],
                dtype=torch.int64,
                device=f"cuda:{torch.cuda.current_device()}",
            )
            dist.all_reduce(state, op=dist.ReduceOp.MIN)
            min_phase, neg_max_phase, all_succeeded = state.cpu().tolist()
        except BaseException as exc:
            raise CoordinatedWeightSyncError(f"CkptEngineIPCWeightSync consensus failed during {phase}") from exc

        if min_phase != -neg_max_phase:
            raise CoordinatedWeightSyncError(
                "CkptEngineIPCWeightSync ranks reached different protocol phases; "
                f"local_phase={phase!r}, observed_phase_id_range=({min_phase}, {-neg_max_phase})"
            ) from error
        if not all_succeeded:
            raise CoordinatedWeightSyncError(f"CkptEngineIPCWeightSync failed during {phase}") from error

    @staticmethod
    def _get_current_gpu_uuid() -> str:
        """Return the UUID of this Ray worker's CUDA device."""
        import torch

        uuid = torch.cuda.get_device_properties(torch.cuda.current_device()).uuid
        return f"GPU-{uuid!s}"

    def _build_zmq_handles(self, tp_size: int, local_uuid: str) -> Dict[str, str]:
        """Build ``{device_uuid: zmq_socket_path}`` for this TP group."""
        import socket

        import torch.distributed as dist

        ri = self.rank_info
        local_endpoint = f"ipc://@unirl-ce-{os.getpid()}-{secrets.token_hex(6)}"
        if ri is not None and dist.is_initialized() and dist.get_world_size() > 1:
            local = {
                "dp_rank": ri.dp_rank,
                "pp_rank": ri.pp_rank,
                "tp_rank": ri.tp_rank,
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
