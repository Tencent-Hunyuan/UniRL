"""v2 full-weight IPC sync (COLOCATE, same-node).

Bucketed CUDA-IPC over ZMQ. Full-weight analogue of v1
``distributed/weight_sync/ipc.py`` expressed for the v2 colocate sibling
model: the rollout engine is a LOCAL sibling, so the v1 ``actor.*.remote(...)``
spawn becomes an in-process ``self._rollout.update_weights_from_ipc(...)``.

That call is *blocking* (the engine ``collective_rpc``s into the Omni
subprocess workers, which park in ``BucketedWeightReceiver.receive_weights`` on
the ZMQ socket). v1 got sender/receiver overlap for free from a non-blocking
Ray ``.remote()``; here we recreate it with a ``threading.Thread`` that fires
the receiver while the main thread runs the ``BucketedWeightSender`` pump, then
``thread.join()`` (which is the per-call barrier — no ``dist.barrier`` needed).

CUDA-IPC is same-node only, so this is colocate-only. Per-Worker socket
uniqueness: each colocate engine gets a distinct ``replica_rank`` = this train
rank (the Omni subprocess spawns before any per-Worker env can be set, so we
pass ``replica_rank`` explicitly to the engine instead of relying on
``DIFFRL_REPLICA_RANK``).

Scope: single-node, TP=1, single-stage (SD3). All torch / vllm-omni imports are
deferred so the driver can import this module for ``remote(...)``.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Dict, Optional

from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.distributed.weight_sync.full.base import FullWeightSync


class IPCWeightSync(FullWeightSync):
    """Colocate full-weight sync via bucketed CUDA-IPC over ZMQ."""

    def __init__(
        self,
        *,
        backend: Any,
        rollout: Any,
        bucket_size_mb: int = 2048,
        flush_cache: bool = True,
        lora_merged: bool = False,
        use_shm: bool = False,
        name_remap: Optional[Dict[str, Optional[str]]] = None,
        track_prefix: str = "",
        wire_dtype: Any = None,
    ) -> None:
        # 2048 MB default: the buffer must fit the largest single tensor in one
        # bucket (BucketedWeightSender asserts this).
        super().__init__(
            backend=backend,
            bucket_size_mb=bucket_size_mb,
            flush_cache=flush_cache,
            lora_merged=lora_merged,
            name_remap=name_remap,
            track_prefix=track_prefix,
            wire_dtype=wire_dtype,
        )
        self._rollout = rollout
        self._use_shm = bool(use_shm)

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def sync(self) -> None:
        """Pump full weights to the co-located engine over per-stage sockets.

        Runs on every train rank. Spawns the engine receiver in a thread (so it
        overlaps the sender pump), pumps each stage's socket, then joins.
        """
        from unirl.rollout.engine.vllm_omni.weight_sync.bucketed_transfer import (
            BucketedWeightSender,
        )
        from unirl.rollout.engine.vllm_omni.weight_sync.ipc_dispatch import zmq_handle

        replica_rank = self._my_rank  # distinct per colocate engine → unique socket

        # Discover stages from the engine (TP-per-stage map). SD3 → {0: 1}.
        try:
            stage_ids = sorted(int(s) for s in self._rollout.tp_per_stage().keys())
        except (AttributeError, NotImplementedError):
            stage_ids = [0]
        if not stage_ids:
            stage_ids = [0]

        recv_error: dict = {}

        def _spawn_receivers() -> None:
            # Engine fans to every stage's Omni worker; each parks on its socket.
            try:
                self._rollout.update_weights_from_ipc(
                    peft_config=None,
                    base_sync_done=False,
                    use_shm=self._use_shm,
                    replica_rank=replica_rank,
                    track_prefix=self._track_prefix,
                )
            except Exception as exc:  # surface, don't let the pump hang forever
                recv_error["exc"] = exc

        thread = threading.Thread(target=_spawn_receivers, daemon=True)
        thread.start()
        try:
            for sid in stage_ids:
                # TP=1 → one receiver per stage at local_rank 0. A fresh
                # generator per stage (each stage receives the full state dict).
                handle = zmq_handle(replica_rank=replica_rank, stage_id=int(sid), local_rank=0)
                sender = BucketedWeightSender(
                    zmq_handle=handle,
                    bucket_size_mb=self._bucket_bytes // (1024 * 1024),
                    use_shm=self._use_shm,
                )
                asyncio.run(sender.async_send_weights(self._iter_full_tensors()))
        finally:
            thread.join()
        if "exc" in recv_error:
            raise RuntimeError("IPCWeightSync: rollout receiver failed") from recv_error["exc"]
        self.weight_version += 1


__all__ = ["IPCWeightSync"]
