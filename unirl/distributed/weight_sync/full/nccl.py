"""v2 full-weight NCCL sync (SEPARATE slabs, cross-node capable).

``NCCLWeightSync`` lives on the TRAIN slab as a sibling of the FSDP ``backend``
ONLY — the rollout engine is on a different device slab, so it is NOT a sibling
and cannot be injected (cross-slab ``HandleRef`` resolution raises). Instead the
driver hands rank 0 the rollout slab's Worker actor handles once
(``set_rollout_targets``); rank 0 then self-drives the rollout side via
non-blocking ``handle.call.remote(...)`` + ``ray.get`` from inside its own
Worker. This provides the concurrency the NCCL rendezvous barrier needs without
threads (train rank 0 and the rollout workers are distinct processes).

Group layout: train rank 0 is group rank 0; rollout engine ``i`` joins at
``rank_offset = i*tp_size + 1`` and its SGLang scheduler subprocesses compute
``global_rank = rank_offset + tp_rank`` internally. With ``tp_size=1`` this is
the historical ``rank_offset = i + 1``. Other train ranks are NOT in the
broadcast group — they participate only in the train-mesh all-gather that
``raw_state_dict`` performs (so rank 0 sees full tensors), then discard.

Rollout TP: the driver passes only the tp_rank==0 worker of each engine as a
target (and ``num_rollout_gpus = num_engines*tp_size``). ``dist.broadcast``
delivers the FULL tensor to every group rank; each SGLang TP rank's own
``weight_loader`` reshards it — so ``sync()`` is unchanged, only ``connect``'s
rank_offset/world_size scale with ``tp_size``.

Driver wiring (in the trainer, once both slabs exist; engine workers alive)::

    addr, port = ws.pick_master()[0]
    tp = rollout.tp_size
    targets = [w for w, ri in zip(rollout.workers, rollout.rank_infos) if ri.tp_rank == 0]
    ws.set_rollout_targets(targets, rollout.role_name)
    ws.connect(master_addr=addr, master_port=port, num_rollout_gpus=len(targets)*tp, tp_size=tp)
    ...
    ws.sync()   # every weight_sync_interval

Scope: single-/multi-node, rollout TP≥1, single-stage. PP>1 needs a per-stage
rank_offset map and is out of scope (raise upstream). Torch/ray imports are
deferred so the driver can import this module for ``remote(...)``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from unirl.distributed.group.dispatch import Dispatch, Execute, distributed
from unirl.distributed.weight_sync.full.base import FullWeightSync


class NCCLWeightSync(FullWeightSync):
    """Separate-slab full-weight sync: rank 0 broadcasts to all rollout GPUs."""

    def __init__(
        self,
        *,
        backend: Any,
        group_name: str = "weight_sync",
        bucket_size_mb: int = 512,
        flush_cache: bool = True,
        lora_merged: bool = False,
        adapter_name: Optional[str] = None,
        name_remap: Optional[Dict[str, Optional[str]]] = None,
        track_prefix: str = "",
        wire_dtype: Any = None,
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
        self._group_name = str(group_name)
        self._model_update_group = None  # set on rank 0 in connect()
        self._rollout_targets: List[Any] = []  # rollout Worker actor handles (rank 0 only)
        self._rollout_role: Optional[str] = None

    # ------------------------------------------------------------------
    # One-time setup (driver-called)
    # ------------------------------------------------------------------

    @distributed(dispatch_mode=Dispatch.BROADCAST, execute_mode=Execute.RANK_ZERO)
    def pick_master(self) -> Tuple[str, int]:
        """Rank 0 returns its ``(node_ip, free_port)`` for the rendezvous."""
        import socket

        import ray

        addr = ray._private.services.get_node_ip_address()
        with socket.socket() as sock:
            sock.bind(("", 0))
            port = sock.getsockname()[1]
        return addr, int(port)

    @distributed(dispatch_mode=Dispatch.BROADCAST, execute_mode=Execute.RANK_ZERO)
    def set_rollout_targets(self, actor_handles: List[Any], role_name: str) -> None:
        """Rank 0 caches the rollout slab's Worker actor handles + role name.

        Handles are plain picklable Ray actor handles (NOT a cross-slab
        ``HandleRef``), so they survive the ``Worker.call`` arg path.
        """
        self._rollout_targets = list(actor_handles)
        self._rollout_role = str(role_name)

    @distributed(dispatch_mode=Dispatch.BROADCAST, execute_mode=Execute.RANK_ZERO)
    def connect(
        self, *, master_addr: str, master_port: int, num_rollout_gpus: int, tp_size: int = 1, pp_size: int = 1
    ) -> None:
        """Bring up the broadcast group (rank 0 + all rollout engine GPUs).

        Fires each rollout worker's ``init_weights_update_group`` NON-BLOCKING
        first, then joins as group rank 0 (which blocks on the barrier), then
        awaits the rollout joins. The non-blocking fire is what lets the
        rendezvous complete — no thread needed (distinct processes).

        Rollout TP: each ``_rollout_targets`` entry is the tp_rank==0 worker of
        one SGLang engine, and SGLang internally assigns ``rank_offset +
        tp_rank`` to its ``tp_size`` scheduler subprocesses. So engine ``i``
        occupies the contiguous NCCL block ``[i*tp_size+1, (i+1)*tp_size]`` and
        the group world size is ``num_engines*tp_size + 1``. ``tp_size=1``
        reproduces the historical ``rank_offset = i+1`` layout exactly.
        """
        import ray

        from unirl.utils.distributed_utils import init_process_group

        if self._rollout_role is None:
            raise RuntimeError("NCCLWeightSync.connect: call set_rollout_targets() first")

        # PP>1 needs a per-stage rank_offset map (each PP stage holds a disjoint
        # slice of the model). Fail closed until that lands; the tp/dp path is
        # the supported one today.
        if pp_size > 1:
            raise NotImplementedError(
                "NCCLWeightSync.connect: rollout pp_size>1 is not implemented "
                f"(got pp_size={pp_size}); only tp_size/dp_size are supported."
            )

        tp = max(1, int(tp_size))
        world = int(num_rollout_gpus) + 1
        refs = [
            handle.call.remote(
                self._rollout_role,
                "init_weights_update_group",
                (),
                {
                    "master_address": master_addr,
                    "master_port": int(master_port),
                    "rank_offset": i * tp + 1,
                    "world_size": world,
                    "group_name": self._group_name,
                    "backend": "nccl",
                    "track_prefix": self._track_prefix,
                },
            )
            for i, handle in enumerate(self._rollout_targets)
        ]
        self._model_update_group = init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_addr}:{int(master_port)}",
            world_size=world,
            rank=0,
            group_name=self._group_name,
        )
        ray.get(refs)

    # ------------------------------------------------------------------
    # Per-step sync
    # ------------------------------------------------------------------

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def sync(self) -> None:
        """Broadcast the current full weights into the rollout engines.

        Every train rank runs the identical bucket loop (lockstep all-gather in
        ``raw_state_dict``). Rank 0, per bucket: tells every rollout worker to
        post matching recvs (non-blocking), broadcasts each tensor, then awaits
        the recvs. Ranks >= 1 just consume the generator (their half of the
        all-gather) and discard.
        """
        import ray
        import torch.distributed as dist

        is_rank0 = self._my_rank == 0
        for bucket, is_last in self._iter_buckets():
            if not is_rank0:
                continue  # ranks >= 1 only drive the train-mesh all-gather
            names = [n for n, _ in bucket]
            dtypes = [str(t.dtype) for _, t in bucket]
            shapes = [list(t.shape) for _, t in bucket]
            recv_refs = [
                handle.call.remote(
                    self._rollout_role,
                    "update_weights_from_distributed",
                    (),
                    {
                        "names": names,
                        "dtypes": dtypes,
                        "shapes": shapes,
                        "group_name": self._group_name,
                        "flush_cache": (self._flush_cache and is_last),
                        "track_prefix": self._track_prefix,
                    },
                )
                for handle in self._rollout_targets
            ]
            for _, tensor in bucket:
                dist.broadcast(tensor.data.contiguous(), 0, group=self._model_update_group)
            ray.get(recv_refs)
        self.weight_version += 1


__all__ = ["NCCLWeightSync"]
