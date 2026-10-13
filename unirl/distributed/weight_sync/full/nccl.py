"""v2 full-weight NCCL sync (SEPARATE slabs, cross-node capable)."""

from __future__ import annotations

import math
import time
from datetime import timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
        operation_timeout_s: float = 300.0,
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
        if not isinstance(group_name, str) or not group_name:
            raise TypeError(f"group_name must be a non-empty string, got {group_name!r}")
        self._group_name = group_name
        self._model_update_group = None  # set on rank 0 in connect()
        self._rollout_targets: List[Any] = []  # rollout Worker actor handles (rank 0 only)
        self._rollout_role: Optional[str] = None
        self._transactional = False
        self._connected = False
        self._broken = False
        if isinstance(operation_timeout_s, bool) or not isinstance(operation_timeout_s, (int, float)):
            raise TypeError(f"operation_timeout_s must be numeric, got {operation_timeout_s!r}")
        self._operation_timeout_s = float(operation_timeout_s)
        if not math.isfinite(self._operation_timeout_s) or self._operation_timeout_s <= 0:
            raise ValueError(f"operation_timeout_s must be finite and > 0, got {operation_timeout_s!r}")

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
    def set_rollout_targets(self, actor_handles: List[Any], role_name: str, transactional: bool = False) -> None:
        """Rank 0 caches the rollout slab's Worker actor handles + role name."""
        if not actor_handles:
            raise ValueError("NCCLWeightSync requires at least one rollout target.")
        if not isinstance(role_name, str) or not role_name:
            raise TypeError(f"role_name must be a non-empty string, got {role_name!r}")
        if not isinstance(transactional, bool):
            raise TypeError(f"transactional must be a bool, got {transactional!r}")
        self._rollout_targets = list(actor_handles)
        self._rollout_role = role_name
        self._transactional = transactional

    @distributed(dispatch_mode=Dispatch.BROADCAST, execute_mode=Execute.RANK_ZERO)
    def connect(
        self, *, master_addr: str, master_port: int, num_rollout_gpus: int, tp_size: int = 1, pp_size: int = 1
    ) -> None:
        """Bring up the broadcast group (rank 0 + all rollout engine GPUs)."""
        import torch

        from unirl.utils.distributed_utils import eager_connect_process_group, init_process_group

        if self._rollout_role is None:
            raise RuntimeError("NCCLWeightSync.connect: call set_rollout_targets() first")
        if self._connected:
            raise RuntimeError("NCCLWeightSync.connect called more than once without shutdown.")
        if not isinstance(master_addr, str) or not master_addr:
            raise TypeError(f"master_addr must be a non-empty string, got {master_addr!r}")
        if isinstance(master_port, bool) or not isinstance(master_port, int) or not 1 <= master_port <= 65535:
            raise ValueError(f"master_port must be an integer in [1, 65535], got {master_port!r}")
        for name, value in (
            ("num_rollout_gpus", num_rollout_gpus),
            ("tp_size", tp_size),
            ("pp_size", pp_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if pp_size != 1:
            raise NotImplementedError(
                "NCCLWeightSync.connect: rollout pp_size>1 is not implemented "
                f"(got pp_size={pp_size}); only tp_size/dp_size are supported."
            )
        expected_gpus = len(self._rollout_targets) * tp_size
        if num_rollout_gpus != expected_gpus:
            raise ValueError(
                f"num_rollout_gpus={num_rollout_gpus} must equal "
                f"len(rollout_targets)({len(self._rollout_targets)}) * tp_size({tp_size})={expected_gpus}."
            )

        world = num_rollout_gpus + 1
        refs = [
            handle.call.remote(
                self._rollout_role,
                "init_weights_update_group",
                (),
                {
                    "master_address": master_addr,
                    "master_port": master_port,
                    "rank_offset": i * tp_size + 1,
                    "world_size": world,
                    "group_name": self._group_name,
                    "backend": "nccl",
                    "track_prefix": self._track_prefix,
                    **({"timeout_s": self._operation_timeout_s} if self._transactional else {}),
                },
            )
            for i, handle in enumerate(self._rollout_targets)
        ]
        try:
            self._model_update_group = init_process_group(
                backend="nccl",
                init_method=f"tcp://{master_addr}:{master_port}",
                world_size=world,
                rank=0,
                group_name=self._group_name,
                timeout=timedelta(seconds=self._operation_timeout_s),
            )
            eager_connect_process_group(
                self._model_update_group,
                torch.device("cuda", torch.cuda.current_device()),
            )
            self._collect_rollout_refs(refs, phase="connect")
        except BaseException:
            self._broken = True
            self._abort_model_group()
            raise
        self._connected = True
        self._broken = False

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def sync(self) -> None:
        """Broadcast the current full weights into the rollout engines."""
        import torch.distributed as dist

        is_rank0 = self._my_rank == 0
        setup_error = None
        if is_rank0:
            if not self._connected:
                setup_error = "NCCLWeightSync.sync called before connect()."
            elif self._broken:
                setup_error = "NCCLWeightSync process group is broken; rebuild the sync actor before retrying."
        self._raise_if_rank0_phase_failed(setup_error, phase="preflight")

        begin_error = None
        if is_rank0 and self._transactional:
            try:
                self._invoke_rollouts(
                    "begin_weights_update",
                    {
                        "group_name": self._group_name,
                        "track_prefix": self._track_prefix,
                    },
                )
            except BaseException as exc:
                begin_error = f"{type(exc).__name__}: {exc}"
        self._raise_if_rank0_phase_failed(begin_error, phase="begin publication")

        for bucket, is_last in self._iter_buckets():
            names = [n for n, _ in bucket]
            dtypes = [str(t.dtype) for _, t in bucket]
            shapes = [list(t.shape) for _, t in bucket]

            prepare_error = None
            if is_rank0 and self._transactional:
                try:
                    self._invoke_rollouts(
                        "prepare_weights_update",
                        {
                            "names": names,
                            "dtypes": dtypes,
                            "shapes": shapes,
                            "group_name": self._group_name,
                            "track_prefix": self._track_prefix,
                        },
                    )
                except BaseException as exc:
                    self._abort_model_group()
                    prepare_error = f"{type(exc).__name__}: {exc}"
            self._raise_if_rank0_phase_failed(prepare_error, phase="prepare bucket")

            update_error = None
            if is_rank0:
                refs: List[Any] = []
                try:
                    refs = self._launch_rollouts(
                        "update_weights_from_distributed",
                        {
                            "names": names,
                            "dtypes": dtypes,
                            "shapes": shapes,
                            "group_name": self._group_name,
                            "flush_cache": (self._flush_cache and is_last),
                            "track_prefix": self._track_prefix,
                        },
                    )
                    works = [
                        dist.broadcast(
                            tensor.data.contiguous(),
                            0,
                            group=self._model_update_group,
                            async_op=True,
                        )
                        for _, tensor in bucket
                    ]
                    timeout = timedelta(seconds=self._operation_timeout_s)
                    for work in works:
                        if not work.wait(timeout=timeout):
                            raise TimeoutError("NCCLWeightSync timed out waiting for a source broadcast.")
                    self._collect_rollout_refs(refs, phase="receive bucket")
                except BaseException as exc:
                    self._cancel_refs(refs)
                    self._abort_model_group()
                    update_error = f"{type(exc).__name__}: {exc}"
            self._raise_if_rank0_phase_failed(update_error, phase="broadcast bucket")

        finish_error = None
        if is_rank0 and self._transactional:
            try:
                self._invoke_rollouts(
                    "finish_weights_update",
                    {
                        "group_name": self._group_name,
                        "track_prefix": self._track_prefix,
                    },
                )
            except BaseException as exc:
                finish_error = f"{type(exc).__name__}: {exc}"
        self._raise_if_rank0_phase_failed(finish_error, phase="finish publication")

    def _launch_rollouts(self, method: str, kwargs: Dict[str, Any]) -> List[Any]:
        return [
            handle.call.remote(
                self._rollout_role,
                method,
                (),
                kwargs,
            )
            for handle in self._rollout_targets
        ]

    def _invoke_rollouts(self, method: str, kwargs: Dict[str, Any]) -> None:
        self._collect_rollout_refs(self._launch_rollouts(method, kwargs), phase=method)

    def _collect_rollout_refs(self, refs: Sequence[Any], *, phase: str) -> None:
        import ray

        pending = list(refs)
        deadline = time.monotonic() + self._operation_timeout_s
        while pending:
            ready, pending = ray.wait(
                pending,
                num_returns=1,
                timeout=max(deadline - time.monotonic(), 0.0),
            )
            if not ready:
                self._cancel_refs(pending)
                raise TimeoutError(
                    f"NCCLWeightSync timed out after {self._operation_timeout_s}s "
                    f"waiting for {len(pending)} rollout receiver(s) during {phase}."
                )
            try:
                ray.get(ready[0])
            except BaseException:
                self._cancel_refs(pending)
                raise

    @staticmethod
    def _cancel_refs(refs: Sequence[Any]) -> None:
        import ray

        for ref in refs:
            try:
                ray.cancel(ref, force=False)
            except Exception:
                pass

    def _raise_if_rank0_phase_failed(self, error: Optional[str], *, phase: str) -> None:
        import torch.distributed as dist

        local = {"rank": self._my_rank, "error": error}
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, local)
            errors = [item for item in gathered if item is not None and item.get("error")]
        else:
            errors = [local] if error is not None else []
        if errors:
            self._broken = True
            first = errors[0]
            raise RuntimeError(f"NCCLWeightSync {phase} failed on rank {first['rank']}: {first['error']}")

    def _abort_model_group(self) -> None:
        abort = getattr(self._model_update_group, "abort", None)
        if callable(abort):
            try:
                abort()
            except Exception:
                pass

    @distributed(dispatch_mode=Dispatch.BROADCAST, execute_mode=Execute.RANK_ZERO)
    def cleanup(self) -> None:
        """Tear down NCCL groups during the trainer's normal cleanup lifecycle."""
        self._teardown_groups()

    def shutdown(self) -> None:
        """Tear down NCCL groups if the worker shuts down directly."""
        self._teardown_groups()

    def _teardown_groups(self) -> None:
        """Destroy receiver and sender groups; retain failed handles for diagnostics."""
        import torch.distributed as dist

        first_error: Optional[BaseException] = None
        receiver_refs: List[Any] = []
        if self._rollout_targets and self._rollout_role is not None:
            receiver_refs = self._launch_rollouts(
                "destroy_weights_update_group",
                {
                    "group_name": self._group_name,
                    "track_prefix": self._track_prefix,
                },
            )
        if self._model_update_group is not None:
            try:
                dist.destroy_process_group(self._model_update_group)
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
            else:
                self._model_update_group = None
        try:
            self._collect_rollout_refs(receiver_refs, phase="destroy process group")
        except BaseException as exc:
            if first_error is None:
                first_error = exc
        if self._model_update_group is None:
            self._connected = False
        if first_error is not None:
            raise first_error


__all__ = ["NCCLWeightSync"]
