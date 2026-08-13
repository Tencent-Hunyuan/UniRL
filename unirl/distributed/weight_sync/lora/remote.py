"""Cross-process LoRA weight-sync: rank-0 Ray push to non-sibling engines."""

from __future__ import annotations

import logging
from typing import List, Optional

from unirl.distributed.group.dispatch import Dispatch, Execute, distributed
from unirl.distributed.weight_sync.lora.base import LoraWeightSyncBase

logger = logging.getLogger(__name__)


class RemoteLoraWeightSync(LoraWeightSyncBase):
    """Cross-process LoRA push to engine(s) that are NOT same-Worker siblings."""

    def __init__(
        self,
        *,
        backend,
        param_prefix: str = "",
        adapter_name: Optional[str] = None,
        verify: bool = False,
        track_prefix: str = "",
        copy: bool = False,
    ) -> None:
        super().__init__(
            backend=backend,
            param_prefix=param_prefix,
            adapter_name=adapter_name,
            verify=verify,
            track_prefix=track_prefix,
        )
        self._copy = bool(copy)
        self._targets: List[tuple] = []
        self._cached = None

    @distributed(dispatch_mode=Dispatch.BROADCAST, execute_mode=Execute.RANK_ZERO)
    def set_rollout_targets(self, targets: List[tuple]) -> None:
        """Rank 0 caches the rollout engines' ``(role_name, worker_handles)`` pairs."""
        self._targets = [(str(role), list(workers)) for role, workers in targets]

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def extract(self) -> None:
        """Gather the trained LoRA adapter and cache it on rank 0 (returns nothing)."""
        self._extract_to_cache()

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def push(self) -> None:
        """Ship the adapter cached by :meth:`extract` to every rollout engine."""
        self._push_from_cache()

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def sync(self) -> None:
        """:meth:`extract` + :meth:`push` in one dispatch — for the no-dance case."""
        self._extract_to_cache()
        self._push_from_cache()

    def _extract_to_cache(self) -> None:
        """Collective gather on every rank; rank 0 caches ``(tensors, peft_config)``."""
        lora_tensors, peft_config = self._extract()
        rank = self.rank_info.rank if self.rank_info is not None else 0
        if rank == 0:
            self._cached = (lora_tensors, peft_config)

    def _push_from_cache(self) -> None:
        """Rank 0 ships the cached adapter to the targets, then clears the cache."""
        rank = self.rank_info.rank if self.rank_info is not None else 0
        if rank != 0:
            return
        if self._cached is None:
            raise RuntimeError("RemoteLoraWeightSync.push: call extract() (or sync()) first")
        if not self._targets:
            raise RuntimeError("RemoteLoraWeightSync.push: call set_rollout_targets() first")
        lora_tensors, peft_config = self._cached
        self._cached = None

        import ray

        method = "set_lora_from_tensors_copy" if self._copy else "set_lora_from_tensors"
        refs = [
            worker.call.remote(role, method, (self._adapter_name, lora_tensors), {"peft_config": peft_config})
            for role, workers in self._targets
            for worker in workers
        ]
        ray.get(refs)
        logger.info(
            "[LoRA-SYNC] rank 0: pushed %d LoRA tensors to %d engine(s) via %s (adapter=%s, track=%s)",
            len(lora_tensors),
            len(self._targets),
            method,
            self._adapter_name,
            self._track_prefix or "<single>",
        )
        if self._verify:
            self._verify_loaded(lora_tensors, peft_config)

    def _verify_loaded(self, lora_tensors, peft_config) -> None:
        """Assert each rollout engine's loaded LoRA matches what we just pushed."""
        import ray

        from unirl.distributed.weight_sync.transfer.ipc_dispatch import (
            DIFFRL_LORA_INT_ID,
        )

        exp_a, exp_b = self._expected_checksums(lora_tensors, peft_config)
        pending = [
            (
                role,
                worker.call.remote(role, "tp_per_stage", (), {}),
                worker.call.remote(role, "loaded_lora_checksums", (), {"adapter_id": int(DIFFRL_LORA_INT_ID)}),
            )
            for role, workers in self._targets
            for worker in workers
        ]
        for role, topology_ref, loaded_ref in pending:
            topology, loaded = ray.get([topology_ref, loaded_ref])
            self._assert_loaded(
                exp_a,
                exp_b,
                loaded,
                topology=topology,
                label=f"engine {role!r}",
            )
        logger.info(
            "[LoRA-SYNC] rank 0: verify OK across %d engine(s) (%d lora_A / %d lora_B layers match)",
            len(self._targets),
            len(exp_a),
            len(exp_b),
        )


__all__ = ["RemoteLoraWeightSync"]
