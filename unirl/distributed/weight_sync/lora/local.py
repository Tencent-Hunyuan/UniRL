"""Colocate LoRA weight-sync: push the trained adapter into a same-Worker sibling engine, in-process."""

from __future__ import annotations

import logging
from typing import Optional

from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.distributed.weight_sync.lora.base import LoraWeightSyncBase

logger = logging.getLogger(__name__)


class LocalLoraWeightSync(LoraWeightSyncBase):
    """Push one track's trained FSDP LoRA adapter into a co-located rollout engine."""

    def __init__(
        self,
        *,
        backend,
        rollout,
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
        self._rollout = rollout
        self._copy = bool(copy)
        self._cached = None

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def extract(self) -> None:
        """Extract LoRA while the trainer is resident and cache it on CPU."""
        self._cached = self._extract()

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def push(self) -> None:
        """Load the cached LoRA after trainer offload and rollout wake-up."""
        if self._cached is None:
            raise RuntimeError("LocalLoraWeightSync.push: call extract() (or sync()) first")
        lora_tensors, peft_config = self._cached
        self._cached = None

        ri = self.rank_info
        rank = ri.rank if ri is not None else 0
        if ri is not None and ri.tp_rank != 0:  # extract on every rank, push only from the TP leader
            logger.debug(
                "[LoRA-SYNC] rank %s: extracted %d LoRA tensors, no push (tp=%s/%s, adapter=%s, track=%s)",
                rank,
                len(lora_tensors),
                ri.tp_rank,
                ri.tp_size,
                self._adapter_name,
                self._track_prefix or "<single>",
            )
            return

        # A grouped vLLM-Omni replica has one controller Remote plus TP/SP
        # follower Remotes. The controller broadcasts the adapter to all of its
        # subprocesses; followers deliberately have no backend of their own.
        if not getattr(self._rollout, "_is_replica_head", True):
            return

        setter = self._rollout.set_lora_from_tensors_copy if self._copy else self._rollout.set_lora_from_tensors
        setter(self._adapter_name, lora_tensors, peft_config=peft_config)
        logger.info(
            "[LoRA-SYNC] rank %s: pushed %d LoRA tensors to rollout via %s (adapter=%s, track=%s)",
            rank,
            len(lora_tensors),
            "copy" if self._copy else "handle",
            self._adapter_name,
            self._track_prefix or "<single>",
        )
        if self._verify:
            self._verify_loaded(lora_tensors, peft_config)

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def sync(self) -> None:
        """Extract and push in one call when trainer and rollout can coexist."""
        self.extract()
        self.push()

    def _verify_loaded(self, lora_tensors, peft_config) -> None:
        """Assert the sibling engine's loaded LoRA matches what we just pushed."""
        from unirl.distributed.weight_sync.transfer.ipc_dispatch import (
            DIFFRL_LORA_INT_ID,
        )

        exp_a, exp_b = self._expected_checksums(lora_tensors, peft_config)
        topology = self._rollout.tp_per_stage()
        loaded = self._rollout.loaded_lora_checksums(adapter_id=int(DIFFRL_LORA_INT_ID))
        rank = self.rank_info.rank if self.rank_info is not None else 0
        self._assert_loaded(
            exp_a,
            exp_b,
            loaded,
            topology=topology,
            label=f"train-rank {rank} rollout",
        )
        logger.info(
            "[LoRA-SYNC] rank %s: verify OK (%d lora_A / %d lora_B layers match)",
            rank,
            len(exp_a),
            len(exp_b),
        )


__all__ = ["LocalLoraWeightSync"]
