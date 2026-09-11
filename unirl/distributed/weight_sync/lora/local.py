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
    ) -> None:
        super().__init__(
            backend=backend,
            param_prefix=param_prefix,
            adapter_name=adapter_name,
            verify=verify,
            track_prefix=track_prefix,
        )
        self._rollout = rollout
        self._cached = None

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def extract(self) -> None:
        """Read the adapter out of the FSDP model and cache it (returns nothing)."""
        self._extract_to_cache()

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def push(self) -> None:
        """Load the adapter cached by :meth:`extract` into the engine."""
        self._push_from_cache()

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def sync(self) -> None:
        """:meth:`extract` + :meth:`push` in one dispatch — for the no-dance case."""
        self._extract_to_cache()
        self._push_from_cache()

    def _extract_to_cache(self) -> None:
        """Cache ``(tensors, peft_config)``; the read needs the trainer resident."""
        self._cached = self._extract()

    def _push_from_cache(self) -> None:
        """Load the cached adapter into the engine, keeping the cache for reuse."""
        if self._cached is None:
            raise RuntimeError("LocalLoraWeightSync.push: call extract() (or sync()) first")
        # The cache outlives the push on purpose. The adapter only changes when
        # the optimizer steps, so a caller that pushes again before the next step
        # -- an accumulate window, or an eval's later chunks -- can reuse it
        # instead of onloading the trainer just to read identical weights.
        lora_tensors, peft_config = self._cached
        self._rollout.set_lora_from_tensors(self._adapter_name, lora_tensors, peft_config=peft_config)
        rank = self.rank_info.rank if self.rank_info is not None else 0
        logger.info(
            "[LoRA-SYNC] rank %s: dispatched %d LoRA tensors to local rollout receiver (adapter=%s, track=%s)",
            rank,
            len(lora_tensors),
            self._adapter_name,
            self._track_prefix or "<single>",
        )
        if self._verify:
            self._verify_loaded(lora_tensors, peft_config)

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
