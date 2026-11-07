"""Colocate LoRA weight-sync: push the trained adapter into a same-Worker sibling engine, in-process."""

from __future__ import annotations

import logging
from typing import Any

from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.distributed.weight_sync.lora.base import LoraWeightSyncBase

logger = logging.getLogger(__name__)


class LocalLoraWeightSync(LoraWeightSyncBase):
    """Push one track's trained FSDP LoRA adapter into a co-located rollout engine."""

    def __init__(self, *, backend: Any, rollout: Any, **kwargs: Any) -> None:
        super().__init__(backend=backend, **kwargs)
        self._rollout = rollout

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def extract(self) -> None:
        """Read the adapter out of the FSDP model and cache it (returns nothing)."""
        self._cached = self._extract()

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def push(self) -> None:
        """Load the adapter cached by :meth:`extract` into the engine."""
        self._push_from_cache()

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def sync(self) -> None:
        """:meth:`extract` + :meth:`push` in one dispatch — for the no-dance case."""
        self._cached = self._extract()
        self._push_from_cache()

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
        rank = self._my_rank
        logger.info(
            "[LoRA-SYNC] rank %s: dispatched %d LoRA tensors to local rollout receiver (adapter=%s, track=%s)",
            rank,
            len(lora_tensors),
            self._adapter_name,
            self._track_prefix or "<single>",
        )
        if self._verify:
            from unirl.distributed.weight_sync.transfer.ipc_dispatch import UNIRL_LORA_INT_ID

            exp_a, exp_b = self._expected_checksums(lora_tensors, peft_config)
            topology = self._rollout.tp_per_stage()
            loaded = self._rollout.loaded_lora_checksums(adapter_id=UNIRL_LORA_INT_ID)
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
