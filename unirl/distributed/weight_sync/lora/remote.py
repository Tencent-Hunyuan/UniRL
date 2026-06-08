"""Cross-process LoRA weight-sync: rank-0 Ray push to non-sibling engines.

For any layout where the engine lives on a different Worker than the FSDP backend
— ``DiffusionTrainer`` separate slabs (one engine, possibly DP-replicated) and
HI3's two-engine trainer (AR + DiT on a disjoint GPU partition). Cross-slab
``HandleRef`` resolution raises, so the driver hands rank 0 the rollout Workers'
actor handles once (:meth:`set_rollout_targets`, like ``NCCLWeightSync``); then
:meth:`sync` (or the :meth:`extract` / :meth:`push` phases) gathers the adapter (a
train-mesh collective) and, on rank 0, pushes it to each engine via a plain Ray RPC
onto the engine's ``set_lora_from_tensors``. The adapter payload and the push
transport never leave the handler.

The extract gathers the full FSDP model (``state_dict()``), so a colocate trainer
whose engines share the cards (HI3) must split :meth:`extract` (run while the
engines are asleep, base onloaded) from :meth:`push` (run after offloading the base
and waking the engines); :meth:`sync` fuses both for separate slabs where there is
no contention. This handler does NO memory management — the trainer owns it.
"""

from __future__ import annotations

import logging
from typing import List

from unirl.distributed.group.dispatch import Dispatch, Execute, distributed
from unirl.distributed.weight_sync.lora.base import LoraWeightSyncBase

logger = logging.getLogger(__name__)


class RemoteLoraWeightSync(LoraWeightSyncBase):
    """Cross-process LoRA push to engine(s) that are NOT same-Worker siblings.

    ``copy=True`` routes the push through the engine's byte-copy receiver
    (``set_lora_from_tensors_copy``), required for TP>1 stages where the zero-copy
    handle's one-shot ``file_descriptor`` breaks the ``collective_rpc`` broadcast to
    ranks 2..N (HI3); leave it False for TP=1 engines (SD3 separate slabs).
    ``verify`` (``loaded_lora_checksums`` read-back) is vLLM-Omni-only.
    """

    def __init__(
        self,
        *,
        backend,
        param_prefix: str = "",
        adapter_name: str = "default",
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
        # Rollout engines' (role_name, [worker_handles]) pairs, cached on rank 0 by
        # the driver's set_rollout_targets() (plain Ray handles, NOT a HandleRef).
        self._targets: List[tuple] = []
        # Rank-0 hold of (lora_tensors, peft_config) between extract() and push() —
        # lets a memory-constrained trainer (HI3) gather while engines are asleep,
        # offload the base, wake the engines, then push.
        self._cached = None

    @distributed(dispatch_mode=Dispatch.BROADCAST, execute_mode=Execute.RANK_ZERO)
    def set_rollout_targets(self, targets: List[tuple]) -> None:
        """Rank 0 caches the rollout engines' ``(role_name, worker_handles)`` pairs.

        Handles are plain picklable Ray actor handles (NOT a cross-slab
        ``HandleRef``), so they survive the ``Worker.call`` arg path. One pair for a
        single-engine trainer (separate slabs); two for HI3 (AR + DiT). Mirrors
        ``NCCLWeightSync.set_rollout_targets``; only rank 0 pushes in ``push``.
        """
        self._targets = [(str(role), list(workers)) for role, workers in targets]

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def extract(self) -> None:
        """Gather the trained LoRA adapter and cache it on rank 0 (returns nothing).

        Runs on every train rank (``BROADCAST``): the extract is a train-mesh
        collective whose ``state_dict()`` gathers the full FSDP model to GPU, so a
        memory-constrained colocate trainer (HI3) MUST call this while its engines
        are asleep (base onloaded). The adapter stays inside the handler — cached on
        rank 0 for the matching :meth:`push`. Separate-slab trainers can just call
        :meth:`sync`.
        """
        self._extract_to_cache()

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def push(self) -> None:
        """Ship the adapter cached by :meth:`extract` to every rollout engine.

        Rank 0 only (ranks >= 1 no-op): pushes the (full, CPU) adapter to each engine
        Worker via a plain Ray RPC onto ``set_lora_from_tensors`` (or
        ``set_lora_from_tensors_copy`` when ``copy``). PRECONDITION: the engines are
        awake (the trainer wakes them after offloading the base).
        """
        self._push_from_cache()

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def sync(self) -> None:
        """:meth:`extract` + :meth:`push` in one dispatch — for the no-dance case.

        Use this when the engines can stay awake during the extract (separate slabs:
        train + rollout are on different GPUs, no memory contention). Colocate /
        memory-constrained trainers must split into :meth:`extract` / :meth:`push`
        around their own base-offload + engine-wake.
        """
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
        """Assert each rollout engine's loaded LoRA matches what we just pushed.

        Queried cross-slab via ``loaded_lora_checksums`` on each target Worker.
        vLLM-Omni-only (SGLang has no ``loaded_lora_checksums``).
        """
        import ray

        from unirl.rollout.engine.vllm_omni.weight_sync.ipc_dispatch import (
            DIFFRL_LORA_INT_ID,
        )

        exp_a, exp_b = self._expected_checksums(lora_tensors, peft_config)
        pending = [
            (role, worker.call.remote(role, "loaded_lora_checksums", (), {"adapter_id": int(DIFFRL_LORA_INT_ID)}))
            for role, workers in self._targets
            for worker in workers
        ]
        for role, ref in pending:
            self._assert_loaded(exp_a, exp_b, ray.get(ref), label=f"engine {role!r}")
        logger.info(
            "[LoRA-SYNC] rank 0: verify OK across %d engine(s) (%d lora_A / %d lora_B layers match)",
            len(self._targets),
            len(exp_a),
            len(exp_b),
        )


__all__ = ["RemoteLoraWeightSync"]
