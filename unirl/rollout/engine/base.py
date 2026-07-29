"""Rollout engine base class for the ``Sample`` → ``Sample`` path.

Concrete engines take all runtime deps as ``__init__`` kwargs and complete
construction in one shot — no separate ``initialize(device)`` step. After
``__init__`` returns the engine is fully usable: model loaded, worker
subprocesses spawned, dist groups brought up. This matches the actor flow where
``_setup_distributed_env`` runs before the engine is built.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import torch

from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.distributed.group.remote import Remote
from unirl.types.sample import Sample

#: The rollout batch contract (LIN-522). Single-turn engines return one ``Sample``;
#: the agentic engine returns a list of variable-depth trajectory ``Sample``s (one
#: per rollout). The broad generation method uses this union; the single-turn
#: subclass refines its return to ``Sample``.
RolloutOutput = Union[Sample, List[Sample]]


class BaseEngineConfig(ABC):
    """Marker base for all rollout engine config dataclasses.

    Used as the type annotation / base class for the engine config dataclasses.
    Each concrete engine config maps itself to its runtime engine class via
    :meth:`make_engine`.
    """

    def make_engine(self, **deps: Any) -> "BaseRolloutEngine":
        """Construct the runtime engine declared by this config.

        ``deps`` carry the runtime injections (``device``, ``strategy``,
        ``rank``, ``model_config``); the engine ctor contract is uniformly
        ``Engine(config=self, **deps)``. Subclasses override to import (lazily,
        so config modules stay importable without the engine's heavy optional
        deps) and return their engine class.
        """
        raise NotImplementedError(f"{type(self).__name__} must implement make_engine()")


class BaseRolloutEngine(Remote, ABC):
    """Rollout engine ABC. One-shot construction; new types only."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def shutdown(self) -> None:
        """Release worker subprocesses and any other engine-owned resources."""

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def sleep(self) -> None:
        """Best-effort runtime offload. Default no-op.

        Decorated so the driver-side ``Handle.sleep()`` dispatches to every
        worker. Subclasses that override should re-apply ``@distributed``
        on their override (Handle's method-binding sees the subclass's
        attribute and won't pick up a base-class decorator alone).
        """

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def wake_up(self) -> None:
        """Restore runtime resources after ``sleep``. Default no-op.

        Same dispatch contract as :meth:`sleep`; see its docstring.
        """

    def onload_weights(self, *, track_prefix: str = "") -> None:
        """Restore the resources needed to receive a weight update."""
        del track_prefix
        self.wake_up()

    @property
    def is_offloaded(self) -> bool:
        """Whether the engine has released its runtime resources."""
        return False

    def health_check(self) -> bool:
        """Return True iff the engine is ready to serve a generate call."""
        return True

    def get_memory_info(self) -> Dict[str, float]:
        """Per-engine GPU memory snapshot. Default reads CUDA totals."""
        if not torch.cuda.is_available():
            return {}
        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "cached_gb": torch.cuda.memory_reserved() / 1e9,
        }

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @abstractmethod
    def generate(self, sample: Sample) -> RolloutOutput:
        """Synchronously run rollout generation for one request batch.

        Dispatch belongs to each concrete contract: single-turn engines use
        ``DP_SCATTER`` and return one ``Sample``; the agentic coordinator uses
        ``BROADCAST + RANK_ZERO`` and returns a trajectory list.
        """

    # ------------------------------------------------------------------
    # Control plane — sync methods reached via the raw ``Worker.call`` RPC (the
    # un-decorated weight-sync pattern), so they interleave with an in-flight
    # ``generate`` on a threaded Worker (``worker_max_concurrency>1``).
    # ------------------------------------------------------------------

    def abort(self, ids: Optional[List[str]] = None) -> List[Sample]:
        """Best-effort cancel of in-flight generation; return any partials.

        Default no-op (``[]``). Engines whose backend supports it (SGLang) cancel
        running requests; sync/batch backends can only drop not-yet-started work.
        """
        del ids
        return []

    def pause(self) -> None:
        """Stop admitting new generation (best-effort). Default no-op."""

    def resume(self) -> None:
        """Resume generation after :meth:`pause`. Default no-op."""

    # ------------------------------------------------------------------
    # Weight sync — bucketed CUDA-IPC (verl-omni pattern)
    # ------------------------------------------------------------------

    def update_weights_from_ipc(
        self,
        *,
        peft_config: Optional[dict] = None,
        base_sync_done: bool = False,
        use_shm: bool = False,
        track_prefix: str = "",
    ) -> None:
        """Receive a state dict over a per-rank ZMQ + CUDA-IPC channel."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Weight sync — NCCL broadcast
    # ------------------------------------------------------------------

    def init_weights_update_group(
        self,
        *,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str = "nccl",
        track_prefix: str = "",
    ) -> None:
        """Bring up a trainer-rollout NCCL process group on the engine side."""
        raise NotImplementedError

    def update_weights_from_distributed(
        self,
        *,
        names: List[str],
        dtypes: List[str],
        shapes: List[List[int]],
        group_name: str,
        target_modules: Optional[List[str]] = None,
        flush_cache: bool = True,
        track_prefix: str = "",
    ) -> None:
        """Receive a bucket of weights via the established NCCL group."""
        raise NotImplementedError

    def destroy_weights_update_group(
        self,
        *,
        group_name: str,
        track_prefix: str = "",
    ) -> None:
        """Tear down a previously-initialized NCCL update group."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Weight sync — LoRA tensor bag
    # ------------------------------------------------------------------

    def set_lora_from_tensors(
        self,
        adapter_name: str,
        lora_tensors: Dict[str, torch.Tensor],
        *,
        peft_config: Optional[dict] = None,
    ) -> None:
        """Load a LoRA adapter directly from in-memory tensors."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Weight sync — SGLang-shape one-bag tensor payload
    # ------------------------------------------------------------------

    def update_weights_from_tensor(
        self,
        *,
        serialized_named_tensors: List[str],
        target_modules: Optional[List[str]] = None,
        load_format: Optional[str] = None,
        flush_cache: bool = True,
        track_prefix: str = "",
    ) -> None:
        """Receive a state-dict packed as a single SGLang-shape payload per TP rank."""
        del track_prefix
        raise NotImplementedError


class BaseSingleTurnRolloutEngine(BaseRolloutEngine, ABC):
    """Nominal contract for engines that fill and return one ``Sample``.

    The class intentionally does not prescribe batching or provide a batching
    wrapper. Each engine owns those semantics and applies its own
    ``@distributed`` decorator. The contract is synchronous. An engine meant to
    serve as an agentic inner must make ``generate`` safe for CONCURRENT
    callers (the agentic drain calls it from one thread per trajectory and
    relies on the backend batching the in-flight requests together); an engine
    that cannot serve concurrently serializes internally instead.
    """

    #: Policy weight version the current weights correspond to (bumped on each
    #: weight sync; stamped onto generated Parts by :meth:`_stamp_weight_version`).
    _weight_version: int = 0

    @abstractmethod
    def generate(self, sample: Sample) -> Sample:
        """Synchronously fill and return one request ``Sample``."""

    def _stamp_weight_version(self, sample: Sample) -> Sample:
        """Stamp ``self._weight_version`` onto the frontier (last) gen Part."""
        v = getattr(self, "_weight_version", None)
        if v is None or not sample.parts:
            return sample
        gen = sample.parts[-1].fill(weight_version=int(v))
        return sample.with_parts([*sample.parts[:-1], gen])


__all__ = ["BaseRolloutEngine", "BaseSingleTurnRolloutEngine", "RolloutOutput"]
