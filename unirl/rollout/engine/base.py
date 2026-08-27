"""Rollout engine contracts. Engines complete construction in ``__init__``; no separate initialize step."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch

from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.distributed.group.remote import Remote
from unirl.types.sample import Sample


class BaseEngineConfig(ABC):
    """Marker base for rollout engine config dataclasses."""

    def make_engine(self, **deps: Any) -> "BaseRolloutEngine":
        """Construct the runtime engine declared by this config; ctor contract is ``Engine(config=self, **deps)``."""
        raise NotImplementedError(f"{type(self).__name__} must implement make_engine()")


class BaseRolloutEngine(Remote, ABC):
    """Rollout engine ABC: fill and return one ``Sample``; ``generate`` may be called concurrently."""

    @abstractmethod
    def shutdown(self) -> None:
        """Release worker subprocesses and any other engine-owned resources."""

    # Overrides of sleep/wake_up must re-apply @distributed; Handle binds the subclass attribute only.
    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def sleep(self) -> None:
        """Best-effort runtime offload. Default no-op."""

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def wake_up(self) -> None:
        """Restore runtime resources after ``sleep``. Default no-op."""

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def set_stopping(self, stopping: bool = True) -> None:
        """Arm cooperative rollout suspension. Default no-op."""
        del stopping

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
        """Per-engine GPU memory snapshot."""
        if not torch.cuda.is_available():
            return {}
        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "cached_gb": torch.cuda.memory_reserved() / 1e9,
        }

    @abstractmethod
    def generate(self, sample: Sample) -> Sample:
        """Synchronously fill and return one request ``Sample``; each concrete contract owns its dispatch mode."""

    def generate_on_slot(self, sample: Sample) -> Sample:
        """Undecorated per-engine entry point for driver-side lane dispatch."""
        return self.generate(sample)

    def abort(self, ids: Optional[List[str]] = None) -> List[Sample]:
        """Best-effort cancel of in-flight generation; return any partials. Default no-op."""
        del ids
        return []

    def pause(self) -> None:
        """Stop admitting new generation (best-effort). Default no-op."""

    def resume(self) -> None:
        """Resume generation after :meth:`pause`. Default no-op."""

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

    def set_lora_from_tensors(
        self,
        adapter_name: str,
        lora_tensors: Dict[str, torch.Tensor],
        *,
        peft_config: Optional[dict] = None,
    ) -> None:
        """Load a LoRA adapter directly from in-memory tensors."""
        raise NotImplementedError

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

    _version: int = 0

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def set_version(self, train_version: int) -> None:
        """Set the committed-update version after a full weight publication."""
        if train_version < 0:
            raise ValueError(f"train_version must be >= 0, got {train_version}")
        self._version = train_version

    def _stamp_output_version(self, sample: Sample) -> Sample:
        """Stamp ``self._version`` onto the frontier (last) gen Part."""
        if not sample.parts:
            return sample
        gen = sample.parts[-1].fill(output_version=self._version)
        return sample.with_parts([*sample.parts[:-1], gen])


__all__ = ["BaseRolloutEngine"]
