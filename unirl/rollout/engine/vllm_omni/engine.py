"""``vllm_omni`` engine core — wiring + delegation only.

A thin core over the backend seam: it names no concrete modality (the adapter,
picked from the registry by ``config.modality``, owns the ``Sample`` →
``Sample`` conversion and the per-modality topology knobs)
and no concrete backend (the seam owns the runtime — boot, ports, env quirks,
the per-stage ``collective_rpc`` fan-out). Weight sync is a :class:`WeightSync`
component constructed over the seam; the offload lifecycle (a single flag)
lives directly on the engine. The frozen ``base.py`` surface is implemented as
thin forwards here — they must be real class attributes anyway (``Worker.call``
dispatches by name; ``@distributed`` binds the most-derived attribute) — which
also absorbs the surface quirks (``track_prefix``) so the component keeps
clean signatures.

One-shot construction: after ``__init__`` returns, the ``Omni`` orchestrator
is spawned and the engine is usable. ``generate`` / ``sleep`` / ``wake_up``
re-apply ``@distributed`` (the decorator is not inherited — see ``base.py``).
``set_lora_from_tensors_copy`` additionally keeps v1's ``@distributed(BROADCAST)``
— the documented exception to the "weight-sync entry points undecorated" rule:
it is how the HI3 two-engine LoRA sync reaches engines anchored on disjoint
worker partitions.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

import torch

from unirl.config.require import require
from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.rollout.engine.base import BaseRolloutEngine
from unirl.rollout.engine.vllm_omni.adapters import get_adapter
from unirl.rollout.engine.vllm_omni.backends import VLLMOmniBackend
from unirl.rollout.engine.vllm_omni.config import VLLMOmniEngineConfig, VLLMOmniPorts
from unirl.rollout.engine.vllm_omni.weight_sync import WeightSync
from unirl.sde.runtime import ensure_sample_sigmas
from unirl.types.sample import Sample

logger = logging.getLogger(__name__)


class VLLMOmniRolloutEngine(BaseRolloutEngine):
    """Rollout engine backed by vllm-omni's ``Omni`` orchestrator (v2 layout)."""

    _component_name = "vllm_omni"

    def __init__(
        self,
        config: VLLMOmniEngineConfig,
        *,
        device: Optional[torch.device] = None,
        strategy: Any = None,
        rank: Optional[int] = None,
        model_config: Any = None,
        ports: Optional[VLLMOmniPorts] = None,
    ) -> None:
        self.cfg = config
        self._version = 0
        self._generate_lock = threading.Lock()
        self._shutdown_lock = threading.Lock()
        self._shutdown_requested = False
        self._shutdown_complete = False
        self.device = device
        self.strategy = strategy
        self.rank = rank
        self.model_config = model_config
        self._is_offloaded = False
        logger.info(
            "VLLM-Omni engine config (complete typed config): %s; model_config_available=%s model_config=%s",
            config,
            model_config is not None,
            model_config,
        )

        self.adapter = get_adapter(config.modality)(
            config, model_config, strategy=strategy, tokenize_fn=self._tokenize_prompt
        )

        self.schedule_policy = self.adapter.schedule_policy() if self.adapter.needs_sigmas else None

        if ports is None:
            ports = VLLMOmniPorts.reserve()

        intent = config.server_intent(
            model_config=model_config,
            ports=ports,
            extra=self.adapter.boot_kwargs(),
        )
        self._backend = VLLMOmniBackend.boot(intent)

        self._weight_sync = WeightSync(
            self._backend,
            uses_lora=bool(getattr(model_config, "use_lora", False)),
            lora_copy_transport=self.adapter.lora_copy_transport,
        )

    def _tokenize_prompt(self, text: str, *, task: str, sys_type: str) -> List[int]:
        """Late-bound bridge handed to the adapter as ``tokenize_fn``."""
        return self._backend.tokenize_prompt(text, task=task, sys_type=sys_type)

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def generate(self, sample: Sample) -> Sample:
        """Generate one whole DP shard synchronously."""
        return self._generate_locked(sample)

    def _generate_locked(self, sample: Sample) -> Sample:
        with self._generate_lock:
            if self._shutdown_requested:
                raise RuntimeError("VLLMOmniRolloutEngine.generate called after shutdown")
            return self._stamp_output_version(self._generate_core(sample))

    def _generate_core(self, sample: Sample) -> Sample:
        """Synchronous whole-Sample generation: validate, σ-pin, run, decode."""
        require(
            not self._is_offloaded,
            "VLLMOmniRolloutEngine.generate: engine is offloaded (wake_up first).",
        )
        self.adapter.validate_request(sample)
        if self.adapter.needs_sigmas:
            require(self.schedule_policy is not None, f"{type(self.adapter).__name__} has no sigma schedule policy")
            self._ensure_sample_sigmas(sample)
        calls = self.adapter.build_inputs(sample)
        per_request = self._backend.generate(
            calls,
            attach_lora=self._weight_sync.lora_loaded,
            ar_lora_passthrough=self.adapter.ar_lora_passthrough,
        )
        return self.adapter.build_response(sample, per_request)

    def _ensure_sample_sigmas(self, sample: Sample) -> None:
        """Pin the σ schedule onto the diffusion gen Part's ``DiffusionSamplingParams.sigmas``.

        σ is computed from the model-owned schedule policy and shared across the
        Part's samples. Idempotent — a pre-pinned σ is left as-is.
        """
        ensure_sample_sigmas(sample, self.schedule_policy)

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def sleep(self) -> None:
        """Fan ``handle_sleep_task`` to every stage's workers (level 2)."""
        if self._is_offloaded:
            return
        self._backend.sleep_task()
        self._is_offloaded = True
        self._weight_sync.mark_weights_released()

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def wake_up(self) -> None:
        """Fan ``handle_wake_task`` to every stage's workers + restore LoRA."""
        if not self._is_offloaded:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._backend.wake_task()
        try:
            self._weight_sync.restore_lora_after_wake()
        except Exception:
            self._is_offloaded = True
            raise
        self._is_offloaded = False

    @property
    def is_offloaded(self) -> bool:
        return self._is_offloaded

    def health_check(self) -> bool:
        return self._backend.ping()

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def shutdown(self) -> None:
        shutdown_lock = getattr(self, "_shutdown_lock", None)
        if shutdown_lock is None:
            backend = getattr(self, "_backend", None)
            if backend is not None:
                backend.shutdown()
            return

        with shutdown_lock:
            if getattr(self, "_shutdown_complete", False):
                return

            generate_lock = getattr(self, "_generate_lock", None)
            if generate_lock is None:
                self._shutdown_requested = True
                backend = getattr(self, "_backend", None)
                if backend is not None:
                    backend.shutdown()
            else:
                with generate_lock:
                    self._shutdown_requested = True
                backend = getattr(self, "_backend", None)
                if backend is not None:
                    with generate_lock:
                        backend.shutdown()
            self._shutdown_complete = True

    def tp_per_stage(self) -> Dict[int, int]:
        """``{stage_id: tensor_parallel_size}`` per stage (parsed from the
        stage YAML at boot). The IPC weight-sync handler needs this to skip
        orphan train ranks that exceed a stage's TP size."""
        return self._backend.tp_per_stage()

    def update_weights_from_ipc(
        self,
        *,
        peft_config: Optional[dict] = None,
        base_sync_done: bool = False,
        use_shm: bool = False,
        replica_rank: Optional[int] = None,
        track_prefix: str = "",
    ) -> None:
        del track_prefix
        self._weight_sync.update_weights_from_ipc(
            peft_config=peft_config,
            base_sync_done=base_sync_done,
            use_shm=use_shm,
            replica_rank=replica_rank,
        )
        self._version += 1

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
        del track_prefix
        self._weight_sync.init_weights_update_group(
            master_address=master_address,
            master_port=master_port,
            rank_offset=rank_offset,
            world_size=world_size,
            group_name=group_name,
            backend=backend,
        )

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
        del track_prefix
        self._weight_sync.update_weights_from_distributed(
            names=names,
            dtypes=dtypes,
            shapes=shapes,
            group_name=group_name,
            target_modules=target_modules,
            flush_cache=flush_cache,
        )
        self._version += 1

    def destroy_weights_update_group(
        self,
        *,
        group_name: str,
        track_prefix: str = "",
    ) -> None:
        del track_prefix
        self._weight_sync.destroy_weights_update_group(group_name=group_name)

    def update_weights_from_tensor(
        self,
        *,
        serialized_named_tensors: List[str],
        target_modules: Optional[List[str]] = None,
        load_format: Optional[str] = None,
        flush_cache: bool = True,
        track_prefix: str = "",
    ) -> None:
        del track_prefix
        self._weight_sync.update_weights_from_tensor(
            serialized_named_tensors=serialized_named_tensors,
            target_modules=target_modules,
            load_format=load_format,
            flush_cache=flush_cache,
        )
        self._version += 1

    def set_lora_from_tensors(
        self,
        adapter_name: str,
        lora_tensors: Dict[str, torch.Tensor],
        *,
        peft_config: Optional[dict] = None,
    ) -> None:
        self._weight_sync.set_lora_from_tensors(adapter_name, lora_tensors, peft_config=peft_config)

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def set_lora_from_tensors_copy(
        self,
        adapter_name: str,
        lora_tensors: Dict[str, torch.Tensor],
        *,
        peft_config: Optional[dict] = None,
    ) -> None:
        """Byte-copy LoRA push for the HI3 two-engine trainer.

        Decorated (v1 parity, the documented §dispatch exception):
        ``RemoteLoraWeightSync(copy=True)`` reaches the disjoint-partition HI3
        engines through this entry point.
        """
        self._weight_sync.set_lora_from_tensors_copy(adapter_name, lora_tensors, peft_config=peft_config)

    def loaded_param_checksums(self, *, names: List[str]) -> dict:
        return self._weight_sync.loaded_param_checksums(names=names)

    def loaded_lora_checksums(self, *, adapter_id: int, names: Optional[List[str]] = None) -> dict:
        return self._weight_sync.loaded_lora_checksums(adapter_id=adapter_id, names=names)

    @property
    def lora_dirty(self) -> bool:
        """True when LoRA is in use but the adapter must be (re)pushed."""
        return self._weight_sync.lora_dirty


__all__ = ["VLLMOmniRolloutEngine"]
