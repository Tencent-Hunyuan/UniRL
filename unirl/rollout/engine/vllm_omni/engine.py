"""``vllm_omni`` engine core — wiring + delegation only."""

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
        del device, rank
        self.cfg = config
        self.lock = threading.Lock()
        self.closed = False
        self._is_offloaded = False
        # A transition failure means sequential stage RPCs may have left the
        # engine partially awake/asleep. Normalize through sleep before another
        # wake; generate refuses until residency is consistent again.
        self.transition_failed = False
        logger.info(
            "VLLM-Omni engine config (complete typed config): %s; model_config_available=%s model_config=%s",
            config,
            model_config is not None,
            model_config,
        )

        self.adapter = get_adapter(config.modality)(
            config,
            model_config,
            strategy=strategy,
            tokenize_fn=lambda text, *, task, sys_type: self.backend.tokenize_prompt(
                text, task=task, sys_type=sys_type
            ),
        )
        self.packs_groups = self.adapter.packs_groups

        self.schedule_policy = self.adapter.schedule_policy() if self.adapter.needs_sigmas else None

        if ports is None:
            ports = VLLMOmniPorts.reserve()

        intent = config.server_intent(
            ports=ports,
            extra=self.adapter.boot_kwargs(),
        )
        self.backend = VLLMOmniBackend.boot(intent)

        self.weight_sync = WeightSync(
            self.backend,
            lora_copy_transport=self.adapter.lora_copy_transport,
        )

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def generate(self, sample: Sample) -> Sample:
        """Generate one whole DP shard synchronously."""
        with self.lock:
            if self.closed:
                raise RuntimeError("VLLMOmniRolloutEngine.generate called after shutdown")
            # Defense-in-depth for logical and partially completed lifecycle
            # transitions: callers that swallow a wake/sleep exception must never
            # generate with base weights or with only a subset of stages resident.
            require(
                not self._is_offloaded and not self.transition_failed,
                "VLLMOmniRolloutEngine.generate: engine is offloaded or its last lifecycle transition failed "
                "(wake_up first).",
            )
            if not self.adapter.supports_row_canvas:
                sample.reject_row_canvases(engine=f"VLLMOmniRolloutEngine[{self.cfg.modality}]")
            self.adapter.validate_request(sample)
            if self.adapter.needs_sigmas:
                ensure_sample_sigmas(sample, self.schedule_policy)
            calls = self.adapter.build_inputs(sample)
            per_request = self.backend.generate(
                calls,
                attach_lora=self.weight_sync.lora_loaded,
                ar_lora_passthrough=self.adapter.ar_lora_passthrough,
            )
            return self._stamp_output_version(self.adapter.build_response(sample, per_request))

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def sleep(self) -> None:
        """Sleep every stage at level 1 (AR via EngineCore, diffusion via worker task)."""
        if self._is_offloaded and not self.transition_failed:
            return
        try:
            self.backend.sleep_task()
        except Exception:
            # Stage RPCs are sequential. A later failure may leave only a
            # prefix asleep, so neither generate nor wake may trust the old
            # physical flag; a subsequent sleep/wake retries normalization.
            self._is_offloaded = False
            self.transition_failed = True
            raise
        self._is_offloaded = True
        self.transition_failed = False

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def wake_up(self) -> None:
        """Wake every stage (AR via EngineCore, diffusion via worker task) + restore LoRA."""
        if self.transition_failed:
            # Recover an unknown partial stage state to one known boundary
            # before attempting another wake. If this retry fails, retain the
            # latch so generate remains blocked and cleanup can retry sleep.
            try:
                self.backend.sleep_task()
            except Exception:
                logger.exception("vLLM-Omni failed to normalize a partial lifecycle transition")
                raise
            self._is_offloaded = True
            self.transition_failed = False

        if not self._is_offloaded:
            return

        # This body executes INSIDE each colocated train actor (BROADCAST).
        # Return the actor's train-phase allocation peak to the driver before
        # the engine subprocess re-maps its ~50 GiB weight pool: without
        # activation checkpointing the peak stays reserved in the actor's
        # caching allocator and the post-wake generate OOMs at a 2 MiB
        # allocation (LIN-382 qwen e2e-c/d — a driver-side flush in
        # trainer.train_step demonstrably does NOT reach this process).
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            self.backend.wake_task()
        except Exception:
            # ``wake_task`` fans stages sequentially. Mark the physical
            # state unknown before rollback so caller cleanup never skips
            # sleep merely because the pre-wake flag was True.
            self._is_offloaded = False
            self.transition_failed = True
            try:
                self.backend.sleep_task()
            except Exception:
                logger.exception("vLLM-Omni rollback sleep failed after partial wake")
            else:
                self._is_offloaded = True
                self.transition_failed = False
            raise
        self._is_offloaded = False  # physical state changed before LoRA restore
        try:
            self.weight_sync.restore_lora_after_wake()
        except Exception:
            # Roll back the physically awakened engine. Because all-stage sleep
            # is sequential, a rollback failure means physical residency is
            # unknown rather than reliably awake.
            try:
                self.backend.sleep_task()
            except Exception:
                self._is_offloaded = False
                self.transition_failed = True
                logger.exception("vLLM-Omni rollback sleep failed after LoRA restore error")
            else:
                self._is_offloaded = True
                self.transition_failed = False
            raise

    @property
    def is_offloaded(self) -> bool:
        return self._is_offloaded

    def health_check(self) -> bool:
        return self.backend.ping()

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def shutdown(self) -> None:
        with self.lock:
            if self.closed:
                return
            self.closed = True
            self.backend.shutdown()

    def tp_per_stage(self) -> Dict[int, int]:
        """``{stage_id: tensor_parallel_size}`` per stage (parsed from the"""
        return self.backend.tp_per_stage()

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
        self.weight_sync.update_weights_from_ipc(
            peft_config=peft_config,
            base_sync_done=base_sync_done,
            use_shm=use_shm,
            replica_rank=replica_rank,
        )

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
        self.weight_sync.init_weights_update_group(
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
        self.weight_sync.update_weights_from_distributed(
            names=names,
            dtypes=dtypes,
            shapes=shapes,
            group_name=group_name,
            target_modules=target_modules,
            flush_cache=flush_cache,
        )

    def destroy_weights_update_group(
        self,
        *,
        group_name: str,
        track_prefix: str = "",
    ) -> None:
        del track_prefix
        self.weight_sync.destroy_weights_update_group(group_name=group_name)

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
        self.weight_sync.update_weights_from_tensor(
            serialized_named_tensors=serialized_named_tensors,
            target_modules=target_modules,
            load_format=load_format,
            flush_cache=flush_cache,
        )

    def set_lora_from_tensors(
        self,
        adapter_name: str,
        lora_tensors: Dict[str, torch.Tensor],
        *,
        peft_config: Optional[dict] = None,
    ) -> None:
        self.weight_sync.set_lora_from_tensors(adapter_name, lora_tensors, peft_config=peft_config)

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def set_lora_from_tensors_copy(
        self,
        adapter_name: str,
        lora_tensors: Dict[str, torch.Tensor],
        *,
        peft_config: Optional[dict] = None,
    ) -> None:
        """Byte-copy LoRA push for the HI3 two-engine trainer."""
        self.weight_sync.set_lora_from_tensors_copy(adapter_name, lora_tensors, peft_config=peft_config)

    def loaded_param_checksums(self, *, names: List[str]) -> dict:
        return self.weight_sync.loaded_param_checksums(names=names)

    def loaded_lora_checksums(self, *, adapter_id: int, names: Optional[List[str]] = None) -> dict:
        return self.weight_sync.loaded_lora_checksums(adapter_id=adapter_id, names=names)


__all__ = ["VLLMOmniRolloutEngine"]
