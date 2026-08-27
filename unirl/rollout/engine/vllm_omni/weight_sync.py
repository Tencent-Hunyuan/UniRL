"""Weight sync — the canonical sync ops + LoRA lifecycle, owned by one component."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch

from unirl.rollout.engine.vllm_omni.backends import Backend

logger = logging.getLogger(__name__)


class WeightSync:
    """Sync ops + LoRA lifecycle over the seam (one instance per engine)."""

    def __init__(
        self,
        backend: Backend,
        *,
        uses_lora: bool,
        lora_copy_transport: bool,
    ) -> None:
        self._backend = backend
        self._uses_lora = bool(uses_lora)
        self._lora_copy_transport = bool(lora_copy_transport)
        self._lora_loaded = False
        self._weights_released = False
        self._last_lora_name: Optional[str] = None
        self._last_lora_tensors: Optional[Dict[str, Any]] = None
        self._last_peft_config: Optional[dict] = None

    @property
    def lora_loaded(self) -> bool:
        """True when a pushed adapter should be activated on generate."""
        return self._lora_loaded

    @property
    def lora_dirty(self) -> bool:
        """True when LoRA is in use but the adapter must be (re)pushed."""
        return self._uses_lora and (self._weights_released or not self._lora_loaded)

    def update_weights_from_ipc(
        self,
        *,
        peft_config: Optional[dict] = None,
        base_sync_done: bool = False,
        use_shm: bool = False,
        replica_rank: Optional[int] = None,
    ) -> None:
        self._backend.update_from_ipc(
            peft_config=peft_config,
            base_sync_done=base_sync_done,
            use_shm=use_shm,
            replica_rank=replica_rank,
        )
        if peft_config and base_sync_done:
            self._lora_loaded = True
            self._weights_released = False

    def init_weights_update_group(
        self,
        *,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str = "nccl",
    ) -> None:
        self._backend.init_weights_group(
            master_address=str(master_address),
            master_port=int(master_port),
            rank_offset=int(rank_offset),
            world_size=int(world_size),
            group_name=str(group_name),
            backend=str(backend),
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
    ) -> None:
        self._backend.update_from_distributed(
            names=list(names),
            dtypes=list(dtypes),
            shapes=[list(s) for s in shapes],
            group_name=str(group_name),
            target_modules=list(target_modules) if target_modules else None,
            flush_cache=bool(flush_cache),
        )

    def destroy_weights_update_group(self, *, group_name: str) -> None:
        self._backend.destroy_weights_group(group_name=str(group_name))

    def update_weights_from_tensor(
        self,
        *,
        serialized_named_tensors: List[str],
        target_modules: Optional[List[str]] = None,
        load_format: Optional[str] = None,
        flush_cache: bool = True,
    ) -> None:
        self._backend.update_from_tensor(
            serialized_named_tensors=list(serialized_named_tensors),
            target_modules=list(target_modules) if target_modules else None,
            load_format=load_format,
            flush_cache=bool(flush_cache),
        )

    def set_lora_from_tensors(
        self,
        adapter_name: str,
        lora_tensors: Dict[str, torch.Tensor],
        *,
        peft_config: Optional[dict] = None,
    ) -> None:
        """Hot-swap the adapter via the zero-copy shm-handle transport."""
        self._cache_lora(adapter_name, lora_tensors, peft_config)
        self._backend.set_lora_handle(adapter_name=adapter_name, lora_tensors=lora_tensors, peft_config=peft_config)
        self._lora_loaded = True
        self._weights_released = False

    def set_lora_from_tensors_copy(
        self,
        adapter_name: str,
        lora_tensors: Dict[str, torch.Tensor],
        *,
        peft_config: Optional[dict] = None,
    ) -> None:
        """Hot-swap the adapter via the TP>1-safe byte-copy transport."""
        self._cache_lora(adapter_name, lora_tensors, peft_config)
        self._backend.set_lora_copy(adapter_name=adapter_name, lora_tensors=lora_tensors, peft_config=peft_config)
        self._lora_loaded = True
        self._weights_released = False

    def _cache_lora(self, adapter_name: str, lora_tensors: Dict[str, Any], peft_config: Optional[dict]) -> None:
        """Clone the adapter state so a sleep/wake cycle can re-push it."""
        self._last_lora_name = adapter_name
        if isinstance(lora_tensors, dict):
            self._last_lora_tensors = {
                name: t.detach().clone() if isinstance(t, torch.Tensor) else t for name, t in lora_tensors.items()
            }
        else:
            self._last_lora_tensors = lora_tensors
        self._last_peft_config = dict(peft_config or {})

    def loaded_param_checksums(self, *, names: List[str]) -> dict:
        return self._backend.param_checksums(names=list(names))

    def loaded_lora_checksums(self, *, adapter_id: int, names: Optional[List[str]] = None) -> dict:
        return self._backend.lora_checksums(adapter_id=int(adapter_id), names=names)

    def mark_weights_released(self) -> None:
        """The engine released the runtime memory — the worker-side LoRA pool"""
        self._weights_released = True

    def restore_lora_after_wake(self) -> None:
        """Re-push the cached adapter after a wake (v1 parity)."""
        if self._last_lora_tensors is None:
            self._weights_released = False
            return
        logger.info(
            "[LoRA-WAKE] Re-loading LoRA after sleep/wake. adapter_name=%s",
            self._last_lora_name,
        )
        try:
            if self._lora_copy_transport:
                self.set_lora_from_tensors_copy(
                    self._last_lora_name,
                    self._last_lora_tensors,
                    peft_config=self._last_peft_config,
                )
            else:
                self.set_lora_from_tensors(
                    self._last_lora_name,
                    self._last_lora_tensors,
                    peft_config=self._last_peft_config,
                )
        except Exception as exc:
            self._lora_loaded = False
            raise RuntimeError(
                f"[LoRA-WAKE] Failed to re-load LoRA adapter "
                f"{self._last_lora_name!r} after sleep/wake; refusing "
                f"to continue serving because rollout would silently "
                f"run the base model, drifting old/new log-probs and "
                f"the GRPO ratio. Original error: {exc!r}"
            ) from exc
        logger.info("[LoRA-WAKE] LoRA re-loaded successfully.")


__all__ = ["WeightSync"]
