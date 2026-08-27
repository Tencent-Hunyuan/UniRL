"""Weight sync — the canonical sync ops + LoRA lifecycle, owned by one component."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch

from unirl.rollout.engine.sglang.backends import Backend

logger = logging.getLogger(__name__)


class WeightSync:
    """Sync ops + LoRA lifecycle over the seam (one instance per engine)."""

    def __init__(self, backend: Backend, *, uses_lora: bool) -> None:
        self._backend = backend
        self._uses_lora = bool(uses_lora)
        self._active_adapter: Optional[str] = None
        self._lora_loaded = False
        self._lora_version = 0

    def update_weights_from_tensor(
        self,
        *,
        serialized_named_tensors: List[str],
        load_format: Optional[str] = None,
        flush_cache: bool = True,
    ) -> None:
        if not serialized_named_tensors:
            raise ValueError("serialized_named_tensors must be non-empty")
        self._backend.update_from_tensor(
            serialized_named_tensors=serialized_named_tensors,
            load_format=load_format,
            flush_cache=flush_cache,
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
    ) -> None:
        self._backend.init_weights_group(
            master_address=master_address,
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
        flush_cache: bool = True,
    ) -> None:
        if not names:
            raise ValueError("names must be non-empty for distributed update")
        clean_dtypes = [d.replace("torch.", "") if isinstance(d, str) else d for d in dtypes]
        self._backend.update_from_distributed(
            names=list(names),
            dtypes=clean_dtypes,
            shapes=[list(shape) for shape in shapes],
            group_name=str(group_name),
            flush_cache=flush_cache,
        )

    def destroy_weights_update_group(self, *, group_name: str) -> None:
        self._backend.destroy_weights_group(group_name=str(group_name))

    def set_lora_from_tensors(
        self,
        adapter_name: str,
        lora_tensors: Dict[str, torch.Tensor],
        *,
        peft_config: Optional[dict] = None,
    ) -> None:
        """Push a LoRA adapter from in-memory tensors."""
        nickname = self._next_lora_nickname(adapter_name)
        self._backend.set_lora(
            lora_name=nickname,
            lora_tensors=lora_tensors,
            config_dict=peft_config,
        )
        self._active_adapter = nickname
        self._lora_loaded = True
        logger.info(
            "sglang: LoRA adapter %r loaded as %r (%d tensor keys)",
            adapter_name,
            nickname,
            len(lora_tensors),
        )

    def _next_lora_nickname(self, adapter_name: str) -> str:
        self._lora_version += 1
        return f"{adapter_name}_v{self._lora_version}"

    def mark_weights_released(self) -> None:
        """The engine released the runtime weights — the loaded LoRA pool is gone."""
        self._lora_loaded = False

    @property
    def active_adapter(self) -> Optional[str]:
        """The adapter name generation should tag requests with (None = base)."""
        return self._active_adapter if self._lora_loaded else None

    @property
    def lora_dirty(self) -> bool:
        """True when LoRA is in use but the adapter must be (re)pushed before generate."""
        return self._uses_lora and not self._lora_loaded


__all__ = ["WeightSync"]
