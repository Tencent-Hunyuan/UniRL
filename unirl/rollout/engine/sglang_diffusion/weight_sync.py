"""Weight sync — the canonical sync ops + LoRA lifecycle, owned by one component."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch

from unirl.rollout.engine.sglang_diffusion.backends import Backend
from unirl.utils.peft_merge import adapt_lora_for_sglang

logger = logging.getLogger(__name__)


def _partition_lora_tensors(
    tensors: Dict[str, torch.Tensor],
    target_modules: List[str],
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Split component-prefixed LoRA keys for SGLang's one-target IPC API."""
    groups: Dict[str, Dict[str, torch.Tensor]] = {name: {} for name in target_modules}
    default_target = target_modules[0]
    prefixed_targets = sorted(target_modules, key=len, reverse=True)
    for key, tensor in tensors.items():
        target = default_target
        normalized_key = key
        for candidate in prefixed_targets:
            prefix = f"{candidate}."
            if key.startswith(prefix):
                target = candidate
                normalized_key = key[len(prefix) :]
                break
        groups[target][normalized_key] = tensor
    return {name: values for name, values in groups.items() if values}


class WeightSync:
    """Sync ops + LoRA lifecycle over the seam (one instance per engine)."""

    def __init__(
        self,
        backend: Backend,
        *,
        pipeline_prefix: str,
        target_modules: List[str],
        uses_lora: bool,
    ) -> None:
        self._backend = backend
        self._pipeline_prefix = pipeline_prefix
        self._target_modules = list(target_modules)
        self._uses_lora = uses_lora
        self._lora_loaded = False

    def update_weights_from_tensor(
        self,
        *,
        serialized_named_tensors: List[str],
        target_modules: Optional[List[str]] = None,
        load_format: Optional[str] = None,
        flush_cache: bool = True,
    ) -> None:
        if not serialized_named_tensors:
            raise ValueError("serialized_named_tensors must be non-empty")
        self._backend.update_from_tensor(
            serialized_named_tensors=serialized_named_tensors,
            target_modules=list(target_modules or self._target_modules),
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
        target_modules: Optional[List[str]] = None,
        flush_cache: bool = True,
    ) -> None:
        if not names:
            raise ValueError("names must be non-empty for distributed update")
        self._backend.update_from_distributed(
            names=[self.normalize_tensor_weight_name(name) for name in names],
            dtypes=list(dtypes),
            shapes=[list(shape) for shape in shapes],
            group_name=str(group_name),
            target_modules=list(target_modules or self._target_modules),
            flush_cache=flush_cache,
        )

    def normalize_tensor_weight_name(self, name: str) -> str:
        """Make old pipeline-qualified names relative to upstream target modules."""
        for target_module in sorted(self._target_modules, key=len, reverse=True):
            prefix = f"{target_module}."
            if name.startswith(prefix):
                return name[len(prefix) :]
        return name

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
        stripped = adapt_lora_for_sglang(
            lora_tensors,
            pipeline_prefix=self._pipeline_prefix,
        )
        adapter_alpha = None
        adapter_rank = None
        if peft_config is not None:
            adapter_alpha = peft_config.get("lora_alpha")
            adapter_rank = peft_config.get("r")
        if adapter_alpha is not None and int(adapter_alpha) != adapter_alpha:
            raise ValueError(f"SGLang requires integral lora_alpha; got {adapter_alpha!r}")
        if adapter_rank is not None and int(adapter_rank) != adapter_rank:
            raise ValueError(f"SGLang requires integral LoRA rank; got {adapter_rank!r}")
        grouped = _partition_lora_tensors(stripped, self._target_modules)
        for target_module, target_tensors in grouped.items():
            self._backend.set_lora(
                lora_tensors=target_tensors,
                target_module=target_module,
                lora_alpha=(int(adapter_alpha) if adapter_alpha is not None else None),
                lora_rank=(int(adapter_rank) if adapter_rank is not None else None),
            )
        self._lora_loaded = True

        layer_names = set()
        for key in stripped:
            if key.endswith(".alpha"):
                continue
            base = key
            for suffix in (".lora_A.weight", ".lora_B.weight", ".lora_A", ".lora_B"):
                if base.endswith(suffix):
                    base = base[: -len(suffix)]
                    break
            layer_names.add(base)
        logger.info(
            "SGLang LoRA loaded from tensors (adapter=%s) — %d layers",
            adapter_name,
            len(layer_names),
        )

    def loaded_param_checksums(self, *, names: List[str]) -> Dict[int, List[Dict[str, str]]]:
        output = self._backend.weights_checksum(module_names=list(names))
        return {0: [{str(k): str(v) for k, v in output.items()}]}

    def mark_weights_released(self) -> None:
        """The engine released the runtime weights — the loaded LoRA pool is gone."""
        self._lora_loaded = False

    @property
    def lora_dirty(self) -> bool:
        """True when LoRA is in use but the adapter must be (re)pushed before generate."""
        return self._uses_lora and not self._lora_loaded


__all__ = ["WeightSync"]
