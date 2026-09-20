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
        else:
            if len(target_modules) > 1:
                raise ValueError(
                    f"LoRA tensor {key!r} has no component prefix; expected one of {tuple(prefixed_targets)!r}"
                )
        groups[target][normalized_key] = tensor
    populated = {name: values for name, values in groups.items() if values}
    if len(target_modules) > 1 and set(populated) != set(target_modules):
        missing = sorted(set(target_modules) - set(populated))
        raise ValueError(f"LoRA update is missing component tensors for {missing}")
    return populated


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
        if not target_modules:
            raise ValueError("SGLang diffusion weight sync requires at least one target module")
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
        update_targets = self._single_update_target(
            target_modules,
            operation="tensor weight update",
        )
        self._backend.update_from_tensor(
            serialized_named_tensors=serialized_named_tensors,
            target_modules=update_targets,
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
        update_targets = self._single_update_target(
            target_modules,
            operation="distributed weight update",
        )
        self._backend.update_from_distributed(
            names=list(names),
            dtypes=list(dtypes),
            shapes=[list(shape) for shape in shapes],
            group_name=str(group_name),
            target_modules=update_targets,
            flush_cache=flush_cache,
        )

    def _single_update_target(
        self,
        target_modules: Optional[List[str]],
        *,
        operation: str,
    ) -> List[str]:
        targets = list(target_modules or self._target_modules)
        if len(targets) != 1:
            raise NotImplementedError(
                f"SGLang diffusion {operation} supports exactly one pipeline component; "
                f"got {targets!r}. Use component-specific syncs for dual-transformer models."
            )
        return targets

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
        if not stripped:
            raise ValueError("SGLang LoRA update contains no tensors after name adaptation")
        adapter_alpha = None
        if peft_config is not None:
            adapter_alpha = peft_config.get("lora_alpha")
        if adapter_alpha is not None and int(adapter_alpha) != adapter_alpha:
            raise ValueError(f"SGLang requires integral lora_alpha; got {adapter_alpha!r}")
        lora_alpha = int(adapter_alpha) if adapter_alpha is not None else None
        grouped = _partition_lora_tensors(stripped, self._target_modules)
        self._lora_loaded = False
        group_count = len(grouped)
        for index, (target_module, target_tensors) in enumerate(grouped.items()):
            try:
                self._backend.set_lora(
                    lora_tensors=target_tensors,
                    target_module=target_module,
                    lora_alpha=lora_alpha,
                )
            except Exception as exc:
                raise RuntimeError(
                    "SGLang LoRA update failed for component "
                    f"{target_module!r} after {index}/{group_count} component updates; "
                    "the backend may be partially updated, so retry the complete adapter update"
                ) from exc
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
        """Require a conservative LoRA repush after native sleep/wake."""
        self._lora_loaded = False

    @property
    def lora_dirty(self) -> bool:
        """True when LoRA is in use but the adapter must be (re)pushed before generate."""
        return self._uses_lora and not self._lora_loaded


__all__ = ["WeightSync"]
