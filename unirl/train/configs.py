from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

LoraModuleSelection = Union[str, Tuple[str, ...]]


@dataclass
class FrozenAdapterSpec:
    """A frozen, inference-only LoRA adapter loaded at build time."""

    name: str = ""
    path: str = ""


def normalize_frozen_adapters(specs: Any) -> List[FrozenAdapterSpec]:
    """Validate ``LoraConfig.frozen_adapters`` entries into typed specs."""
    if not specs:
        return []
    normalized: List[FrozenAdapterSpec] = []
    for entry in specs:
        if isinstance(entry, FrozenAdapterSpec):
            name, path = entry.name, entry.path
        elif hasattr(entry, "get"):
            name, path = entry.get("name"), entry.get("path")
        else:
            name, path = getattr(entry, "name", None), getattr(entry, "path", None)
        if not name or not path:
            raise ValueError(f"frozen_adapters entries need non-empty 'name' and 'path'; got {entry!r}.")
        if str(name) == "default":
            raise ValueError("frozen_adapters: 'default' is the trainable adapter and cannot be frozen.")
        normalized.append(FrozenAdapterSpec(name=str(name), path=str(path)))
    names = [s.name for s in normalized]
    if len(set(names)) != len(names):
        raise ValueError(f"frozen_adapters names must be unique, got {names}.")
    return normalized


@dataclass
class LoraConfig:
    rank: int = 8
    alpha: int = 16
    target_modules: Any = ("q_proj", "k_proj", "v_proj", "o_proj")
    exclude_modules: Any = None
    module_prefix: str = ""
    dropout: float = 0.0
    bias: str = "none"
    task_type: str = "FEATURE_EXTRACTION"
    # Frozen inference-only adapters next to the trainable ``default`` (e.g. OPD
    # teachers): {name, path} entries (``Any`` for the same OmegaConf 2.3 reason
    # as ``target_modules``).
    frozen_adapters: Any = None


@dataclass
class EmaLoraConfig:
    rank: int = 8
    alpha: int = 16
    target_modules: Any = ("q_proj", "k_proj", "v_proj", "o_proj")
    exclude_modules: Any = None
    dropout: float = 0.0
    bias: str = "none"
    task_type: str = "FEATURE_EXTRACTION"
    default_adapter: str = "default"
    shadow_adapter: str = "old"
    ema_decay: float = 0.001
    ema_decay_type: str = "constant"
    ema_flat_steps: int = 0
    ema_uprate: float = 0.001
    ema_uphold: float = 0.5
    timing: str = "rollout_end"


@dataclass
class EmaFullConfig:
    target_decay: float = 0.9999
    timing: str = "optimizer_step"
    shadow_prefix: str = "shadow_"


@dataclass
class FSDPConfig:
    param_dtype: str = "bf16"
    cpu_offload: bool = False
    mixed_precision: bool = True
    # Match FSDP2's default: cast floating block inputs to param_dtype.
    cast_forward_inputs: bool = True
    # Shard degree: full = the whole world, hybrid = one node (replicate across nodes), no_shard = nobody (DDP — every rank keeps the full model).
    fsdp_mode: str = "full"
    reshard_after_forward: bool = True
    activation_checkpointing: bool = False
    # AC/FSDP composition order. "outside" (default) keeps FSDP's gather/cast
    # hooks out of the checkpointed region — the composition every pre-knob AC
    # recipe ran. "inside" re-enters the hooks during recompute; opt in per
    # recipe where that order was actually validated. See fsdp_wrap.
    ac_wrap_order: str = "outside"
    use_torch_compile: bool = False
    # Defer FSDP2 gradient reduce-scatter only under ZeRO-2.
    defer_grad_sync: bool = False
    forward_prefetch: bool = False
    master_dtype: Optional[str] = None
    # Disable root sharding when stages call submodules outside the root forward.
    root_wrap: bool = True
    checkpoint_format: str = "torch"
    checkpoint_async: bool = False
    sp_size: int = 1
    ep_size: int = 1


__all__ = [
    "LoraConfig",
    "LoraModuleSelection",
    "EmaLoraConfig",
    "EmaFullConfig",
    "FSDPConfig",
]
