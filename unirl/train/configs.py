from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

from unirl.config.require import require

LoraModuleSelection = Union[str, Tuple[str, ...]]
_FSDP_MODES = ("full", "hybrid", "no_shard")


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
    # Shard degree: full = the whole world, hybrid = hsdp_shard_size ranks
    # (replicate across groups), no_shard = nobody (DDP — every rank keeps the full model).
    fsdp_mode: str = "full"
    hsdp_shard_size: int = 8
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


def normalize_fsdp_mode(fsdp_mode: str) -> str:
    """Canonicalize a configured shard mode, rejecting anything unrecognized."""
    mode = str(fsdp_mode).strip().lower()
    require(
        mode in _FSDP_MODES,
        f"training.fsdp.fsdp_mode={fsdp_mode!r} is not one of {list(_FSDP_MODES)}; "
        "an unrecognized mode would silently fall back to full sharding.",
    )
    return mode


def resolve_fsdp_mesh_shape(
    fsdp_mode: str,
    *,
    world_size: int,
    hsdp_shard_size: int,
) -> Optional[Tuple[int, int]]:
    """Validate FSDP geometry and return its ``(replicate, shard)`` mesh shape."""
    require(
        isinstance(world_size, int) and world_size >= 1,
        f"training.fsdp world_size must be a positive integer, got {world_size!r}.",
    )
    fsdp_mode = normalize_fsdp_mode(fsdp_mode)
    if fsdp_mode == "full" or (fsdp_mode == "no_shard" and world_size == 1):
        return None
    if fsdp_mode == "no_shard":
        return (world_size, 1)

    require(
        isinstance(hsdp_shard_size, int) and hsdp_shard_size >= 2,
        f"training.fsdp.hsdp_shard_size must be an integer >= 2, got {hsdp_shard_size!r}.",
    )
    require(
        world_size > hsdp_shard_size,
        f"training.fsdp.fsdp_mode='hybrid' requires world_size > hsdp_shard_size "
        f"to form at least two replica groups, got world_size={world_size}, "
        f"hsdp_shard_size={hsdp_shard_size}. Use fsdp_mode='full' for one shard group.",
    )
    require(
        world_size % hsdp_shard_size == 0,
        f"training.fsdp.fsdp_mode='hybrid' requires world_size divisible by "
        f"hsdp_shard_size, got world_size={world_size}, hsdp_shard_size={hsdp_shard_size}.",
    )
    return (world_size // hsdp_shard_size, hsdp_shard_size)


__all__ = [
    "LoraConfig",
    "LoraModuleSelection",
    "EmaLoraConfig",
    "EmaFullConfig",
    "FSDPConfig",
    "normalize_fsdp_mode",
    "resolve_fsdp_mesh_shape",
]
