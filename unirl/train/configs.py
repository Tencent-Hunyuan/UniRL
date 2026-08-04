from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

LoraModuleSelection = Union[str, Tuple[str, ...]]


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
    fsdp_mode: str = "full"
    reshard_after_forward: bool = True
    activation_checkpointing: bool = False
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
