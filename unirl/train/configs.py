from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class LoraConfig:
    rank: int = 8
    alpha: int = 16
    target_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    dropout: float = 0.0
    bias: str = "none"
    task_type: str = "FEATURE_EXTRACTION"


@dataclass
class EmaLoraConfig:
    rank: int = 8
    alpha: int = 16
    target_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
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
    # Optional high-precision master dtype for the TRAINABLE params (e.g. "fp32").
    # When set, fsdp_wrap keeps the trainable (LoRA) sharded master + optimizer
    # states at this dtype while MixedPrecisionPolicy(param_dtype) still casts the
    # all-gathered compute copy to param_dtype (bf16) — the standard "fp32 master +
    # bf16 compute" recipe flow_grpo relies on for stable low-gradient RL. None
    # (default) = master dtype follows param_dtype (the prior all-bf16 behavior).
    master_dtype: Optional[str] = None


__all__ = [
    "LoraConfig",
    "EmaLoraConfig",
    "EmaFullConfig",
    "FSDPConfig",
]
