"""Backend-agnostic schema dataclasses for the training stack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class OptimizerConfig:
    """AdamW-style optimizer hyperparameters consumed by the training actor."""

    learning_rate: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    weight_decay: float
    param_group_lrs: Optional[Dict[str, float]] = None


@dataclass
class LrSchedulerConfig:
    """Learning-rate scheduler hyperparameters."""

    type: str
    warmup_steps: int
    total_steps: int


def resolve_trainable_module(bundle: object, trainable_attr: str):
    """The module a backend wraps + optimizes + checkpoints."""
    tm = getattr(bundle, "trainable_module", None)
    return tm() if callable(tm) else getattr(bundle, trainable_attr)


__all__ = [
    "LrSchedulerConfig",
    "OptimizerConfig",
    "resolve_trainable_module",
]
