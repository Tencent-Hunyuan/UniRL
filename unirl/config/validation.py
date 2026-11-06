"""Shared validation helpers for component configs."""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any

import torch
from omegaconf import DictConfig

from unirl.utils.dtypes import parse_torch_dtype

logger = logging.getLogger(__name__)


class PrecisionName(str, Enum):
    """Canonical precision aliases accepted by config fields."""

    BF16 = "bf16"
    FP16 = "fp16"
    FP32 = "fp32"


_CANONICAL_BY_DTYPE = {
    torch.bfloat16: PrecisionName.BF16,
    torch.float16: PrecisionName.FP16,
    torch.float32: PrecisionName.FP32,
}


def validate_precision_type(value: Any, *, field: str) -> str:
    """Return the canonical precision alias (``bf16``/``fp16``/``fp32``)."""
    dtype = parse_torch_dtype(value, field_name=field)
    return _CANONICAL_BY_DTYPE[dtype].value


_NO_MEMORY_SAVER_MESSAGE = (
    "colocated rollout sleeps the SGLang engine to hand the GPU back to training, "
    "but enable_memory_saver is unset: SGLang then installs the Noop "
    "TorchMemorySaverAdapter, so release_memory_occupation reports success and "
    "frees nothing. Set enable_memory_saver: true on the SGLang engine config or its engine_kwargs "
    "(plus enable_weights_cpu_backup: true to move weights to host across the "
    "sleep), and re-check mem_fraction_static against the wake window afterwards."
)


def validate_memory_saver_contract(rollout_cfg: DictConfig, *, strict: bool) -> None:
    """A colocated engine that sleeps must enable the memory saver, or sleep frees nothing."""
    engine_cfg = rollout_cfg.get("config")
    while isinstance(engine_cfg, DictConfig) and not (engine_cfg.get("_target_") or "").endswith("SGLangEngineConfig"):
        engine_cfg = engine_cfg.get("inner")
    if not isinstance(engine_cfg, DictConfig):
        return
    # Same precedence as SGLangEngineConfig.server_intent: a set field overrides engine_kwargs.
    enabled = engine_cfg.get("enable_memory_saver")
    if enabled is None:
        enabled = (engine_cfg.get("engine_kwargs") or {}).get("enable_memory_saver")
    if enabled:
        return
    if strict:
        raise ValueError(_NO_MEMORY_SAVER_MESSAGE)
    logger.warning(_NO_MEMORY_SAVER_MESSAGE)


__all__ = [
    "PrecisionName",
    "validate_memory_saver_contract",
    "validate_precision_type",
]
