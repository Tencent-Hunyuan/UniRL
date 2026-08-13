"""Config surface."""

from __future__ import annotations

from unirl.config.require import require
from unirl.config.validation import (
    PrecisionName,
    is_direct_sampling,
    validate_dynamic_dotpaths,
    validate_lora_target_modules,
    validate_offload_contract,
    validate_precision_type,
    validate_rollout_layout,
    validate_training_batch_geometry,
    validate_weight_sync_contract,
)

__all__ = [
    "PrecisionName",
    "is_direct_sampling",
    "require",
    "validate_dynamic_dotpaths",
    "validate_lora_target_modules",
    "validate_offload_contract",
    "validate_precision_type",
    "validate_rollout_layout",
    "validate_training_batch_geometry",
    "validate_weight_sync_contract",
]
