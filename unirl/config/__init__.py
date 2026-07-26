"""Config surface.

Public entry points:
  - ``validate_recipe`` (``contracts``): the driver-side cross-component gate
    every ``unirl/train_*.py`` runs before building its trainer, plus the
    individual ``validate_*`` contracts it composes. Writing a new contract
    also needs ``RecipeFacts`` — import that from ``unirl.config.contracts``.
  - ``PrecisionName`` / ``validate_precision_type`` (``validation``): shared
    per-field helpers used by config dataclasses' ``__post_init__``.
  - ``require`` (``require``): one-line precondition helper for ``__post_init__``
    and cross-component contracts.
"""

from __future__ import annotations

from unirl.config.contracts import (
    is_direct_sampling,
    validate_offload_contract,
    validate_recipe,
    validate_rollout_layout,
    validate_weight_sync_contract,
)
from unirl.config.require import require
from unirl.config.validation import PrecisionName, validate_precision_type

__all__ = [
    "PrecisionName",
    "is_direct_sampling",
    "require",
    "validate_offload_contract",
    "validate_precision_type",
    "validate_recipe",
    "validate_rollout_layout",
    "validate_weight_sync_contract",
]
