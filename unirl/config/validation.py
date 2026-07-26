"""Shared per-field validation helpers for component configs.

Helpers here (e.g. :func:`validate_precision_type`) are called from individual
``__post_init__`` bodies so every dataclass that owns the same kind of field
validates it the same way, with the same error message.

The rules that span *multiple* recipe sections (engine ↔ sync, offload, layout)
live in :mod:`unirl.config.contracts` instead — they need no torch, and keeping
them dependency-free lets the same code gate the driver and statically guard
every shipped recipe in CI.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

import torch

from unirl.utils.dtypes import parse_torch_dtype


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
    """Return the canonical precision alias (``bf16``/``fp16``/``fp32``).

    Delegates alias expansion to ``parse_torch_dtype`` so all precision fields
    accept the same inputs (``bf16``/``bfloat16``, ``fp16``/``float16``/``half``,
    ``fp32``/``float32``/``float``) and raise the same ``ValueError`` on unknown
    names. Caller supplies ``field`` for error-message attribution.
    """
    dtype = parse_torch_dtype(value, field_name=field)
    return _CANONICAL_BY_DTYPE[dtype].value


__all__ = [
    "PrecisionName",
    "validate_precision_type",
]
