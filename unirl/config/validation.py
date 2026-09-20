"""Shared per-field validation helpers for component configs."""

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
    """Return the canonical precision alias (``bf16``/``fp16``/``fp32``)."""
    dtype = parse_torch_dtype(value, field_name=field)
    return _CANONICAL_BY_DTYPE[dtype].value


__all__ = ["PrecisionName", "validate_precision_type"]
