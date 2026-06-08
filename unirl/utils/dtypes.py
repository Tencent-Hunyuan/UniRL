"""Small dtype helpers shared across config-driven runtime components."""

from __future__ import annotations

import inspect
from typing import Any, Dict, Optional

import torch

_DTYPE_ALIASES = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "half": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
    "float": torch.float32,
}


def parse_torch_dtype(
    value: Any,
    *,
    field_name: str = "dtype",
    allow_none: bool = False,
    allow_auto: bool = False,
) -> Optional[torch.dtype]:
    """Normalize a string-ish dtype name to ``torch.dtype``."""
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{field_name} must not be None.")
    if isinstance(value, torch.dtype):
        return value

    key = str(value).strip().lower()
    if allow_auto and key == "auto":
        return None
    if key in _DTYPE_ALIASES:
        return _DTYPE_ALIASES[key]

    valid = "bf16/fp16/fp32" + ("/auto" if allow_auto else "")
    raise ValueError(f"Unsupported {field_name}={value!r}. Use one of {valid}.")


def inject_model_dtype_kwarg(
    *,
    model_cls: Any,
    model_kwargs: Dict[str, Any],
    dtype: Optional[torch.dtype],
) -> None:
    """Populate the correct dtype kwarg for a model bundle constructor if supported."""
    if dtype is None:
        return
    try:
        params = inspect.signature(model_cls.__init__).parameters
    except Exception:
        params = {}

    if "torch_dtype" in params:
        model_kwargs["torch_dtype"] = dtype
    elif "dtype" in params:
        model_kwargs["dtype"] = dtype
