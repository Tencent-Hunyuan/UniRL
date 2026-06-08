"""Canonical Python package for this repository (`unirl`)."""

from __future__ import annotations

import importlib
from typing import Dict, Tuple

__version__ = "0.1.0"

_LAZY_ATTRS: Dict[str, Tuple[str, str]] = {
    # shared types
    "RewardRequest": ("unirl.types", "RewardRequest"),
    "RewardResponse": ("unirl.types", "RewardResponse"),
    "RewardType": ("unirl.types", "RewardType"),
    "SamplingRequirements": ("unirl.types.sampling", "SamplingRequirements"),
    # sde
    "get_sigma_schedule": ("unirl.sde", "get_sigma_schedule"),
    # reward
    "RewardBackend": ("unirl.reward.base", "RewardBackend"),
    # utils
    "load_function": ("unirl.utils", "load_function"),
    "set_seed": ("unirl.utils", "set_seed"),
    "configure_logger": ("unirl.utils", "configure_logger"),
}

__all__ = ["__version__", *_LAZY_ATTRS.keys()]


def __getattr__(name: str):
    if name in _LAZY_ATTRS:
        module_name, attr_name = _LAZY_ATTRS[name]
        module = importlib.import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals().keys()) | set(__all__))
