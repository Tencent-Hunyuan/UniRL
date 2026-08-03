"""FastVideo model adapters."""

from unirl.rollout.engine.fastvideo.adapters.base import (
    FastVideoModelAdapter,
    get_adapter,
    registered_adapters,
)
from unirl.rollout.engine.fastvideo.adapters.wan21 import Wan21FastVideoAdapter

__all__ = [
    "FastVideoModelAdapter",
    "Wan21FastVideoAdapter",
    "get_adapter",
    "registered_adapters",
]
