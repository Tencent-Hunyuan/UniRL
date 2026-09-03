"""FastVideo model adapters."""

from unirl.rollout.engine.fastvideo.adapters.base import (
    FastVideoModelAdapter,
    get_adapter,
    registered_adapters,
)
from unirl.rollout.engine.fastvideo.adapters.wan21 import Wan21FastVideoAdapter
from unirl.rollout.engine.fastvideo.adapters.wan22 import Wan22FastVideoAdapter

__all__ = [
    "FastVideoModelAdapter",
    "Wan21FastVideoAdapter",
    "Wan22FastVideoAdapter",
    "get_adapter",
    "registered_adapters",
]
