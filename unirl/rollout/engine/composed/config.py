"""Composed rollout-engine configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from unirl.rollout.engine.base import BaseEngineConfig


@dataclass
class ComposedRolloutEngineConfig(BaseEngineConfig):
    """Two-stage prompt-enhancement (PE) composed rollout engine."""

    ar: Any
    diffusion: Any

    sleep_diffusion_on_start: bool = True

    pe_instruction: Optional[str] = None

    pe_marker: Optional[str] = None

    pe_max_chars: Optional[int] = None


__all__ = ["ComposedRolloutEngineConfig"]
