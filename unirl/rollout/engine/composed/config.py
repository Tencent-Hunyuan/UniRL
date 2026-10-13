"""Composed rollout-engine configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Optional

from unirl.rollout.engine.base import BaseEngineConfig, RolloutCapabilities, resolve_rollout_capabilities


@dataclass
class ComposedRolloutEngineConfig(BaseEngineConfig):
    """Two-stage prompt-enhancement (PE) composed rollout engine."""

    ar: Any
    diffusion: Any

    sleep_diffusion_on_start: bool = True

    pe_instruction: Optional[str] = None

    pe_marker: Optional[str] = None

    pe_max_chars: Optional[int] = None

    @classmethod
    def resolve_rollout_capabilities(
        cls,
        config: Mapping[str, object],
        *,
        track_prefix: str = "",
    ) -> RolloutCapabilities:
        """Expose capabilities of the generation or explicitly routed sync child."""
        child_name = track_prefix or "diffusion"
        if child_name not in {"ar", "diffusion"}:
            raise ValueError(f"Unknown composed rollout track_prefix {track_prefix!r}.")
        return resolve_rollout_capabilities(config.get(child_name), track_prefix=track_prefix)


__all__ = ["ComposedRolloutEngineConfig"]
