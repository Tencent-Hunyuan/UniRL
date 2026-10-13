"""Agentic rollout-engine configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from unirl.rollout.engine.base import BaseEngineConfig, RolloutCapabilities, resolve_rollout_capabilities


@dataclass
class AgenticRolloutEngineConfig(BaseEngineConfig):
    """Config for the multi-turn (agentic) rollout engine."""

    inner: Any
    env: Any

    max_turns: int = 8
    episode_sampling: Any = None

    @classmethod
    def resolve_rollout_capabilities(
        cls,
        config: Mapping[str, object],
        *,
        track_prefix: str = "",
    ) -> RolloutCapabilities:
        """Forward the configured inner engine's capabilities."""
        return resolve_rollout_capabilities(config.get("inner"), track_prefix=track_prefix)

    def make_engine(self, **deps: Any):
        """Construct the runtime :class:`AgenticRolloutEngine` (lazy import)."""
        from unirl.rollout.engine.agentic.engine import AgenticRolloutEngine

        return AgenticRolloutEngine(config=self, **deps)


__all__ = ["AgenticRolloutEngineConfig"]
