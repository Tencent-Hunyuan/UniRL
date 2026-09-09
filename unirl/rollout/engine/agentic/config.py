"""Agentic rollout-engine configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from unirl.rollout.engine.base import BaseEngineConfig


@dataclass
class AgenticRolloutEngineConfig(BaseEngineConfig):
    """Config for the multi-turn (agentic) rollout engine."""

    inner: Any
    env: Any

    max_turns: int = 8
    episode_sampling: Any = None

    def make_engine(self, **deps: Any):
        """Construct the runtime :class:`AgenticRolloutEngine` (lazy import)."""
        from unirl.rollout.engine.agentic.engine import AgenticRolloutEngine

        return AgenticRolloutEngine(config=self, **deps)


__all__ = ["AgenticRolloutEngineConfig"]
