"""Agentic rollout-engine configuration (LIN-522).

Registered as a peer rollout engine (alongside ``sglang`` / ``composed`` /
``vllm_omni`` / ``trainside``) whose ``_target_`` points at
:class:`AgenticRolloutEngine`. The agentic engine wraps **one inner rollout
engine** (the single-turn generator it calls each turn) and **one environment**
(the tool/world side), and drives multi-turn rollout across a DP-replicated slab
with a rank-0 coordinator.

Like :class:`ComposedRolloutEngineConfig`, the ``inner`` and ``env`` fields are
kept ``Any``: each carries its own ``_target_`` and is built by the worker walker
(``Worker._resolve_init_kwargs``) before the engine is constructed — so each
worker gets its **own local** inner engine + environment instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from unirl.rollout.engine.synchronous import BaseEngineConfig


@dataclass
class AgenticRolloutEngineConfig(BaseEngineConfig):
    """Config for the multi-turn (agentic) rollout engine."""

    inner: Any
    env: Any

    max_turns: int = 8
    episode_sampling: Any = None
    per_worker_concurrency: int = 8
    partial_rollout: bool = False

    def make_engine(self, **deps: Any):
        """Construct the runtime :class:`AgenticRolloutEngine` (lazy import)."""
        from unirl.rollout.engine.agentic.engine import AgenticRolloutEngine

        return AgenticRolloutEngine(config=self, **deps)


__all__ = ["AgenticRolloutEngineConfig"]
