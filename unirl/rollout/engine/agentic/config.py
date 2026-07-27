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

from unirl.rollout.engine.base import BaseEngineConfig


@dataclass
class AgenticRolloutEngineConfig(BaseEngineConfig):
    """Config for the multi-turn (agentic) rollout engine."""

    #: Inner single-turn rollout engine config (e.g. ``SGLangEngineConfig``); kept
    #: ``Any`` so it is built from its own ``_target_`` via ``inner.make_engine``.
    inner: Any
    #: The environment (the world side of a turn) — an :class:`Environment`, kept
    #: ``Any`` so a ``_target_`` (e.g. ``ToolEnvironment`` + tools) is instantiated
    #: per worker by the walker. Must be re-entrant: one instance serves all of a
    #: worker's concurrent trajectories (LIN-522).
    env: Any

    #: Hard per-trajectory turn bound (the loop's ``for _ in range(max_turns)``).
    max_turns: int = 8
    #: Per-turn sampling params (``BaseSamplingParams``); its ``samples_per_prompt``
    #: is the GRPO group size ``n`` (read via ``total_samples_per_prompt``).
    episode_sampling: Any = None
    #: Max concurrent trajectories per worker — the drain thread-pool size (one
    #: trajectory per thread; a thread holds its trajectory across tool-wait).
    #: Set a small multiple of the inner backend ``concurrency`` so trajectories
    #: in tool-wait don't starve the GPU.
    per_worker_concurrency: int = 8
    #: Partial rollout (LIN-531): expose the coordinator as a ``submit``/``poll``/
    #: ``abort`` interface so the *trainer* can over-sample, commit the first
    #: ``batch_size`` complete GRPO groups, checkpoint the unfinished tail at a
    #: **turn boundary**, and carry it to the next round (resumed under new
    #: weights). ``False`` ⇒ ``abort`` never fires, so behaviour is byte-identical
    #: to the barrier ``generate``. The engine provides the mechanism; the policy
    #: (how many, when to sync) lives in the trainer.
    partial_rollout: bool = False

    def make_engine(self, **deps: Any):
        """Construct the runtime :class:`AgenticRolloutEngine` (lazy import)."""
        from unirl.rollout.engine.agentic.engine import AgenticRolloutEngine

        return AgenticRolloutEngine(config=self, **deps)


__all__ = ["AgenticRolloutEngineConfig"]
