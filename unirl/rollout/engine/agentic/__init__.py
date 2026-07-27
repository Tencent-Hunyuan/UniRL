"""Agentic (multi-turn, tool-use) rollout engine (LIN-522).

A rank-0 coordinator over a DP-replicated slab of per-worker drain thread
pools; ``generate`` returns a flat ``List[Sample]`` of variable-depth
trajectories.
"""

from unirl.rollout.engine.agentic.config import AgenticRolloutEngineConfig
from unirl.rollout.engine.agentic.engine import AgenticRolloutEngine

__all__ = ["AgenticRolloutEngine", "AgenticRolloutEngineConfig"]
