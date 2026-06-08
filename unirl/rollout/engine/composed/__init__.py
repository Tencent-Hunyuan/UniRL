"""Composed rollout engine — peer of sglang / sglang_llm / vllm_omni / trainside.

Holds two child engines (``llm`` + ``diffusion``) and orchestrates the
prompt-enhancement (PE) serial flow internally:

    raw prompts → LLM child (N PE candidates per prompt)
                → diffusion child (M images per PE)
                → 2-track ``RolloutResp`` with explicit lineage

The pipeline / actor mixin treat this engine as a single ``BaseRolloutEngine``;
the multi-track ``RolloutResp`` lets the dual-rollout output flow through the
existing ``_diffusion_track_key`` (Step 5) without any pipeline change.
"""

from unirl.rollout.engine.composed.config import ComposedRolloutEngineConfig
from unirl.rollout.engine.composed.engine import ComposedRolloutEngine

__all__ = [
    "ComposedRolloutEngine",
    "ComposedRolloutEngineConfig",
]
