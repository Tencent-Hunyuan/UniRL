"""Compatibility re-export of the SDE flow-match scheduler.

The implementation now lives in
``unirl.rollout.engine.vllm_omni._shared.flow_match_sde_scheduler``
so the SD3.5 RL pipeline can import the same class. This module keeps
the legacy import path used by ``hi3/pipeline.py`` working without
churn.
"""

from __future__ import annotations

from unirl.rollout.engine.vllm_omni._shared.flow_match_sde_scheduler import (
    FlowMatchSDEDiscreteScheduler,
    FlowMatchSDESchedulerOutput,
)

__all__ = ["FlowMatchSDEDiscreteScheduler", "FlowMatchSDESchedulerOutput"]
