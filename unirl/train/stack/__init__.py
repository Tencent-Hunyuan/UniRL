"""Train stack package: one family-agnostic driver + pluggable micro-batch planners."""

from unirl.train.stack.anchor import prepare_segment_anchors, validate_anchor_contract
from unirl.train.stack.base import TrainStack, TrainStepResult
from unirl.train.stack.loss_scale import resolve_loss_scales
from unirl.train.stack.planner import CountPlanner, MicroPlanner, TokenBudgetPlanner, _build_micro_batch_slices

__all__ = [
    "CountPlanner",
    "MicroPlanner",
    "TokenBudgetPlanner",
    "TrainStack",
    "TrainStepResult",
    "_build_micro_batch_slices",
    "prepare_segment_anchors",
    "resolve_loss_scales",
    "validate_anchor_contract",
]
