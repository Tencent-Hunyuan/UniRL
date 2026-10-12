"""Train stack package: shared base, single-algorithm and unified-model stacks, pluggable micro-batch planners."""

from unirl.train.stack.anchor import prepare_segment_anchors, validate_anchor_contract
from unirl.train.stack.base import TrainStepResult, aggregate_update_results
from unirl.train.stack.loss_scale import resolve_loss_scales
from unirl.train.stack.planner import CountPlanner, MicroPlanner, TokenBudgetPlanner, _build_micro_batch_slices
from unirl.train.stack.single import TrainStack

__all__ = [
    "CountPlanner",
    "MicroPlanner",
    "TokenBudgetPlanner",
    "TrainStack",
    "TrainStepResult",
    "_build_micro_batch_slices",
    "aggregate_update_results",
    "prepare_segment_anchors",
    "resolve_loss_scales",
    "validate_anchor_contract",
]
