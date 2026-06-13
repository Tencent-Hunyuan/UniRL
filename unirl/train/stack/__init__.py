"""Train stack package: one family-agnostic stack + pluggable micro-batch planners.

``unirl.train.stack`` was a single module; it is now a package, split along the one
real seam — the pure micro-batch *planning* (types, bin-packing math, planner
classes; CPU-testable, no FSDP/Ray) vs the distributed *driver* (``TrainStack``).
The public surface is unchanged — ``TrainStack`` and ``TrainStepResult`` import
from here exactly as before, and recipes' ``_target_: unirl.train.stack.TrainStack``
/ ``...TokenBudgetPlanner`` resolve through these re-exports.
"""

from unirl.train.stack.base import TrainStack, TrainStepResult
from unirl.train.stack.planning import CountPlanner, MicroPlanner, TokenBudgetPlanner, _build_micro_batch_slices

__all__ = [
    "CountPlanner",
    "MicroPlanner",
    "TokenBudgetPlanner",
    "TrainStack",
    "TrainStepResult",
    "_build_micro_batch_slices",
]
