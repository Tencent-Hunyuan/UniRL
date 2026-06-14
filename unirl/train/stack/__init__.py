"""Train stack package: one family-agnostic driver + pluggable micro-batch planners.

Two seams:

- :class:`TrainStack` (:mod:`~unirl.train.stack.base`) — the distributed driver
  (device align, π_old freeze, micro-accumulation, optimizer step, EMA).
- the :mod:`~unirl.train.stack.planner` subpackage — the pure, CPU-testable
  micro-batch *planning* (plan types, bin-packing math, planner strategies), with no
  FSDP/Ray. :class:`TrainStack` composes one injected ``micro_planner``.

The public surface re-exported here is what recipes and siblings depend on:
``_target_: unirl.train.stack.TrainStack`` / ``...TokenBudgetPlanner`` resolve through
these names, and ``unified_model_stack`` imports ``_build_micro_batch_slices`` from here.
"""

from unirl.train.stack.base import TrainStack, TrainStepResult
from unirl.train.stack.planner import CountPlanner, MicroPlanner, TokenBudgetPlanner, _build_micro_batch_slices

__all__ = [
    "CountPlanner",
    "MicroPlanner",
    "TokenBudgetPlanner",
    "TrainStack",
    "TrainStepResult",
    "_build_micro_batch_slices",
]
