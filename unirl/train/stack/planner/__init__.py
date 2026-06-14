"""Micro-batch planners: how an update's samples are grouped into micro-batches.

One family-agnostic seam injected into :class:`~unirl.train.stack.base.TrainStack`
(composition, not subclassing). Two strategies:

- :class:`CountPlanner` (:mod:`~unirl.train.stack.planner.count`) — fixed-count
  micros; the default, used by the count-based diffusion / FlowGRPO recipes.
- :class:`TokenBudgetPlanner` (:mod:`~unirl.train.stack.planner.packed`) —
  token-budget bin-packing for varlen LLMs.

Both produce a :data:`Plan` of contiguous ``(start, end)`` ranges (see
:mod:`~unirl.train.stack.planner.types`); the driver only ever slices. Recipes
reference the planners by ``_target_`` (resolved through ``unirl.train.stack``).
"""

from unirl.train.stack.planner.count import CountPlanner, _count_plan
from unirl.train.stack.planner.packed import TokenBudgetPlanner
from unirl.train.stack.planner.types import (
    MicroPlanner,
    Plan,
    Range,
    UpdatePlan,
    _build_micro_batch_slices,
    _positive_int,
)

__all__ = [
    "CountPlanner",
    "MicroPlanner",
    "Plan",
    "Range",
    "TokenBudgetPlanner",
    "UpdatePlan",
    "_build_micro_batch_slices",
    "_count_plan",
    "_positive_int",
]
