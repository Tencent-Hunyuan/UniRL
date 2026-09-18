"""Fixed-count micro-batching — the default planner."""

from __future__ import annotations

from unirl.algorithms.base import StageAlgorithm
from unirl.train.stack.planner.types import Plan, _build_micro_batch_slices, _update_ranges
from unirl.types.sample import Part


def _count_plan(*, total: int, num_updates: int, micro_batch_size: int) -> Plan:
    """Fixed-count plan: contiguous equal updates, each split into ``micro_batch_size`` micros."""
    plan: Plan = []
    for u_start, u_end in _update_ranges(total_size=total, num_updates=num_updates):
        plan.append(
            [
                (u_start + ms, u_start + me)
                for ms, me in _build_micro_batch_slices(total_size=u_end - u_start, micro_batch_size=micro_batch_size)
            ]
        )
    return plan


class CountPlanner:
    """Fixed-count micro-batches: every micro holds ``micro_batch_size`` samples."""

    def arrange(self, part: Part, *, num_updates: int, micro_batch_size: int) -> tuple[Part, Plan]:
        return part, _count_plan(
            total=int(part.batch_size),
            num_updates=num_updates,
            micro_batch_size=micro_batch_size,
        )

    def validate(self, algorithm: StageAlgorithm) -> None:
        return None
