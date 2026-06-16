"""Plan vocabulary and the planner contract — the package leaf.

A *plan* is a recipe, built before any forward runs, that says how each optimizer
step's samples are grouped into micro-batches::

    Plan       = List[UpdatePlan]   # the whole rollout shard
    UpdatePlan = List[Range]        # one optimizer step (= one "update")
    Range      = (start, end)       # one micro's CONTIGUOUS sample membership

A micro is ALWAYS a contiguous range: the token-budget planner reorders the track
up front (sort-then-slice, see :mod:`unirl.train.stack.planner.packed`) so no index
lists are ever threaded through the driver.

This module holds the plan types, the shared range helpers, and the
:class:`MicroPlanner` Protocol that :class:`~unirl.train.stack.base.TrainStack`
composes (one injected ``micro_planner``) instead of subclassing. The concrete
strategies live in :mod:`~unirl.train.stack.planner.count` (fixed-count, the
default) and :mod:`~unirl.train.stack.planner.packed` (token-budget packing).
"""

from __future__ import annotations

from typing import List, Protocol, Tuple, runtime_checkable

from unirl.algorithms import StageAlgorithm
from unirl.types.rollout_resp import RolloutTrack

# A plan is built before any forward runs:
#     Plan       = List[UpdatePlan]   # the whole rollout shard
#     UpdatePlan = List[Range]        # one optimizer step (= one "update")
#     Range      = (start, end)       # one micro's CONTIGUOUS sample membership
# Because packing reorders the track up front (sort-then-slice), a micro is ALWAYS
# a contiguous range — no index lists.
Range = Tuple[int, int]
UpdatePlan = List[Range]
Plan = List[UpdatePlan]


def _positive_int(*, name: str, value: object) -> int:
    resolved = int(value)
    if resolved < 1:
        raise ValueError(f"{name} must be >= 1. Got {resolved}.")
    return resolved


def _update_ranges(*, total_size: int, num_updates: int) -> Tuple[Range, ...]:
    """Partition ``[0, total_size)`` into ``num_updates`` equal contiguous updates.

    One range = one optimizer step. Even divisibility is required: the per-worker
    batch is fixed (DP sharding is even) and a ragged final update would silently
    drop samples and desync grad accumulation across DP ranks.
    """
    total = _positive_int(name="total_size", value=total_size)
    n = _positive_int(name="num_updates_per_batch", value=num_updates)
    if total % n != 0:
        raise ValueError(
            f"num_updates_per_batch={n} must evenly divide the per-worker batch "
            f"size ({total}); got remainder {total % n}. Adjust batch_size, "
            f"samples_per_prompt, or num_updates_per_batch."
        )
    update_size = total // n
    return tuple((i * update_size, (i + 1) * update_size) for i in range(n))


def _build_micro_batch_slices(
    *,
    total_size: int,
    micro_batch_size: int,
) -> Tuple[Range, ...]:
    """Contiguous fixed-count micro ranges over ``[0, total_size)``."""
    resolved_total_size = _positive_int(name="total_size", value=total_size)
    resolved_micro_batch_size = _positive_int(name="micro_batch_size", value=micro_batch_size)
    slices: List[Range] = []
    start = 0
    while start < resolved_total_size:
        end = min(start + resolved_micro_batch_size, resolved_total_size)
        slices.append((start, end))
        start = end
    return tuple(slices)


@runtime_checkable
class MicroPlanner(Protocol):
    """How an update's samples are grouped into micro-batches.

    :meth:`arrange` returns ``(track, plan)``: the track to train on — possibly
    reordered so packed micros are contiguous (sort-then-slice) — and one
    :data:`UpdatePlan` of contiguous ranges per optimizer step. :meth:`validate` is
    the algorithm precondition the grouping needs, checked once when the stack is built.
    """

    def arrange(
        self, resp_track: RolloutTrack, *, num_updates: int, micro_batch_size: int
    ) -> Tuple[RolloutTrack, Plan]: ...

    def validate(self, algorithm: StageAlgorithm) -> None: ...
