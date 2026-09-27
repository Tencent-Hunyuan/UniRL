"""Update-level sample ordering composed with a micro-batch planner."""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch

from unirl.train.stack.planner.count import CountPlanner, _count_plan
from unirl.train.stack.planner.types import MicroPlanner, Plan
from unirl.types.sample import Part

# ``order[i]`` is the source row at arranged position ``i``; ``None`` means identity.
Arrangement = Tuple[Part, Plan, Optional[torch.Tensor]]


def _micro_part(part: Part, order: Optional[torch.Tensor], start: int, end: int) -> Part:
    """Gather arranged positions ``[start, end)`` without materializing the full arranged Part."""
    return part.slice(start, end) if order is None else part.select(order[start:end])


def _restore_row_order(anchor: torch.Tensor, order: Optional[torch.Tensor], *, field: str) -> torch.Tensor:
    """Map an anchor concatenated in arranged order back to the Part's source row order."""
    if order is None:
        return anchor
    if int(anchor.shape[0]) != int(order.numel()):
        raise ValueError(
            f"shuffle_updates cannot restore anchor field {field!r}: expected one row per sample "
            f"({order.numel()}), got leading dim {anchor.shape[0]} (packed per-token anchors are unsupported)."
        )
    return anchor[torch.argsort(order).to(anchor.device)]


class UpdatePlanner:
    """Arrange update membership (shuffled via a lazy ``order``), then delegate micro-batching."""

    def __init__(self, micro_planner: MicroPlanner, *, shuffle_updates: bool, shuffle_seed: int) -> None:
        if shuffle_updates and not isinstance(micro_planner, CountPlanner):
            raise ValueError(
                f"shuffle_updates is only implemented for CountPlanner micro-batching; got {type(micro_planner).__name__}."
            )
        self.micro_planner = micro_planner
        self.shuffle_updates = shuffle_updates
        self.shuffle_seed = shuffle_seed

    def arrange_many(
        self,
        parts: Sequence[Part],
        *,
        num_updates: int,
        micro_batch_size: int,
        shuffle_step: Optional[int],
    ) -> Tuple[Arrangement, ...]:
        """Arrange lineage-aligned parts; one permutation of the coarsest part orders them all."""
        if not (self.shuffle_updates and num_updates > 1):
            return tuple(
                (*self.micro_planner.arrange(part, num_updates=num_updates, micro_batch_size=micro_batch_size), None)
                for part in parts
            )
        if shuffle_step is None:
            raise ValueError("shuffle_updates=True requires a stable shuffle_step (for example, rollout_id).")
        plans = [
            _count_plan(total=int(part.batch_size), num_updates=num_updates, micro_batch_size=micro_batch_size)
            for part in parts
        ]
        coarse = min(parts, key=lambda part: int(part.batch_size))
        coarse_size = int(coarse.batch_size)
        generator = torch.Generator()
        generator.manual_seed((self.shuffle_seed + shuffle_step) & 0xFFFFFFFF)
        coarse_order = torch.randperm(coarse_size, generator=generator)

        orders = []
        for part in parts:
            size = int(part.batch_size)
            if size % coarse_size:
                raise ValueError(
                    "Aligned update shuffle requires each Part batch size to be an integer multiple "
                    f"of the smallest size ({coarse_size}); got {[int(p.batch_size) for p in parts]}."
                )
            fanout = size // coarse_size
            if part is not coarse:
                if len(coarse.sample_ids) != coarse_size or len(part.sample_ids) != size:
                    raise ValueError("Aligned update shuffle across multiple Parts requires sample_ids on every Part.")
                for index, ancestor in enumerate(coarse.sample_ids):
                    descendants = part.sample_ids[index * fanout : (index + 1) * fanout]
                    if not all(sid == ancestor or sid.startswith(f"{ancestor}/") for sid in descendants):
                        raise ValueError(
                            "Aligned update shuffle requires each larger Part to be a contiguous lineage "
                            f"expansion of the smallest Part; {descendants!r} are not all descendants of {ancestor!r}."
                        )
            orders.append((coarse_order[:, None] * fanout + torch.arange(fanout)).reshape(-1))
        return tuple(zip(parts, plans, orders))
