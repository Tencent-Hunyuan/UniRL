"""π_old anchor contract shared by every train stack."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from unirl.algorithms.base import StageAlgorithm
from unirl.train.stack.planner import Plan, arranged_slice, restore_row_order
from unirl.types.sample import Part


def validate_anchor_contract(algorithm: StageAlgorithm) -> None:
    """Reject anchor declarations whose per-micro outputs would be discarded."""
    recomputes_anchor = algorithm.recomputes_anchor
    if not isinstance(recomputes_anchor, bool):
        raise TypeError(f"{type(algorithm).__name__}.recomputes_anchor must be a bool attribute.")
    if recomputes_anchor and not algorithm.anchor_fields:
        raise ValueError(f"{type(algorithm).__name__} recomputes its anchor but declares no anchor_fields.")


def prepare_segment_anchors(
    algorithm: StageAlgorithm, part: Part, plan: Plan, *, order: Optional[torch.Tensor]
) -> None:
    """Freeze declared anchors over the plan's micros, before any update, and write them back onto ``part``."""
    if part.segment is None:
        return
    micro_slices = [r for update in plan for r in update]
    if not algorithm.recomputes_anchor or len(micro_slices) == 1:
        algorithm.prepare_segment(conditions=part.conditions, segment=part.segment)
        return
    collected: Dict[str, List[torch.Tensor]] = {field: [] for field in algorithm.anchor_fields}
    for start, end in micro_slices:
        micro = arranged_slice(part, order, start, end)
        algorithm.prepare_segment(conditions=micro.conditions, segment=micro.segment)
        for field in collected:
            value = getattr(micro.segment, field, None)
            if value is None:
                raise RuntimeError(
                    f"{type(algorithm).__name__} declares anchor field {field!r} but a micro produced None."
                )
            collected[field].append(value)
    for field, tensors in collected.items():
        setattr(
            part.segment,
            field,
            restore_row_order(torch.cat(tensors, dim=0), order, segment=part.segment, field=field),
        )
