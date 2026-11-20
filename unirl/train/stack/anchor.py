"""π_old anchor contract shared by every train stack."""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional

import torch

from unirl.algorithms.base import StageAlgorithm
from unirl.train.stack.planner import Plan, arranged_slice, restore_row_order
from unirl.types.sample import Part


def first_update_part(algorithm: StageAlgorithm, part: Part) -> Part:
    """Reuse the current prediction as the first ART anchor without gathering cached velocities."""
    if not algorithm.first_update_self_anchor:
        return part
    return replace(part, segment=replace(part.segment, art_old_predictions=None, art_self_anchor=True))


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
    algorithm.validate_part(part)
    if part.segment is None:
        return
    if algorithm.first_update_self_anchor:
        if len(plan) != 2:
            raise ValueError("ART requires two disjoint update windows.")
        if part.segment.art_old_predictions is not None:
            return
        for start, end in plan[1]:
            micro = arranged_slice(part, order, start, end)
            algorithm.prepare_segment(conditions=micro.conditions, segment=micro.segment)
            predictions = micro.segment.art_old_predictions
            if start == plan[1][0][0]:
                anchors = predictions.new_zeros((part.batch_size, *predictions.shape[1:]))
            if order is None:
                anchors[start:end] = predictions
            else:
                anchors.index_copy_(0, order[start:end].to(anchors.device), predictions)
        part.segment.art_old_predictions = anchors
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
            value = getattr(micro.segment, field)
            if value is None:
                raise RuntimeError(
                    f"{type(algorithm).__name__} declares anchor field {field!r} but a micro produced None."
                )
            collected[field].append(value)
    for field, tensors in collected.items():
        shapes = {tuple(tensor.shape[1:]) for tensor in tensors}
        if len(shapes) != 1:
            raise RuntimeError(
                f"{type(algorithm).__name__} anchor field {field!r} differs in shape across micros ({sorted(shapes)}); "
                "an anchor stored per token needs one output canvas per rollout."
            )
        setattr(
            part.segment,
            field,
            restore_row_order(torch.cat(tensors, dim=0), order, segment=part.segment, field=field),
        )


def prepare_part_for_training(
    algorithm: StageAlgorithm,
    part: Part,
    plan: Plan,
    *,
    order: Optional[torch.Tensor],
    model: torch.nn.Module,
) -> Part:
    """Freeze ``part``'s anchors in eval mode, run the algorithm's ``prepare_part`` hook, then restore the mode."""
    was_training = model.training
    model.eval()
    try:
        prepare_segment_anchors(algorithm, part, plan, order=order)
        prepared = algorithm.prepare_part(part)
    finally:
        model.train(was_training)
    return prepared
