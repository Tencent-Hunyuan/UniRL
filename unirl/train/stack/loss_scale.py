"""Per-micro ``loss_scale`` resolution shared by every train stack; see ``../readme.md`` Gotchas."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from unirl.algorithms.base import LossNorm, StageAlgorithm
from unirl.distributed.group.remote import RankInfo
from unirl.train.backend.base_backend import BaseFSDP2Backend
from unirl.train.stack.planner import UpdatePlan
from unirl.types.sample import Part


def resolve_loss_scales(
    algorithm: StageAlgorithm,
    part: Part,
    *,
    micros: UpdatePlan,
    order: Optional[torch.Tensor],
    backend: BaseFSDP2Backend,
    rank_info: Optional[RankInfo],
    owner: str,
) -> Tuple[List[float], Optional[float]]:
    """Per-micro ``loss_scale`` and DP-global loss units (``None`` = sample share; ``0.0`` = all scales 0)."""
    units = algorithm.loss_units(part)
    if units is None:
        update_total = sum(end - start for start, end in micros)
        return [(end - start) / update_total for start, end in micros], None
    if rank_info is not None and rank_info.sp_size > 1:
        # Reject sequence parallelism until loss denominators include the SP dimension.
        raise ValueError(
            f"{owner}: loss_agg_mode='token-mean' is not validated under "
            f"sequence parallelism (sp_size={rank_info.sp_size}); use sp_size=1."
        )
    counts = units.tolist()
    rows = order.tolist() if order is not None else range(len(counts))
    weights = [float(sum(counts[row] for row in rows[start:end])) for start, end in micros]
    (global_total,) = backend.all_reduce_loss_sums([sum(weights)])
    if global_total <= 0.0:
        return [0.0] * len(weights), global_total
    norm = LossNorm(denominator=global_total, grad_world=backend.gradient_average_world_size())
    return [w * norm.grad_world / norm.denominator for w in weights], global_total


__all__ = ["resolve_loss_scales"]
