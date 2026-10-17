"""Per-micro ``loss_scale`` resolution shared by every train stack; see ``../readme.md`` Gotchas."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from unirl.algorithms.base import StageAlgorithm
from unirl.distributed.group.remote import RankInfo
from unirl.train.backend.base_backend import BaseFSDP2Backend
from unirl.train.stack.planner import UpdatePlan
from unirl.types.loss_agg import LossAggMode
from unirl.types.sample import Part


def _row_token_counts(part: Part, *, owner: str) -> List[float]:
    """Valid-token count of every row (loss_mask-aware), read back in one host transfer."""
    segment = part.segment
    if segment is None:
        raise ValueError(f"{owner}: loss_agg_mode='token-mean' requires a segment.")
    cu = segment.cu_seqlens
    loss_mask = getattr(segment, "loss_mask", None)
    if loss_mask is not None and cu is not None:
        csum = torch.nn.functional.pad(loss_mask.cumsum(0, dtype=torch.float64), (1, 0))
        return (csum[cu[1:]] - csum[cu[:-1]]).tolist()
    if segment.lengths is not None:
        return segment.lengths.tolist()
    raise ValueError(
        f"{owner}: loss_agg_mode='token-mean' requires a packed segment "
        "(cu_seqlens/lengths) — build it via TextSegment.pack(...)."
    )


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
    """Per-micro ``loss_scale`` and DP-global valid tokens (``None`` = sample share; ``0.0`` = all scales 0)."""
    if getattr(algorithm, "loss_agg_mode", None) != LossAggMode.TOKEN_MEAN:
        update_total = sum(end - start for start, end in micros)
        return [(end - start) / update_total for start, end in micros], None
    if rank_info is not None and rank_info.sp_size > 1:
        # Reject sequence parallelism until loss denominators include the SP dimension.
        raise ValueError(
            f"{owner}: loss_agg_mode='token-mean' is not validated under "
            f"sequence parallelism (sp_size={rank_info.sp_size}); use sp_size=1."
        )
    counts = _row_token_counts(part, owner=owner)
    rows = order.tolist() if order is not None else range(len(counts))
    weights = [float(sum(counts[row] for row in rows[start:end])) for start, end in micros]
    (global_total,) = backend.all_reduce_loss_sums([sum(weights)])
    if global_total <= 0.0:
        return [0.0] * len(weights), global_total
    dp_world = backend.gradient_average_world_size()
    return [w * dp_world / global_total for w in weights], global_total


__all__ = ["resolve_loss_scales"]
