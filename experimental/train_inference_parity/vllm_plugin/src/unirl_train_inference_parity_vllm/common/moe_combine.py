"""Shared correctness-first MoE output combination."""

from __future__ import annotations

import torch


def moe_combine(
    contributions: torch.Tensor,
    row_map: torch.Tensor,
) -> torch.Tensor:
    """Combine top-k expert rows with the parity rounding contract."""
    if contributions.dim() != 2 or row_map.dim() != 2:
        raise ValueError("moe_combine requires 2D contributions and row_map tensors")
    tokens, topk = row_map.shape
    hidden = int(contributions.shape[-1])
    total = torch.zeros(
        (tokens, hidden),
        dtype=torch.float32,
        device=contributions.device,
    )
    zero = torch.zeros_like(total)
    for slot in range(topk):
        rows = row_map[:, slot].long()
        contribution = contributions.index_select(0, rows.clamp_min(0)).float()
        total = total + torch.where((rows >= 0)[:, None], contribution, zero)
    total = total.to(torch.bfloat16).float()
    return total.to(torch.bfloat16)


__all__ = ["moe_combine"]
