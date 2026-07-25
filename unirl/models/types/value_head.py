"""Scalar value head for PPO-style critic training on AR hidden states."""

from __future__ import annotations

import torch
import torch.nn as nn


class ValueHead(nn.Module):
    """Linear critic ``V(h)`` on last hidden states.

    Kept in FP32 for stable value loss math (mirrors replay log-prob FP32 policy).
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1, bias=True, dtype=torch.float32)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Map ``[..., H]`` hidden states to ``[...,]`` scalar values."""
        return self.proj(hidden.float()).squeeze(-1)


__all__ = ["ValueHead"]
