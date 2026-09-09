"""Scalar value head for PPO-style critic training on AR hidden states."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class ValueHead(nn.Module):
    """Linear critic ``V(h)`` on the hidden state that predicts each action."""

    def __init__(self, hidden_size: int, *, device: Any = None) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1, bias=True, dtype=torch.float32, device=device)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Start from ``V(s)=0``; deterministic and safe after sharded meta-init."""
        nn.init.zeros_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Map ``[..., H]`` hidden states to FP32 scalar values ``[...]``."""
        weight = self.proj.weight.float()
        bias = self.proj.bias.float() if self.proj.bias is not None else None
        return F.linear(hidden.float(), weight, bias).squeeze(-1)


__all__ = ["ValueHead"]
