"""Qwen3 MoE router using an explicit parity softmax provider."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .context import exact_enabled
from .linear_ops import providers


def router_forward(self, hidden_states):
    if not exact_enabled():
        original = getattr(router_forward, "_unirl_parity_original", None)
        if not callable(original):
            raise RuntimeError("Qwen3 parity router original is unavailable")
        return original(self, hidden_states)
    hidden = hidden_states.reshape(-1, self.hidden_dim)
    logits = F.linear(hidden, self.weight)
    probabilities = providers().softmax(logits.float(), dim=-1)
    weights, indices = torch.topk(probabilities, self.top_k, dim=-1)
    if self.norm_topk_prob:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return logits, weights.float(), indices


__all__ = ["router_forward"]
