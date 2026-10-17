"""HF-ordered router implemented with public vLLM BI providers."""

from __future__ import annotations

import types

import torch

from ...common.providers import linear, softmax


def _gate_forward(self, hidden_states):
    if self.bias is not None:
        raise ValueError("Qwen3-MoE parity router gate requires bias=False")
    hidden = hidden_states.reshape(-1, hidden_states.shape[-1]).contiguous()
    return linear(hidden, self.weight, None), None


def install_gate(gate) -> None:
    gate.forward = types.MethodType(_gate_forward, gate)


def _compute_routing(
    self,
    hidden_states,
    router_logits,
    indices_type,
    **kwargs,
):
    del hidden_states, kwargs
    probabilities = softmax(router_logits.float().contiguous(), dim=-1)
    weights, indices = torch.topk(probabilities, self.top_k, dim=-1)
    if self.renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights.float(), indices.to(indices_type or torch.int32)


def install_hf_router(experts) -> None:
    router = experts.router
    if router.scoring_func != "softmax" or router.renormalize is not True:
        raise ValueError("Qwen3-MoE parity requires softmax routing with renormalize=true")
    router._compute_routing = types.MethodType(_compute_routing, router)


__all__ = ["install_gate", "install_hf_router"]
