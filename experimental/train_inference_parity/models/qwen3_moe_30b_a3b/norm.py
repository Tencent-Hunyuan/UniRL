"""Exact RMSNorm forward with a finite Torch surrogate gradient."""

from __future__ import annotations

import torch

from .context import exact_enabled
from .linear_ops import providers


class _RmsNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, eps):
        ctx.save_for_backward(hidden, weight)
        ctx.eps = float(eps)
        return providers().rms_norm(hidden, weight.to(hidden.dtype), float(eps))

    @staticmethod
    def backward(ctx, grad_output):
        hidden, weight = ctx.saved_tensors
        needs_hidden, needs_weight = ctx.needs_input_grad[:2]
        with torch.enable_grad():
            hidden_ref = hidden.detach().requires_grad_(needs_hidden)
            weight_ref = weight.detach().requires_grad_(needs_weight)
            variance = hidden_ref.float().square().mean(dim=-1, keepdim=True)
            output = (hidden_ref.float() * torch.rsqrt(variance + ctx.eps) * weight_ref.float()).to(hidden.dtype)
            requested = [
                tensor
                for tensor, needed in (
                    (hidden_ref, needs_hidden),
                    (weight_ref, needs_weight),
                )
                if needed
            ]
            computed = torch.autograd.grad(
                output,
                requested,
                grad_output,
                allow_unused=True,
            )
        iterator = iter(computed)
        return (
            next(iterator) if needs_hidden else None,
            next(iterator) if needs_weight else None,
            None,
        )


def rmsnorm_forward(self, hidden_states):
    if not exact_enabled():
        original = getattr(rmsnorm_forward, "_unirl_parity_original", None)
        if not callable(original):
            raise RuntimeError("Qwen3 parity RMSNorm original is unavailable")
        return original(self, hidden_states)
    return _RmsNorm.apply(
        hidden_states,
        self.weight,
        float(self.variance_epsilon),
    )


__all__ = ["rmsnorm_forward"]
