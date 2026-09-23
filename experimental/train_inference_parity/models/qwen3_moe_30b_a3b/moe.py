"""Exact selected-token MoE forward with a Torch surrogate backward."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .context import exact_enabled
from .contract import require_train_tp, validate_parallel_dimensions
from .linear_ops import providers


def _dispatch(hidden, topk_ids, experts):
    tokens, topk = topk_ids.shape
    flat_ids = topk_ids.reshape(-1).long()
    order = torch.argsort(flat_ids, stable=True)
    rows = (
        hidden[:, None, :]
        .expand(tokens, topk, hidden.shape[-1])
        .reshape(-1, hidden.shape[-1])
        .index_select(0, order)
        .contiguous()
    )
    counts = torch.bincount(flat_ids, minlength=experts)
    return rows, counts, order


def _grouped_linear(inputs, weights, counts, *, public):
    outputs = []
    offset = 0
    for expert in range(int(weights.shape[0])):
        count = int(counts[expert].item())
        if count:
            rows = inputs[offset : offset + count].contiguous()
            outputs.append(
                providers().linear(rows, weights[expert], None) if public else F.linear(rows, weights[expert])
            )
        offset += count
    return torch.cat(outputs, dim=0)


def _combine(contributions, order, tokens, topk):
    from unirl_train_inference_parity_vllm.common.moe_combine import moe_combine

    row_map = torch.empty(tokens * topk, dtype=torch.long, device=order.device)
    row_map[order] = torch.arange(tokens * topk, device=order.device)
    return moe_combine(contributions, row_map.view(tokens, topk))


def _validate_bfloat16(*tensors: torch.Tensor) -> None:
    invalid = [str(tensor.dtype) for tensor in tensors if tensor.dtype != torch.bfloat16]
    if invalid:
        raise ValueError(f"Qwen3 parity MoE requires BF16 tensors, got {invalid}")


def _moe_forward(hidden, topk_ids, topk_weights, w13, w2, *, public):
    _validate_bfloat16(hidden, w13, w2)
    tokens, hidden_size = hidden.shape
    topk = int(topk_ids.shape[1])
    experts = int(w13.shape[0])
    if int(w13.shape[1]) % 2:
        raise ValueError(f"Qwen3 parity gate/up width must be even, got {int(w13.shape[1])}")
    intermediate = int(w13.shape[1]) // 2
    if tuple(w2.shape) != (experts, hidden_size, intermediate):
        raise ValueError(
            "Qwen3 parity down-projection shape mismatch: "
            f"got {tuple(w2.shape)}, expected {(experts, hidden_size, intermediate)}"
        )
    train_tp = require_train_tp()
    validate_parallel_dimensions(hidden_size, intermediate, train_tp)
    dispatched, counts, order = _dispatch(hidden, topk_ids, experts)
    local_intermediate = intermediate // train_tp
    fc1_parts = []
    for rank in range(train_tp):
        begin, end = rank * local_intermediate, (rank + 1) * local_intermediate
        rank_weight = torch.cat(
            (w13[:, begin:end], w13[:, intermediate + begin : intermediate + end]),
            dim=1,
        ).contiguous()
        fc1_parts.append(_grouped_linear(dispatched, rank_weight, counts, public=public))
    packed = torch.cat(fc1_parts, dim=-1).view(-1, train_tp, 2, local_intermediate)
    activation = (F.silu(packed[:, :, 0]) * packed[:, :, 1]).reshape(-1, intermediate)
    local_hidden = hidden_size // train_tp
    fc2_parts = []
    for rank in range(train_tp):
        rank_weight = w2[
            :,
            rank * local_hidden : (rank + 1) * local_hidden,
            :,
        ].contiguous()
        fc2_parts.append(_grouped_linear(activation, rank_weight, counts, public=public))
    fc2 = torch.cat(fc2_parts, dim=-1)
    sorted_weights = topk_weights.reshape(-1).index_select(0, order)
    contributions = (fc2 * sorted_weights[:, None]).to(torch.bfloat16)
    return _combine(contributions, order, tokens, topk)


class _Moe(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, topk_ids, topk_weights, w13, w2):
        ctx.save_for_backward(hidden, topk_ids, topk_weights, w13, w2)
        return _moe_forward(
            hidden,
            topk_ids,
            topk_weights,
            w13,
            w2,
            public=True,
        )

    @staticmethod
    def backward(ctx, grad_output):
        hidden, topk_ids, topk_weights, w13, w2 = ctx.saved_tensors
        needs = ctx.needs_input_grad
        sources = (hidden, topk_weights, w13, w2)
        rebuilt = [
            tensor.detach().requires_grad_(needed)
            for tensor, needed in zip(
                sources,
                (needs[0], needs[2], needs[3], needs[4]),
                strict=True,
            )
        ]
        with torch.enable_grad():
            output = _moe_forward(
                rebuilt[0],
                topk_ids,
                rebuilt[1],
                rebuilt[2],
                rebuilt[3],
                public=False,
            )
            requested = [tensor for tensor in rebuilt if tensor.requires_grad]
            computed = torch.autograd.grad(
                output,
                requested,
                grad_output,
                allow_unused=True,
            )
        iterator = iter(computed)
        gradients = [next(iterator) if tensor.requires_grad else None for tensor in rebuilt]
        return gradients[0], None, gradients[1], gradients[2], gradients[3]


def sparse_moe_forward(self, hidden_states):
    if not exact_enabled():
        original = getattr(sparse_moe_forward, "_unirl_parity_original", None)
        if not callable(original):
            raise RuntimeError("Qwen3 parity MoE original is unavailable")
        return original(self, hidden_states)
    shape = hidden_states.shape
    hidden = hidden_states.reshape(-1, shape[-1])
    _logits, weights, indices = self.gate(hidden)
    output = _Moe.apply(
        hidden,
        indices,
        weights,
        self.experts.gate_up_proj,
        self.experts.down_proj,
    )
    return output.view(shape)


__all__ = ["sparse_moe_forward"]
