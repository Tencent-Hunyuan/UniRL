"""Correctness-first Qwen3-MoE path using vLLM BI linear and Torch combine."""

from __future__ import annotations

import types

import torch
import torch.nn.functional as F

from ...common.providers import linear
from .router import install_gate, install_hf_router


def _combine(
    c_rows: torch.Tensor,
    row_map: torch.Tensor,
) -> torch.Tensor:
    tokens, topk = row_map.shape
    hidden = int(c_rows.shape[-1])
    total = torch.zeros((tokens, hidden), dtype=torch.float32, device=c_rows.device)
    zero = torch.zeros_like(total)
    for slot in range(topk):
        rows = row_map[:, slot].long()
        contribution = c_rows.index_select(0, rows.clamp_min(0)).float()
        total = total + torch.where((rows >= 0)[:, None], contribution, zero)
    total = total.to(torch.bfloat16).float()
    return total.to(torch.bfloat16)


def _get_w2_column(experts) -> torch.Tensor:
    cached = getattr(experts, "_unirl_parity_w2_column", None)
    if cached is not None:
        return cached
    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
        get_tp_group,
    )

    world = get_tensor_model_parallel_world_size()
    rank = get_tensor_model_parallel_rank()
    stock = experts.w2_weight
    full = stock if world == 1 else get_tp_group().all_gather(stock.contiguous(), dim=-1)
    hidden_local = int(full.shape[1]) // world
    cached = full[:, rank * hidden_local : (rank + 1) * hidden_local].contiguous()
    experts._unirl_parity_w2_column = cached
    return cached


def _install_reload_invalidation(experts) -> None:
    original = experts.weight_loader
    if getattr(original, "_unirl_parity_wrapper", False):
        return

    def weight_loader(self, param, *args, **kwargs):
        result = original(param, *args, **kwargs)
        if param is self.w2_weight:
            self._unirl_parity_w2_column = None
        return result

    weight_loader._unirl_parity_wrapper = True
    experts.weight_loader = types.MethodType(weight_loader, experts)
    for parameter in (experts.w13_weight, experts.w2_weight):
        parameter.weight_loader = experts.weight_loader


def _expert_forward(block, hidden_states: torch.Tensor) -> torch.Tensor:
    from vllm.distributed import (
        get_tensor_model_parallel_world_size,
        get_tp_group,
    )
    from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (
        moe_permute,
    )

    input_was_1d = hidden_states.dim() == 1
    hidden = hidden_states.reshape(-1, hidden_states.shape[-1]).contiguous()
    router_logits, _ = block.gate(hidden)
    topk_weights, topk_ids = block.experts.router.select_experts(
        hidden_states=hidden,
        router_logits=router_logits,
    )
    num_experts = int(block.n_routed_experts)
    permuted, _scale, offsets, inverse, _indices = moe_permute(
        hidden,
        None,
        topk_ids.to(torch.int32),
        num_experts,
    )
    counts = (offsets[1:] - offsets[:-1]).detach().cpu().tolist()
    world = get_tensor_model_parallel_world_size()
    w2_column = _get_w2_column(block.experts)
    activations = []
    outputs = []
    offset = 0
    for expert, count_value in enumerate(counts):
        count = int(count_value)
        if count == 0:
            continue
        rows = permuted[offset : offset + count].contiguous()
        offset += count
        local_gate_up = linear(
            rows,
            block.experts.w13_weight[expert],
            None,
        )
        gathered = local_gate_up if world == 1 else get_tp_group().all_gather(local_gate_up.contiguous(), dim=-1)
        local_intermediate = int(local_gate_up.shape[-1]) // 2
        rank_packed = gathered.view(count, world, 2, local_intermediate)
        gate = rank_packed[:, :, 0].reshape(count, -1)
        up = rank_packed[:, :, 1].reshape(count, -1)
        activation = F.silu(gate) * up
        local_down = linear(
            activation.contiguous(),
            w2_column[expert],
            None,
        )
        down = local_down if world == 1 else get_tp_group().all_gather(local_down.contiguous(), dim=-1)
        activations.append(activation)
        outputs.append(down)
    if not outputs:
        return torch.zeros_like(hidden)
    expert_output = torch.cat(outputs, dim=0)
    slots = int(topk_ids.numel())
    permuted_weights = torch.zeros(
        slots,
        dtype=topk_weights.dtype,
        device=topk_weights.device,
    )
    permuted_weights[inverse.long()] = topk_weights.reshape(-1)
    contributions = (expert_output * permuted_weights[:, None]).to(torch.bfloat16)
    result = _combine(
        contributions,
        inverse.view_as(topk_ids).to(torch.int32),
    )
    return result.squeeze(0) if input_was_1d else result


def make_moe_block(base_class):
    class ParityMoeBlock(base_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            install_gate(self.gate)
            install_hf_router(self.experts)
            _install_reload_invalidation(self.experts)

        def forward(self, hidden_states):
            return _expert_forward(self, hidden_states)

        def _unirl_before_sleep(self):
            self.experts._unirl_parity_w2_column = None

    ParityMoeBlock.__name__ = "ParityQwen3MoeSparseMoeBlock"
    return ParityMoeBlock


__all__ = ["make_moe_block"]
