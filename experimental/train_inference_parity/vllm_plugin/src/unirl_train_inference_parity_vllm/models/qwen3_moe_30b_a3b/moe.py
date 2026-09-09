"""Correctness-first Qwen3-MoE path using vLLM BI linear and Torch combine."""

from __future__ import annotations

import types

import torch
import torch.nn.functional as F

from ...common.moe_combine import moe_combine
from ...common.providers import linear
from .router import install_gate, install_hf_router


def _validate_tp_routing(
    counts: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    num_experts: int,
    group,
    world: int,
) -> None:
    """Collectively fail before route-dependent expert collectives diverge."""
    shape = tuple(topk_ids.shape)
    counts_valid = bool(
        counts.dim() == 1
        and counts.numel() == num_experts
        and torch.all(counts >= 0).item()
        and int(counts.sum().item()) == int(topk_ids.numel())
    )
    ids_valid = bool(
        topk_ids.dim() == 2
        and (topk_ids.numel() == 0 or (int(topk_ids.min().item()) >= 0 and int(topk_ids.max().item()) < num_experts))
    )
    if world == 1:
        if not counts_valid or not ids_valid:
            raise RuntimeError("Qwen3-MoE parity routing metadata is invalid")
        return
    metadata = torch.tensor(
        (
            topk_ids.dim(),
            topk_ids.numel(),
            shape[0] if len(shape) > 0 else -1,
            shape[1] if len(shape) > 1 else -1,
            counts.numel(),
            num_experts,
            int(counts_valid),
            int(ids_valid),
        ),
        dtype=torch.int64,
        device=topk_ids.device,
    )
    gathered_metadata = group.all_gather(metadata.contiguous(), dim=0).view(
        world,
        metadata.numel(),
    )
    reference_metadata = gathered_metadata[0]
    metadata_matches = bool(torch.all(gathered_metadata == reference_metadata).item())
    metadata_valid = bool(
        torch.all(gathered_metadata[:, 0] == 2).item()
        and torch.all(gathered_metadata[:, 1] == gathered_metadata[:, 2] * gathered_metadata[:, 3]).item()
        and torch.all(gathered_metadata[:, 4] == num_experts).item()
        and torch.all(gathered_metadata[:, 5] == num_experts).item()
        and torch.all(gathered_metadata[:, 6] == 1).item()
        and torch.all(gathered_metadata[:, 7] == 1).item()
    )
    if not metadata_matches or not metadata_valid:
        observed = gathered_metadata.detach().cpu().tolist()
        raise RuntimeError(f"Qwen3-MoE parity TP routing metadata mismatch before expert collectives: {observed}")

    payload = torch.cat(
        (
            counts.detach().to(device=topk_ids.device, dtype=torch.int64),
            topk_ids.detach().reshape(-1).to(dtype=torch.int64),
        )
    ).contiguous()
    gathered_payload = group.all_gather(payload, dim=0).view(world, payload.numel())
    mismatch_ranks = (
        torch.any(gathered_payload != gathered_payload[0], dim=1).nonzero().flatten().detach().cpu().tolist()
    )
    if mismatch_ranks:
        raise RuntimeError(
            "Qwen3-MoE parity TP routing mismatch before expert collectives; "
            f"counts/topk_ids differ on ranks {mismatch_ranks}"
        )


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
    counts_tensor = offsets[1:] - offsets[:-1]
    world = get_tensor_model_parallel_world_size()
    group = get_tp_group()
    _validate_tp_routing(
        counts_tensor,
        topk_ids,
        num_experts=num_experts,
        group=group,
        world=world,
    )
    counts = counts_tensor.detach().cpu().tolist()
    w2_column = _get_w2_column(block.experts)
    slots = int(topk_ids.numel())
    if slots == 0:
        return torch.zeros_like(hidden)
    max_count = max(int(value) for value in counts)
    local_gate_width = int(block.experts.w13_weight.shape[1])
    local_intermediate = local_gate_width // 2
    padded_gate_up = []
    offset = 0
    for expert, count_value in enumerate(counts):
        count = int(count_value)
        rows = permuted[offset : offset + count].contiguous()
        offset += count
        if count == 0:
            local_gate_up = hidden.new_zeros((max_count, local_gate_width))
        else:
            active = linear(
                rows,
                block.experts.w13_weight[expert],
                None,
            )
            padding = hidden.new_zeros((max_count - count, local_gate_width))
            local_gate_up = torch.cat((active, padding), dim=0)
        padded_gate_up.append(local_gate_up)

    local_gate_up = torch.stack(padded_gate_up, dim=0).contiguous()
    gathered_gate_up = local_gate_up if world == 1 else group.all_gather(local_gate_up, dim=-1)
    rank_packed = gathered_gate_up.view(
        num_experts,
        max_count,
        world,
        2,
        local_intermediate,
    )
    gate = rank_packed[:, :, :, 0].reshape(num_experts, max_count, -1)
    up = rank_packed[:, :, :, 1].reshape(num_experts, max_count, -1)
    activation = F.silu(gate) * up

    local_hidden_width = int(w2_column.shape[1])
    padded_down = []
    for expert, count_value in enumerate(counts):
        if int(count_value) == 0:
            local_down = hidden.new_zeros((max_count, local_hidden_width))
        else:
            local_down = linear(
                activation[expert].contiguous(),
                w2_column[expert],
                None,
            )
        padded_down.append(local_down)
    local_down = torch.stack(padded_down, dim=0).contiguous()
    gathered_down = local_down if world == 1 else group.all_gather(local_down, dim=-1)
    expert_output = torch.cat(
        [gathered_down[expert, : int(count)] for expert, count in enumerate(counts) if int(count) > 0],
        dim=0,
    )
    permuted_weights = torch.zeros(
        slots,
        dtype=topk_weights.dtype,
        device=topk_weights.device,
    )
    permuted_weights[inverse.long()] = topk_weights.reshape(-1)
    contributions = (expert_output * permuted_weights[:, None]).to(torch.bfloat16)
    result = moe_combine(
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
