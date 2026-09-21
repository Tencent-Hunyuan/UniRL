"""Public-reference QKV, o_proj and LM-head providers."""

from __future__ import annotations

import types

import torch
import torch.nn as nn

from ...common.providers import linear


def _column_weight_loader(rank: int, world: int):
    def load(parameter, loaded_weight):
        rows = int(loaded_weight.shape[0])
        shard = rows // world
        parameter.data.copy_(
            loaded_weight.narrow(0, rank * shard, shard).to(
                device=parameter.device,
                dtype=parameter.dtype,
            )
        )

    return load


def _reshape_row_to_column(layer) -> None:
    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )

    rank = get_tensor_model_parallel_rank()
    world = get_tensor_model_parallel_world_size()
    old = layer.weight
    replacement = nn.Parameter(
        torch.empty(
            layer.output_size // world,
            layer.input_size,
            device=old.device,
            dtype=old.dtype,
        ),
        requires_grad=False,
    )
    replacement.weight_loader = _column_weight_loader(rank, world)
    layer.weight = replacement
    layer._unirl_parity_output_size = layer.output_size


def _qkv_forward(self, hidden):
    return linear(hidden, self.weight, None), None


def _o_forward(self, hidden):
    from vllm.distributed import get_tp_group

    group = get_tp_group()
    full_input = group.all_gather(hidden.contiguous(), dim=-1)
    local_output = linear(full_input, self.weight, None)
    output = local_output if group.world_size == 1 else group.all_gather(local_output.contiguous(), dim=-1)
    return output, None


def make_attention(base_class):
    class ParityAttention(base_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.qkv_proj.forward = types.MethodType(
                _qkv_forward,
                self.qkv_proj,
            )
            _reshape_row_to_column(self.o_proj)
            self.o_proj.forward = types.MethodType(
                _o_forward,
                self.o_proj,
            )

    ParityAttention.__name__ = "ParityQwen3MoeAttention"
    return ParityAttention


def make_logits_processor(base_class):
    class ParityLogitsProcessor(base_class):
        def _get_logits(self, hidden_states, lm_head, embedding_bias=None):
            if embedding_bias is not None:
                raise ValueError("Qwen3-MoE parity LM head does not support bias")
            from vllm.distributed import get_tp_group

            group = get_tp_group()
            local_logits = linear(hidden_states, lm_head.weight, None)
            logits = local_logits if group.world_size == 1 else group.all_gather(local_logits.contiguous(), dim=-1)
            return logits[..., : self.org_vocab_size]

    ParityLogitsProcessor.__name__ = "ParityQwen3MoeLogitsProcessor"
    return ParityLogitsProcessor


__all__ = ["make_attention", "make_logits_processor"]
