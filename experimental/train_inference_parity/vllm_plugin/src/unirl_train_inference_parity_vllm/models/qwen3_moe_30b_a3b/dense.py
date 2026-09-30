"""Public-reference QKV, o_proj and LM-head providers."""

from __future__ import annotations

import types

import torch
import torch.nn as nn

from ...common.providers import linear

_PARITY_O_PROJECTION_CLASSES: dict[type, type] = {}


def _use_direct_layerwise_reload(layer) -> None:
    """Keep the custom o-proj storage live and load it without meta staging.

    The parity projection replaces vLLM's row-parallel parameter with a
    column-parallel one of equal element count but different shape. vLLM's
    generic meta-staged reload cannot infer that custom layout and copied the
    local q-projection shard into this storage. Skipping meta staging lets the
    bound loader write the canonical full o-proj directly into its CuMem
    weights-pool allocation.
    """
    from vllm.model_executor.model_loader.reload.meta import SKIP_MODULES

    base_class = type(layer)
    parity_class = _PARITY_O_PROJECTION_CLASSES.get(base_class)
    if parity_class is None:
        parity_class = type("UniRLParityOProjection", (base_class,), {})
        _PARITY_O_PROJECTION_CLASSES[base_class] = parity_class
    layer.__class__ = parity_class
    SKIP_MODULES.add(parity_class.__name__)


def _column_weight_loader(self, parameter, loaded_weight):
    rank = int(self._unirl_parity_tp_rank)
    world = int(self._unirl_parity_tp_world)
    rows = int(loaded_weight.shape[0])
    shard = rows // world
    loaded_shard = loaded_weight.narrow(0, rank * shard, shard).to(
        device=parameter.device,
        dtype=parameter.dtype,
    )
    parameter.data.copy_(loaded_shard)


def _reshape_row_to_column(layer) -> None:
    from vllm.distributed import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_world_size,
    )

    rank = get_tensor_model_parallel_rank()
    world = get_tensor_model_parallel_world_size()
    _use_direct_layerwise_reload(layer)
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
    from vllm.model_executor.utils import set_weight_attrs

    layer._unirl_parity_tp_rank = int(rank)
    layer._unirl_parity_tp_world = int(world)
    loader = types.MethodType(_column_weight_loader, layer)
    set_weight_attrs(replacement, {"weight_loader": loader})
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
