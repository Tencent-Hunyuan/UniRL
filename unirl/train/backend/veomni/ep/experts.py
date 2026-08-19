"""Resolve an EP model's fused expert layout into a tensor-stream transform (contract: unirl/train/readme.md)."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from functools import partial
from types import ModuleType

import torch
from torch import nn

from unirl.train.backend.veomni.ep.models import qwen3_moe
from unirl.train.backend.veomni.ep.placement import (
    ep_named_parameters,
    gather_stacked_expert_block,
    has_ep_params,
)

TensorStream = Iterator[tuple[str, torch.Tensor]]

# model_type -> the ep/models module describing the fused expert tensors VeOmni builds for it.
_LAYOUTS: dict[str, ModuleType] = {"qwen3_moe": qwen3_moe, "qwen3_5_moe": qwen3_moe}


def resolve_expert_expander(model: nn.Module) -> Callable[[TensorStream], TensorStream] | None:
    """Return the transform re-emitting this model's fused experts per expert, or None when it shards none."""
    if not has_ep_params(model):
        return None

    model_type = getattr(getattr(model, "config", None), "model_type", None)
    layout = _LAYOUTS.get(model_type)
    if layout is None:
        raise ValueError(
            f"EP experts: no fused expert layout is registered for model_type={model_type!r} "
            f"(known: {sorted(_LAYOUTS)}). Register one under ep/models/ or sync this model's "
            "adapter instead of its full weights."
        )

    unsupported = [name for name, _ in ep_named_parameters(model) if not layout.is_fused_expert_param(name)]
    if unsupported:
        raise ValueError(
            f"EP experts: the {model_type!r} layout does not describe {len(unsupported)} "
            f"EP-sharded parameter(s): {unsupported[:4]}."
        )
    return partial(_expand_experts, layout=layout)


def _expand_experts(stream: TensorStream, *, layout: ModuleType) -> TensorStream:
    """All-gather each rank's fused expert block over the EP group, then split it into per-expert tensors."""
    from unirl.train.backend.veomni import _compat

    _compat.ensure_installed()
    from veomni.distributed.parallel_state import get_parallel_state

    ps = get_parallel_state()
    ep_size = int(ps.ep_size) if getattr(ps, "ep_enabled", False) else 1
    ep_group = ps.ep_group if ep_size > 1 else None

    for name, tensor in stream:
        if not layout.is_fused_expert_param(name):
            yield name, tensor
            continue
        stacked = tensor  # ep_size == 1: this rank already holds the full [E, ...] stack
        if ep_size > 1:
            stacked = gather_stacked_expert_block(tensor, ep_size=ep_size, ep_group=ep_group)
            del tensor
        yield from layout.iter_hf_expert_tensors(name, stacked)
        del stacked


__all__ = ["TensorStream", "resolve_expert_expander"]
