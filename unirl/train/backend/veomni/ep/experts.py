"""Resolve an EP model's fused expert layout into a weight-export transform."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from functools import partial
from typing import Protocol

import torch
from torch import nn

from unirl.train.backend.veomni.ep.models import qwen3_moe
from unirl.train.backend.veomni.ep.placement import (
    ep_named_parameters,
    gather_stacked_expert_block,
    has_ep_params,
)

NamedTensorIterator = Iterator[tuple[str, torch.Tensor]]
ExpertWeightExportTransform = Callable[[NamedTensorIterator], NamedTensorIterator]


class ExpertExportLayout(Protocol):
    """Module contract for exporting one model family's fused expert weights."""

    is_fused_expert_param: Callable[[str], bool]
    iter_hf_expert_tensors: Callable[[str, torch.Tensor], NamedTensorIterator]


_EXPERT_EXPORT_LAYOUTS_BY_MODEL_TYPE: dict[str, ExpertExportLayout] = {
    "qwen3_moe": qwen3_moe,
    "qwen3_5_moe": qwen3_moe,
}


def resolve_expert_weight_export_transform(model: nn.Module) -> ExpertWeightExportTransform | None:
    """Resolve the transform exporting this model's EP-sharded expert weights."""
    if not has_ep_params(model):
        return None

    model_type = getattr(getattr(model, "config", None), "model_type", None)
    layout = _EXPERT_EXPORT_LAYOUTS_BY_MODEL_TYPE.get(model_type)
    if layout is None:
        raise ValueError(
            f"EP experts: no fused expert layout is registered for model_type={model_type!r} "
            f"(known: {sorted(_EXPERT_EXPORT_LAYOUTS_BY_MODEL_TYPE)}). "
            "Register one under ep/models/ or sync this model's "
            "adapter instead of its full weights."
        )

    unsupported = [name for name, _ in ep_named_parameters(model) if not layout.is_fused_expert_param(name)]
    if unsupported:
        raise ValueError(
            f"EP experts: the {model_type!r} layout does not describe {len(unsupported)} "
            f"EP-sharded parameter(s): {unsupported[:4]}."
        )

    from unirl.train.backend.veomni import _compat

    _compat.ensure_installed()
    from veomni.distributed.parallel_state import get_parallel_state

    ps = get_parallel_state()
    if not getattr(ps, "ep_enabled", False):
        raise RuntimeError("EP weight export: the model has EP-sharded parameters, but EP state is disabled.")
    return partial(
        _iter_exported_expert_weights,
        layout=layout,
        ep_size=int(ps.ep_size),
        ep_group=ps.ep_group,
    )


def _iter_exported_expert_weights(
    stream: NamedTensorIterator,
    *,
    layout: ExpertExportLayout,
    ep_size: int,
    ep_group,
) -> NamedTensorIterator:
    """All-gather fused expert blocks and emit per-expert Hugging Face weights."""

    for name, tensor in stream:
        if not layout.is_fused_expert_param(name):
            yield name, tensor
            continue
        stacked = gather_stacked_expert_block(tensor, ep_size=ep_size, ep_group=ep_group)
        del tensor
        yield from layout.iter_hf_expert_tensors(name, stacked)
        del stacked


__all__ = [
    "ExpertWeightExportTransform",
    "resolve_expert_weight_export_transform",
]
