"""Resolve an EP model's fused expert layout into a weight-export transform."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import partial

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


@dataclass(frozen=True)
class ExpertExportLayout:
    """Operations required to export one model family's fused expert weights."""

    is_fused_weight: Callable[[str], bool]
    iter_hf_weights: Callable[[str, torch.Tensor], NamedTensorIterator]


_QWEN3_MOE_EXPORT_LAYOUT = ExpertExportLayout(
    is_fused_weight=qwen3_moe.is_fused_expert_param,
    iter_hf_weights=qwen3_moe.iter_hf_expert_tensors,
)
_EXPERT_EXPORT_LAYOUTS_BY_MODEL_TYPE = {
    "qwen3_moe": _QWEN3_MOE_EXPORT_LAYOUT,
    "qwen3_5_moe": _QWEN3_MOE_EXPORT_LAYOUT,
}


def resolve_expert_weight_export_transform(
    model: nn.Module,
    *,
    expected_ep_size: int,
) -> ExpertWeightExportTransform | None:
    """Resolve the transform exporting this model's EP-sharded expert weights."""
    if not has_ep_params(model):
        if expected_ep_size > 1:
            raise RuntimeError(
                f"EP weight export: backend configured ep_size={expected_ep_size}, "
                "but the model has no EP-sharded parameters."
            )
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

    unsupported = [name for name, _ in ep_named_parameters(model) if not layout.is_fused_weight(name)]
    if unsupported:
        raise ValueError(
            f"EP experts: the {model_type!r} layout does not describe {len(unsupported)} "
            f"EP-sharded parameter(s): {unsupported[:4]}."
        )
    if expected_ep_size <= 1:
        raise RuntimeError(
            f"EP weight export: the model has EP-sharded parameters, but backend ep_size={expected_ep_size}."
        )

    from unirl.train.backend.veomni import _compat

    _compat.ensure_installed()
    from veomni.distributed.parallel_state import get_parallel_state

    ps = get_parallel_state()
    if not getattr(ps, "ep_enabled", False):
        raise RuntimeError("EP weight export: the model has EP-sharded parameters, but EP state is disabled.")
    ep_size = int(ps.ep_size)
    if ep_size != expected_ep_size:
        raise RuntimeError(
            f"EP weight export: backend ep_size={expected_ep_size} does not match parallel-state ep_size={ep_size}."
        )
    return partial(
        _iter_exported_expert_weights,
        layout=layout,
        ep_size=ep_size,
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
        if not layout.is_fused_weight(name):
            yield name, tensor
            continue
        stacked = gather_stacked_expert_block(tensor, ep_size=ep_size, ep_group=ep_group)
        del tensor
        yield from layout.iter_hf_weights(name, stacked)
        del stacked


__all__ = [
    "ExpertWeightExportTransform",
    "NamedTensorIterator",
    "resolve_expert_weight_export_transform",
]
