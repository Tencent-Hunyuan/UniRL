"""Resolve an EP model's fused expert layout into a weight-export transform."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import torch
from torch import nn

from unirl.train.backend.base import ExpertWeightExportTransform, NamedTensorIterator
from unirl.train.backend.veomni.ep.models import hi3, qwen3_moe
from unirl.train.backend.veomni.ep.placement import (
    ep_named_parameters,
    gather_stacked_expert_block,
)


class ExpertExportLayout(Protocol):
    """Module contract for exporting one model family's fused expert weights."""

    is_fused_expert_param: Callable[[str], bool]
    iter_hf_expert_tensors: Callable[[str, torch.Tensor], NamedTensorIterator]


_EXPERT_EXPORT_LAYOUTS_BY_MODEL_TYPE: dict[str, ExpertExportLayout] = {
    "qwen3_moe": qwen3_moe,
    "qwen3_5_moe": qwen3_moe,
    "qwen3_vl_moe": qwen3_moe,
    "qwen3_omni_moe": qwen3_moe,
    "hunyuan_image_3": hi3,
    "hunyuan_image3": hi3,
    "hunyuan": hi3,
    "hunyuan_vl": hi3,
}


def _infer_layout_from_ep_params(
    model: nn.Module,
    ep_params: tuple[tuple[str, nn.Parameter], ...],
) -> ExpertExportLayout | None:
    """Fallback layout inference when a bare decoder submodule has no ``config.model_type``."""
    names = [name for name, _ in ep_params]
    if not names:
        return None
    has_hi3_modules = any(
        type(m).__name__ in ("FusedHunyuanMoE", "FusedExperts", "HunyuanMoE") for m in model.modules()
    )
    if has_hi3_modules or any(".experts.gate_and_up_proj" in name for name in names):
        if all(hi3.is_fused_expert_param(name) for name in names):
            return hi3
    if all(qwen3_moe.is_fused_expert_param(name) for name in names):
        return qwen3_moe
    if all(hi3.is_fused_expert_param(name) for name in names):
        return hi3
    return None


def resolve_expert_weight_export_transform(model: nn.Module) -> ExpertWeightExportTransform:
    """Resolve the transform exporting this model's EP-sharded expert weights."""
    ep_params = ep_named_parameters(model)
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    layout = _EXPERT_EXPORT_LAYOUTS_BY_MODEL_TYPE.get(model_type)
    if layout is None:
        layout = _infer_layout_from_ep_params(model, ep_params)
    if layout is None:
        raise ValueError(
            f"EP experts: no fused expert layout is registered for model_type={model_type!r} "
            f"(known: {sorted(_EXPERT_EXPORT_LAYOUTS_BY_MODEL_TYPE)}). "
            "Register one under ep/models/ or sync this model's "
            "adapter instead of its full weights."
        )

    unsupported = [name for name, _ in ep_params if not layout.is_fused_expert_param(name)]
    if unsupported:
        raise ValueError(
            f"EP experts: the {model_type!r} layout does not describe {len(unsupported)} "
            f"EP-sharded parameter(s): {unsupported[:4]}."
        )

    from unirl.train.backend.veomni import _compat

    _compat.ensure_installed()
    from veomni.distributed.parallel_state import get_parallel_state

    ps = get_parallel_state()
    if not ps.ep_enabled:
        raise RuntimeError("EP weight export: the model has EP-sharded parameters, but EP state is disabled.")
    ep_size = ps.ep_size
    ep_group = ps.ep_group

    def _transform(stream: NamedTensorIterator) -> NamedTensorIterator:
        for name, tensor in stream:
            if not layout.is_fused_expert_param(name):
                yield name, tensor
                continue
            stacked = gather_stacked_expert_block(tensor, ep_size=ep_size, ep_group=ep_group)
            del tensor
            yield from layout.iter_hf_expert_tensors(name, stacked)
            del stacked

    return _transform


__all__ = ["resolve_expert_weight_export_transform"]
