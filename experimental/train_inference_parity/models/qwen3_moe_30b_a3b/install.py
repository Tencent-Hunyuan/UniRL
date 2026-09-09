"""Explicit installer for process-wide Qwen3 parity overrides."""

from __future__ import annotations

import inspect
import sys
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any

from unirl.models.qwen3.ar import register_exact_actor_provider

from .aten import preflight_aten, register_aten
from .attention import fa3_available, make_flash_attn_interface, probe_fa3_backend
from .context import exact_context, require_parity_opt_in
from .linear_ops import exact_log_softmax
from .moe import sparse_moe_forward
from .norm import rmsnorm_forward
from .rope import rotary_forward
from .router import router_forward

_INSTALLED = False


@dataclass
class _AttributePatch:
    target: Any
    name: str
    existed_directly: bool
    original: Any


@dataclass
class _MappingPatch:
    target: MutableMapping
    key: Any
    existed: bool
    original: Any


_PYTHON_PATCHES: list[_AttributePatch | _MappingPatch] = []


def _patch_attribute(target: Any, name: str, replacement: Any) -> None:
    _PYTHON_PATCHES.append(
        _AttributePatch(
            target=target,
            name=name,
            existed_directly=name in vars(target),
            original=getattr(target, name, None),
        )
    )
    setattr(target, name, replacement)


def _patch_mapping(target: MutableMapping, key: Any, replacement: Any) -> None:
    _PYTHON_PATCHES.append(
        _MappingPatch(
            target=target,
            key=key,
            existed=key in target,
            original=target.get(key),
        )
    )
    target[key] = replacement


def _clear_cache(value: Any) -> None:
    clear = getattr(value, "cache_clear", None)
    if callable(clear):
        clear()


def _install_fa3_python_patches() -> None:
    probe_fa3_backend()
    _patch_mapping(sys.modules, "flash_attn_interface", make_flash_attn_interface())

    import transformers.modeling_flash_attention_utils as flash_utils
    import transformers.modeling_utils as modeling_utils
    import transformers.utils as transformers_utils
    import transformers.utils.import_utils as import_utils

    previous = getattr(import_utils, "is_flash_attn_3_available", None)
    _clear_cache(previous)

    def available() -> bool:
        return fa3_available()

    available.cache_clear = lambda: None
    _patch_attribute(import_utils, "is_flash_attn_3_available", available)
    _patch_attribute(transformers_utils, "is_flash_attn_3_available", available)
    _patch_attribute(flash_utils, "is_flash_attn_3_available", available)

    matrix = modeling_utils.FLASH_ATTENTION_COMPATIBILITY_MATRIX[3]
    _patch_mapping(matrix, "general_availability_check", available)
    _patch_mapping(matrix, "pkg_availability_check", lambda *args, **kwargs: fa3_available())


def _install_qwen_python_patches(modeling) -> None:
    setattr(router_forward, "_unirl_parity_original", modeling.Qwen3MoeTopKRouter.forward)
    setattr(rmsnorm_forward, "_unirl_parity_original", modeling.Qwen3MoeRMSNorm.forward)
    setattr(sparse_moe_forward, "_unirl_parity_original", modeling.Qwen3MoeSparseMoeBlock.forward)
    setattr(rotary_forward, "_unirl_parity_original", modeling.Qwen3MoeRotaryEmbedding.forward)
    _patch_attribute(modeling.Qwen3MoeTopKRouter, "forward", router_forward)
    _patch_attribute(modeling.Qwen3MoeRMSNorm, "forward", rmsnorm_forward)
    _patch_attribute(modeling.Qwen3MoeSparseMoeBlock, "forward", sparse_moe_forward)
    _patch_attribute(modeling.Qwen3MoeRotaryEmbedding, "forward", rotary_forward)


def restore_python_patches() -> None:
    """Restore Python objects while leaving irreversible ATen registrations active."""
    global _INSTALLED
    while _PYTHON_PATCHES:
        patch = _PYTHON_PATCHES.pop()
        if isinstance(patch, _MappingPatch):
            if patch.existed:
                patch.target[patch.key] = patch.original
            else:
                patch.target.pop(patch.key, None)
            continue
        if patch.existed_directly:
            setattr(patch.target, patch.name, patch.original)
        else:
            delattr(patch.target, patch.name)
        _clear_cache(patch.original)
    _INSTALLED = False


def install() -> None:
    """Install the adapter after an explicit dedicated-process opt-in."""
    global _INSTALLED
    require_parity_opt_in()
    if _INSTALLED:
        return

    # Probe all external Python surfaces before the irreversible ATen step.
    from unirl_train_inference_parity_vllm.common.providers import preflight_providers
    from unirl_train_inference_parity_vllm.compat import validate_runtime_versions

    validate_runtime_versions()
    preflight_providers(strict=True)
    probe_fa3_backend()
    from transformers.models.qwen3_moe import modeling_qwen3_moe as modeling

    method_contracts = {
        "Qwen3MoeTopKRouter": ("self", "hidden_states"),
        "Qwen3MoeRMSNorm": ("self", "hidden_states"),
        "Qwen3MoeSparseMoeBlock": ("self", "hidden_states"),
        "Qwen3MoeRotaryEmbedding": ("self", "x", "position_ids"),
    }
    for class_name, expected_parameters in method_contracts.items():
        if not hasattr(modeling, class_name):
            raise RuntimeError(f"transformers Qwen3 MoE class is missing: {class_name}")
        method = getattr(getattr(modeling, class_name), "forward", None)
        if not callable(method):
            raise RuntimeError(f"transformers Qwen3 MoE class has no callable forward: {class_name}")
        actual_parameters = tuple(inspect.signature(method).parameters)
        if actual_parameters != expected_parameters:
            raise RuntimeError(
                f"transformers Qwen3 MoE signature drift for {class_name}.forward: "
                f"expected {expected_parameters}, got {actual_parameters}"
            )
    preflight_aten()

    try:
        _install_fa3_python_patches()
        _install_qwen_python_patches(modeling)
        register_aten()
        register_exact_actor_provider(
            context_factory=exact_context,
            log_softmax=exact_log_softmax,
        )
    except Exception:
        restore_python_patches()
        raise
    _INSTALLED = True


def installed() -> bool:
    return _INSTALLED


__all__ = ["install", "installed", "restore_python_patches"]
