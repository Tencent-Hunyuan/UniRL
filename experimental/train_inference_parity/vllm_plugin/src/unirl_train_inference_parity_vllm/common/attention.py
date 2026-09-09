"""Activate only vLLM's attention batch-invariance gates."""

from __future__ import annotations

import importlib
import types

from ..compat import require_symbol, require_value
from ..registry import SymbolResult, symbol_result, value_result

_INSTALLED = False
_ENV_MODULES = (
    "vllm.model_executor.layers.attention.attention",
    "vllm.v1.attention.backends.flash_attn",
    "vllm.v1.attention.backends.fa_utils",
)
_TRITON_MODULE = "vllm.v1.attention.ops.triton_unified_attention"
_FLASH_MODULE = "vllm.v1.attention.backends.flash_attn"


class _EnvProxy:
    def __init__(self, original):
        self._original = original

    def __getattr__(self, name):
        if name == "VLLM_BATCH_INVARIANT":
            return True
        return getattr(self._original, name)


def _probe_fa3() -> tuple[object, type]:
    backend = require_symbol(
        _FLASH_MODULE,
        "FlashAttentionBackend",
        origin=f"{_FLASH_MODULE}.FlashAttentionBackend",
        strict=True,
    )
    get_version = require_symbol(
        _FLASH_MODULE,
        "get_flash_attn_version",
        parameters=(
            ("requires_alibi", "False"),
            ("head_size", "None"),
            ("head_size_v", "None"),
            ("has_sinks", "False"),
        ),
        origin="vllm.v1.attention.backends.fa_utils.get_flash_attn_version",
        strict=True,
    )
    if get_version() != 3:
        raise RuntimeError(f"parity requires vLLM FlashAttention 3, got {get_version()!r}")
    if backend.get_name() != "FLASH_ATTN" or backend.supports_batch_invariance() is not True:
        raise RuntimeError("vLLM FlashAttention backend does not satisfy the batch-invariant contract")
    implementation = backend.get_impl_cls()
    if (
        getattr(implementation, "__module__", None) != _FLASH_MODULE
        or getattr(implementation, "__name__", None) != "FlashAttentionImpl"
    ):
        raise RuntimeError(f"unexpected vLLM FlashAttention implementation {implementation!r}")
    facade = importlib.import_module("vllm.vllm_flash_attn")
    varlen = getattr(facade, "flash_attn_varlen_func", None)
    if not callable(varlen) or getattr(varlen, "__module__", None) != ("vllm.vllm_flash_attn.flash_attn_interface"):
        raise RuntimeError(f"unexpected vLLM FA3 varlen provider {varlen!r}")
    return varlen, implementation


def preflight_attention_contract(*, strict: bool) -> None:
    if not strict:
        raise ValueError("the parity attention contract requires strict=True")
    for module_name in _ENV_MODULES:
        envs = require_value(
            module_name,
            "envs",
            expected_type=types.ModuleType,
        )
        if envs.__name__ != "vllm.envs":
            raise RuntimeError(
                f"conflicting attention env provider at {module_name}.envs: expected vllm.envs, got {envs.__name__}"
            )
        require_value(
            module_name,
            "envs.VLLM_BATCH_INVARIANT",
            expected_type=bool,
        )
    require_value(
        _TRITON_MODULE,
        "is_batch_invariant",
        expected_type=bool,
    )
    _probe_fa3()


def install_attention_contract() -> tuple[SymbolResult, ...]:
    global _INSTALLED
    if _INSTALLED:
        raise RuntimeError("attention parity contract installed twice")
    fa3_varlen, fa3_implementation = _probe_fa3()
    results = []
    for module_name in _ENV_MODULES:
        module = importlib.import_module(module_name)
        original = module.envs
        proxy = _EnvProxy(original)
        module.envs = proxy
        results.append(
            symbol_result(
                f"{module_name}.envs",
                proxy,
                before=original,
                actual=module.envs,
            )
        )
    triton_module = importlib.import_module(_TRITON_MODULE)
    original_batch_invariant = triton_module.is_batch_invariant
    triton_module.is_batch_invariant = True
    results.append(
        value_result(
            f"{_TRITON_MODULE}.is_batch_invariant",
            "literal:true",
            before=original_batch_invariant,
            actual=triton_module.is_batch_invariant,
            verified=triton_module.is_batch_invariant is True,
        )
    )
    results.extend(
        (
            symbol_result(
                "vllm.vllm_flash_attn.flash_attn_varlen_func",
                fa3_varlen,
                before=fa3_varlen,
                actual=fa3_varlen,
            ),
            symbol_result(
                f"{_FLASH_MODULE}.FlashAttentionImpl",
                fa3_implementation,
                before=fa3_implementation,
                actual=fa3_implementation,
            ),
            value_result(
                f"{_FLASH_MODULE}.get_flash_attn_version()",
                "literal:3",
                before=3,
                actual=3,
                verified=True,
            ),
        )
    )
    _INSTALLED = True
    return tuple(results)


__all__ = ["install_attention_contract", "preflight_attention_contract"]
