"""Activate only vLLM's attention batch-invariance gates."""

from __future__ import annotations

import importlib

_INSTALLED = False


class _EnvProxy:
    def __init__(self, original):
        self._original = original

    def __getattr__(self, name):
        if name == "VLLM_BATCH_INVARIANT":
            return True
        return getattr(self._original, name)


def install_attention_contract() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    names = (
        "vllm.model_executor.layers.attention.attention",
        "vllm.v1.attention.backends.flash_attn",
        "vllm.v1.attention.backends.fa_utils",
        "vllm.v1.attention.ops.triton_unified_attention",
    )
    for name in names:
        module = importlib.import_module(name)
        if hasattr(module, "vllm_is_batch_invariant"):
            module.vllm_is_batch_invariant = lambda: True
        if name.endswith("triton_unified_attention") and hasattr(
            module,
            "is_batch_invariant",
        ):
            module.is_batch_invariant = True
        if name.endswith("flash_attn") and hasattr(module, "envs"):
            module.envs = _EnvProxy(module.envs)
    _INSTALLED = True


__all__ = ["install_attention_contract"]
