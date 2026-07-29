"""Compatibility helpers for Qwen3-Omni on the pinned vLLM-Omni runtime."""

from __future__ import annotations

from functools import wraps
from typing import Any

_PATCH_SENTINEL = "_unirl_qwen3_omni_lora_compat"


def patch_qwen3_omni_thinker_class(model_cls: type[Any]) -> None:
    """Backport Qwen3-Omni Thinker LoRA support from vLLM-Omni #3915."""
    if getattr(model_cls, _PATCH_SENTINEL, False) or getattr(model_cls, "supports_lora", False):
        return

    model_cls.supports_lora = True
    for name, default in (
        ("is_3d_moe_weight", False),
        ("is_non_gated_moe", False),
        ("embedding_modules", {}),
        ("lora_skip_prefixes", []),
    ):
        if not hasattr(model_cls, name):
            setattr(model_cls, name, default)

    packed_modules_mapping = dict(model_cls.packed_modules_mapping)
    packed_modules_mapping.setdefault("attn.qkv", ["attn.q", "attn.k", "attn.v"])
    model_cls.packed_modules_mapping = packed_modules_mapping

    original_init = model_cls.__init__

    @wraps(original_init)
    def _patched_init(self: Any, *, vllm_config: Any, prefix: str = "") -> None:
        model_config = vllm_config.model_config
        full_config = model_config.hf_config
        thinker_config = getattr(full_config, "thinker_config", None)
        if thinker_config is None:
            original_init(self, vllm_config=vllm_config, prefix=prefix)
            return

        model_config.hf_config = thinker_config
        try:
            original_init(self, vllm_config=vllm_config, prefix=prefix)
        finally:
            model_config.hf_config = full_config

    model_cls.__init__ = _patched_init

    original_get_mrope_input_positions = model_cls.get_mrope_input_positions

    @wraps(original_get_mrope_input_positions)
    def _patched_get_mrope_input_positions(
        self: Any,
        input_tokens: list[int],
        mm_features: list[Any],
        **kwargs: Any,
    ) -> Any:
        del kwargs
        return original_get_mrope_input_positions(self, input_tokens, mm_features)

    model_cls.get_mrope_input_positions = _patched_get_mrope_input_positions
    setattr(model_cls, _PATCH_SENTINEL, True)


__all__ = ["patch_qwen3_omni_thinker_class"]
