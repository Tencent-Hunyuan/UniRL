"""Make vLLM's MoE LoRA mapping accept HI3's mapping-plus-replacements tuple."""

from __future__ import annotations

_INSTALLED = False


def install() -> None:
    """Unwrap HI3's ``(expert_mapping, replacements)`` result for vLLM LoRA."""
    global _INSTALLED
    if _INSTALLED:
        return
    from vllm.lora import utils as vllm_lora_utils
    from vllm.model_executor import utils as vllm_mu

    original = vllm_mu.get_moe_expert_mapping
    _ = vllm_lora_utils.get_moe_expert_mapping
    if getattr(original, "_unirl_hi3_unwrap", False):
        _INSTALLED = True
        return

    def _patched(model, _orig=original):
        result = _orig(model)
        if (
            isinstance(result, tuple)
            and len(result) == 2
            and not isinstance(result[0], (str, int, float))
            and hasattr(result[0], "__iter__")
            and isinstance(result[1], dict)
        ):
            return result[0]
        return result

    _patched._unirl_hi3_unwrap = True  # type: ignore[attr-defined]
    vllm_mu.get_moe_expert_mapping = _patched
    vllm_lora_utils.get_moe_expert_mapping = _patched

    _INSTALLED = True


install()


__all__ = ["install"]
