"""Runtime patch making ``get_moe_expert_mapping`` tolerate HI3's 2-tuple ``get_expert_mapping`` shape."""

from __future__ import annotations

_INSTALLED = False


def install() -> None:
    """Patch ``get_moe_expert_mapping`` everywhere it's used, unwrapping HI3's 2-tuple to vllm 0.20's flat list."""
    global _INSTALLED
    if _INSTALLED:
        return
    try:
        from vllm.model_executor import utils as vllm_mu
    except ImportError:
        return

    original = getattr(vllm_mu, "get_moe_expert_mapping", None)
    if original is None:
        _INSTALLED = True
        return
    if getattr(original, "_diffrl_hi3_unwrap", False):
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

    _patched._diffrl_hi3_unwrap = True  # type: ignore[attr-defined]
    vllm_mu.get_moe_expert_mapping = _patched

    try:
        from vllm.lora import utils as vllm_lora_utils

        if hasattr(vllm_lora_utils, "get_moe_expert_mapping"):
            vllm_lora_utils.get_moe_expert_mapping = _patched
    except ImportError:
        pass

    _INSTALLED = True


install()


__all__ = ["install"]
