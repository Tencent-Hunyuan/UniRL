"""Runtime patch for HuggingFace ``PreTrainedTokenizer*.convert_tokens_to_ids``."""

from __future__ import annotations

_INSTALLED = False


def install() -> None:
    """Patch ``PreTrainedTokenizer*`` to return 0 for missing single-token ids."""
    global _INSTALLED
    if _INSTALLED:
        return

    candidates: list = []
    try:
        from transformers.tokenization_utils import PreTrainedTokenizer

        candidates.append(PreTrainedTokenizer)
    except ImportError:
        pass
    try:
        from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

        candidates.append(PreTrainedTokenizerFast)
    except ImportError:
        pass

    for cls in candidates:
        method = getattr(cls, "convert_tokens_to_ids", None)
        if method is None:
            continue
        if getattr(method, "_unirl_none_filter", False):
            continue

        original = method

        def _filtered(self, tokens, *args, _orig=original, **kwargs):
            result = _orig(self, tokens, *args, **kwargs)
            if isinstance(tokens, str) and result is None:
                return 0
            return result

        _filtered._unirl_none_filter = True  # type: ignore[attr-defined]
        cls.convert_tokens_to_ids = _filtered

    _INSTALLED = True


install()

from unirl.rollout.engine.vllm_omni.patches import compat_hi3_lora as _hi3_lora_compat  # noqa: F401, E402


class HI3ARWorkerExtension:
    """vllm-omni ``worker_extension_cls`` qualname target for HI3 AR."""

    pass


__all__ = ["install", "HI3ARWorkerExtension"]
