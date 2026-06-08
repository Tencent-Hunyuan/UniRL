"""Runtime patch for HuggingFace ``PreTrainedTokenizer*.convert_tokens_to_ids``.

vllm-omni's ``HunyuanImage3ForCausalMM.__init__`` (~line 1508 of
``vllm_omni/model_executor/models/hunyuan_image3/hunyuan_image3.py``)
looks up ``<img_ratio_33>`` and ``<img_ratio_36>`` via
``tokenizer.convert_tokens_to_ids`` and computes ``ratio_36 + 1``. The
Base ``HunyuanImage-3`` checkpoint ships ratio tokens 0-32 only, so the
``<img_ratio_36>`` lookup returns ``None`` and the addition raises::

    TypeError: unsupported operand type(s) for +: 'NoneType' and 'int'

The Instruct ckpt ships the missing tokens, so this patch is a no-op
there (the underlying call returns a real id and the wrapper is
bypassed). Cross-engine smoke runs on Base because the training-side
bundle has only been validated on Base; the patch is required.

Workaround: patch ``convert_tokens_to_ids`` to return ``0`` when the
underlying call returns ``None`` for a single-token string lookup. The
ratio-33/36 case becomes ``(0, 1)`` -> ``range(0, 1) = [0]``, which
adds token 0 to ``_all_ratio_ids`` -- harmless because the t2i / it2i
flows never sample those ratio tokens.

``convert_tokens_to_ids`` is defined on ``PreTrainedTokenizer`` (slow)
and ``PreTrainedTokenizerFast`` independently -- *not* on the shared
base ``PreTrainedTokenizerBase``. Patch both classes (and skip cleanly
when one isn't importable).

This module also carries ``HI3ARWorkerExtension`` -- the class qualname
target plumbed into Stage 0's ``engine_args.worker_extension_cls`` by
the static stage YAMLs in ``stage_configs/``. vllm-omni imports the
module to resolve the qualname, which fires the module-top ``install()``
call below before the AR model is constructed.
"""

from __future__ import annotations

_INSTALLED = False


def install() -> None:
    """Patch ``PreTrainedTokenizer*`` to return 0 for missing single-token ids.

    Idempotent via the ``_unirl_none_filter`` sentinel attribute
    on the wrapper. Safe to call from any process -- silent no-op when
    transformers isn't importable.
    """
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
            continue  # already patched

        original = method

        def _filtered(self, tokens, *args, _orig=original, **kwargs):
            result = _orig(self, tokens, *args, **kwargs)
            if isinstance(tokens, str) and result is None:
                return 0
            return result

        _filtered._unirl_none_filter = True  # type: ignore[attr-defined]
        cls.convert_tokens_to_ids = _filtered

    _INSTALLED = True


# Module-import side effect: fires when vllm-omni's worker subprocess
# imports this module to resolve the ``HI3ARWorkerExtension`` qualname.
# Idempotent via the ``_INSTALLED`` guard, so re-imports are safe.
install()

# Side-effect: install the HI3-specific MoE-LoRA compat patch needed when
# ``enable_lora=true`` on the AR stage. Importing the module fires its
# ``install()``; the patch is idempotent and a no-op if vllm isn't loaded.
from unirl.rollout.engine.vllm_omni.compat import hi3_lora as _hi3_lora_compat  # noqa: F401, E402


class HI3ARWorkerExtension:
    """vllm-omni ``worker_extension_cls`` qualname target for HI3 AR.

    Carries no methods of its own. Its only purpose is to be the
    qualname that vllm resolves via ``resolve_obj_by_qualname`` --
    resolving it imports this module, which triggers the
    module-top ``install()`` above before vllm constructs the AR model.
    """

    pass


__all__ = ["install", "HI3ARWorkerExtension"]
