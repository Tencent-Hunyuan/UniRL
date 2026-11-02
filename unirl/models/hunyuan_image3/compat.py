"""Runtime transformers-5.x compatibility shims for the HunyuanImage-3 checkpoint."""

from __future__ import annotations

import inspect
from functools import wraps
from typing import Any


def apply_hi3_transformers5_compat() -> None:
    """Idempotently install the transformers-5.x compat shims. Safe to call repeatedly."""
    try:
        from transformers.cache_utils import StaticLayer

        if not getattr(StaticLayer.lazy_initialization, "_hi3_compat", False):
            _orig_lazy = StaticLayer.lazy_initialization

            def _lazy_initialization(self, key_states, value_states=None, *args, **kwargs):
                if value_states is None:
                    value_states = key_states
                return _orig_lazy(self, key_states, value_states, *args, **kwargs)

            _lazy_initialization._hi3_compat = True
            StaticLayer.lazy_initialization = _lazy_initialization
    except Exception:  # noqa: BLE001 — best-effort; a transformers without StaticLayer doesn't need it
        pass

    try:
        from transformers.models.siglip2 import image_processing_siglip2 as _sig

        for _clsname in ("Siglip2ImageProcessor", "Siglip2ImageProcessorFast"):
            _cls = getattr(_sig, _clsname, None)
            if _cls is None or getattr(_cls.preprocess, "_hi3_compat", False):
                continue
            _orig_pp = _cls.preprocess

            def _preprocess(self, *args, _orig=_orig_pp, **kwargs):
                kwargs.setdefault("return_tensors", "pt")
                return _orig(self, *args, **kwargs)

            _preprocess._hi3_compat = True
            _cls.preprocess = _preprocess
    except Exception:  # noqa: BLE001
        pass


def repair_hi3_tokenizer_backend(tokenizer: Any, pretrained_path: Any) -> bool:
    """Re-attach the correct BPE Rust backend to a char-level HI3 tokenizer."""
    import os

    backend = getattr(tokenizer, "_tokenizer", None)
    if backend is None or getattr(backend, "pre_tokenizer", None) is not None:
        return False
    tok_json = os.path.join(str(pretrained_path), "tokenizer.json")
    if not os.path.exists(tok_json):
        return False
    try:
        from tokenizers import Tokenizer as _RustTokenizer

        tokenizer._tokenizer = _RustTokenizer.from_file(tok_json)
    except Exception:  # noqa: BLE001 — best-effort; leave the tokenizer untouched on failure
        return False
    return True


def repair_hi3_tokenizer_batchify(tokenizer: Any) -> bool:
    """Repair the checkpoint's truncated batchify path; see README ``## Gotchas``."""
    original = getattr(tokenizer, "apply_general_template", None)
    if not callable(original) or not callable(getattr(tokenizer, "batch_gen_infer", None)):
        return False
    if getattr(original, "_hi3_batchify_compat", False):
        return False
    signature = inspect.signature(original)
    if not {"message_list", "batchify", "cfg_factor", "uncond_p"}.issubset(signature.parameters):
        return False

    @wraps(original)
    def apply_general_template(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        if not bound.arguments["batchify"]:
            return original(*args, **kwargs)
        options = bound.arguments
        messages = options.pop("message_list")
        options.pop("batchify")
        cfg_factor = options.pop("cfg_factor")
        options.pop("uncond_p")
        if not messages or not all(isinstance(message, list) for message in messages):
            raise ValueError("HI3 batchify expects a non-empty list of message lists")
        return tokenizer.batch_gen_infer(
            infer_fn=original,
            prompt_list=[[] for _ in messages],
            infer_fn_kwargs_list=[dict(options, message_list=message) for message in messages],
            do_classifier_free_guidance=cfg_factor > 1,
            condition_repeat_times=1,
            uncondition_repeat_times=cfg_factor - 1,
        )

    apply_general_template._hi3_batchify_compat = True
    tokenizer.apply_general_template = apply_general_template
    return True
