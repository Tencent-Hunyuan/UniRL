"""Runtime transformers-5.x compatibility shims for the HunyuanImage-3 checkpoint.

The checkpoint's vendored ``trust_remote_code`` modeling was written for
transformers 4.x. Instead of editing the checkpoint files on disk, apply these
idempotent monkeypatches once at bundle-load time — they need no on-disk state,
survive re-downloads, and travel with the unirl code:

  A. transformers 5.x ``StaticLayer.lazy_initialization`` requires
     ``(key_states, value_states)``; the checkpoint's static cache calls it with
     ``key_states`` only. Default ``value_states=key_states`` (key/value share
     shape+dtype, so the lazily-allocated cache is sized correctly).
  B. The Siglip2 image processor returns list-valued ``pixel_values`` unless
     ``return_tensors`` is set; the checkpoint's ``vit_process_image`` /
     ``preprocess`` call it without that, then ``.squeeze(0)`` on a list. Default
     ``return_tensors="pt"``.

The base-only forward ``rope_image_info`` kwarg (Patch C in the on-disk patcher)
is intentionally NOT shimmed here: base's forward tolerates the extra kwarg, and
base's DiT path is not bundle-supported anyway.

Separately, :func:`repair_hi3_tokenizer_backend` fixes a transformers-5.x
tokenizer-load defect (the checkpoint's ``HunyuanImage3TokenizerFast`` Rust
backend loads with ``pre_tokenizer``/``decoder`` = ``None`` -> char-level). It
needs the *loaded* tokenizer instance, so it is called from the text-embed stage
after ``load_tokenizer``, not from :func:`apply_hi3_transformers5_compat`.
"""

from __future__ import annotations

from typing import Any


def apply_hi3_transformers5_compat() -> None:
    """Idempotently install the transformers-5.x compat shims. Safe to call repeatedly."""
    # A — StaticLayer.lazy_initialization(key_states[, value_states])
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

    # B — Siglip2 image processor: default return_tensors="pt". Patch BOTH the
    # Fast and non-Fast classes: the "Fast" suffix is deprecated in transformers
    # 5.x and from_dict may yield a non-Fast instance.
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
    """Re-attach the correct BPE Rust backend to a char-level HI3 tokenizer.

    Under transformers 5.x the checkpoint's ``HunyuanImage3TokenizerFast``
    (a ``PreTrainedTokenizerFast`` subclass) loads its ``tokenizers.Tokenizer``
    backend with ``pre_tokenizer``/``decoder`` = ``None`` — i.e. char-level:
    ``"Good"`` -> ``['G','o','o','d']`` with inter-word spaces dropped. The model
    is then fed char-level, space-less prompts via ``apply_chat_template`` and
    generates text character-by-character (the garbled, space-less AR output).
    ``AutoTokenizer`` (transformers-5.x ``TokenizersBackend``) loads the *same*
    ``tokenizer.json`` correctly (ByteLevel pre-tokenizer + decoder).

    Fix: load ``tokenizer.json`` directly via ``tokenizers.Tokenizer.from_file``
    and swap it in. The special-token vocab (``<boi>``/``<img>``/``</think>`` …)
    is carried in ``tokenizer.json``, so chat-template marker splicing is
    preserved. Mutates ``tokenizer`` in place.

    Idempotent: only acts when the backend is the broken (``pre_tokenizer is
    None``) variant. Returns ``True`` if a repair was applied.
    """
    import os

    backend = getattr(tokenizer, "_tokenizer", None)
    if backend is None or getattr(backend, "pre_tokenizer", None) is not None:
        # Not a fast tokenizer, or already has the ByteLevel pre-tokenizer.
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
