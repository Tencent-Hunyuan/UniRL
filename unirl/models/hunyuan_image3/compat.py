"""Runtime transformers-5.x compatibility shims for the HunyuanImage-3 checkpoint.

The checkpoint's vendored ``trust_remote_code`` modeling was written for
transformers 4.x. Instead of editing the checkpoint files on disk
(``scripts/hi3_ckpt_transformers5_compat.py``), apply these idempotent
monkeypatches once at bundle-load time — they need no on-disk state, survive
re-downloads, and travel with the unirl code:

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
"""
from __future__ import annotations


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

    # B — Siglip2 image processor: default return_tensors="pt"
    try:
        from transformers.models.siglip2 import image_processing_siglip2 as _sig

        _cls = getattr(_sig, "Siglip2ImageProcessorFast", None)
        if _cls is not None and not getattr(_cls.preprocess, "_hi3_compat", False):
            _orig_pp = _cls.preprocess

            def _preprocess(self, *args, **kwargs):
                kwargs.setdefault("return_tensors", "pt")
                return _orig_pp(self, *args, **kwargs)

            _preprocess._hi3_compat = True
            _cls.preprocess = _preprocess
    except Exception:  # noqa: BLE001
        pass
