"""Force the math SDP backend in the sglang worker (LTX-2 attention crash probe).

The LTX-2 DiT attention runs through torch SDPA
(``runtime/layers/attention/layer.py``'s ``F.scaled_dot_product_attention``). On
this cu130 / torch-2.11 build it heap-corrupts (``munmap_chunk(): invalid
pointer`` → ``Fatal Python error: Aborted``) — localized via
``PYTHONFAULTHANDLER`` to ``layer.py:323`` inside ``ltx_2.py:720 forward``.

Setting ``torch.backends.cuda.enable_flash_sdp(False)`` etc. is a *global* flag
that other code can flip back, so to GUARANTEE the math kernel we monkeypatch
``torch.nn.functional.scaled_dot_product_attention`` itself to run inside an
explicit ``sdpa_kernel(SDPBackend.MATH)`` context on every call. Math is the
reference kernel — slower but numerically exact and crash-free. Logs once on
install so we can confirm it actually took effect in the spawn worker.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_SENTINEL = "_unirl_force_math_sdp"


def patch_force_math_sdp() -> None:
    import torch
    import torch.nn.functional as F

    orig = F.scaled_dot_product_attention
    if getattr(orig, _SENTINEL, False):
        return

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except Exception:  # pragma: no cover - very old torch
        # Fall back to the global flags if the context API is unavailable.
        for setter in ("enable_flash_sdp", "enable_mem_efficient_sdp", "enable_cudnn_sdp"):
            fn = getattr(getattr(torch.backends, "cuda", None), setter, None)
            if fn is not None:
                try:
                    fn(False)
                except Exception:
                    pass
        return

    def _math_sdpa(*args, **kwargs):
        with sdpa_kernel(SDPBackend.MATH):
            return orig(*args, **kwargs)

    _math_sdpa._unirl_force_math_sdp = True  # type: ignore[attr-defined]
    F.scaled_dot_product_attention = _math_sdpa
    # Also flip the globals (belt-and-suspenders for code paths that call the
    # C-level op directly rather than the python F.* wrapper).
    for setter, val in (("enable_flash_sdp", False), ("enable_mem_efficient_sdp", False),
                        ("enable_cudnn_sdp", False), ("enable_math_sdp", True)):
        fn = getattr(getattr(torch.backends, "cuda", None), setter, None)
        if fn is not None:
            try:
                fn(val)
            except Exception:
                pass
    logger.warning("UNIRL: forced MATH SDP backend (LTX-2 attention crash workaround) installed")
