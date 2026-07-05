"""Allow unirl's trusted tensor-rebuild classes through sglang's SafeUnpickler.

Full-weight sync ships serialized CUDA tensors whose reduction references unirl's
``_rebuild_cuda_tensor_modified`` (``weight_sync.transfer.sgl_compat``). sglang's
own ``SafeUnpickler`` (``srt/utils/common.py``, the CVE-2025-10164 mitigation)
allowlists only ``builtins``/``torch``/``multiprocessing``/… — NOT ``unirl.`` —
so ``update_weights_from_tensor`` raises ``Blocked unsafe class loading
(unirl…._rebuild_cuda_tensor_modified)`` and the sync (hence the whole full-FT
sglang run) dies at the first weight push.

unirl's payload is internally produced and trusted (unirl's OWN SafeUnpickler in
``sgl_compat`` already allowlists ``unirl.``), so the fix is to teach sglang's
unpickler the same: add the ``unirl.`` prefix to its class allowlist. Idempotent;
import-safe (sglang imported inside the fn). This must run in every process that
deserializes the payload — installed via the patch suite's ``hijack()`` (which
re-installs in spawned SRT children too).
"""

from __future__ import annotations


def patch_safe_unpickler() -> None:
    try:
        from sglang.srt.utils.common import SafeUnpickler
    except Exception:  # noqa: BLE001 — older sglang without the SafeUnpickler shim
        return
    prefixes = getattr(SafeUnpickler, "ALLOWED_MODULE_PREFIXES", None)
    if prefixes is None:
        return
    # ``ALLOWED_MODULE_PREFIXES`` is a class-level set; mutating it covers every
    # instance. Matches sglang's own ``(module + ".").startswith(prefix)`` check.
    if "unirl." not in prefixes:
        prefixes.add("unirl.")
