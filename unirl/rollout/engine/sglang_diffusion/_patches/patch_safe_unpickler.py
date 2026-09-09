"""Allow unirl's trusted tensor-rebuild classes through sglang's SafeUnpickler."""

from __future__ import annotations


def patch_safe_unpickler() -> None:
    try:
        from sglang.srt.utils.common import SafeUnpickler
    except Exception:  # noqa: BLE001 — older sglang without the SafeUnpickler shim
        return
    prefixes = getattr(SafeUnpickler, "ALLOWED_MODULE_PREFIXES", None)
    if prefixes is None:
        return
    if "unirl." not in prefixes:
        prefixes.add("unirl.")
