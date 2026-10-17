"""Allow unirl's trusted tensor-rebuild classes through sglang's SafeUnpickler."""

from __future__ import annotations


def patch_safe_unpickler() -> None:
    from sglang.srt.utils.common import SafeUnpickler

    prefixes = SafeUnpickler.ALLOWED_MODULE_PREFIXES
    if "unirl." not in prefixes:
        prefixes.add("unirl.")
