"""Process-local exact-mode state for the Qwen3 parity actor."""

from __future__ import annotations

import contextlib
import contextvars
import os
from collections.abc import Iterator

_EXACT = contextvars.ContextVar("unirl_parity_exact", default=False)
_PROCESS_EXACT_DEPTH = 0


def require_parity_opt_in() -> None:
    """Fail closed unless this process was dedicated to parity overrides."""
    if os.environ.get("UNIRL_PARITY_ENABLE") != "1":
        raise RuntimeError(
            "Qwen3 parity overrides require UNIRL_PARITY_ENABLE=1. Set it only for a dedicated parity-training process."
        )


def exact_enabled() -> bool:
    # Non-reentrant HF/FSDP checkpointing may replay under a copied Context
    # whose ContextVar value predates the outer algorithm scope. The process is
    # parity-dedicated, so the outer scope also holds a nested process counter.
    return bool(_EXACT.get()) or _PROCESS_EXACT_DEPTH > 0


@contextlib.contextmanager
def exact_mode(enabled: bool = True) -> Iterator[None]:
    global _PROCESS_EXACT_DEPTH
    token = _EXACT.set(bool(enabled))
    increment = int(bool(enabled))
    _PROCESS_EXACT_DEPTH += increment
    try:
        yield
    finally:
        _PROCESS_EXACT_DEPTH -= increment
        if _PROCESS_EXACT_DEPTH < 0:
            raise AssertionError("Qwen3 parity exact-mode process depth became negative")
        _EXACT.reset(token)


def exact_context():
    """Enter exact mode; installation remains an explicit, separate action."""
    return exact_mode(True)


__all__ = ["exact_context", "exact_enabled", "exact_mode", "require_parity_opt_in"]
