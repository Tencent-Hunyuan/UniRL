"""Canonical transition-rule helpers for shared SDE math."""

from __future__ import annotations

from typing import Any, Optional


def normalize_sde_type(value: Any) -> str:
    """Normalize an sde_type value to canonical lowercase text.

    Used at sglang-engine kwarg boundaries that still flow strings; the
    canonical Python-side dispatch uses the typed
    ``cfg.sampling.sde_strategy`` Hydra group instead.
    """
    return str(value or "").strip().lower()


def is_deterministic_sde_type(
    sde_type: str,
    eta: Optional[float] = None,
) -> bool:
    """Whether the transition is deterministic at runtime.

    Expects *sde_type* to already be canonical lowercase.
    """
    if sde_type == "dpm2":
        return True
    if eta is None:
        return False
    return float(eta) == 0.0


__all__ = [
    "is_deterministic_sde_type",
    "normalize_sde_type",
]
