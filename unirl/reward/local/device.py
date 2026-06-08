"""Shared device resolver for reward Specs.

A Spec's ``device: str`` field accepts ``"cpu"``, ``"cuda"``, or ``"auto"``.
``"auto"`` defers to the backend's ``base_device`` (also
``"cpu"``/``"cuda"``/``"auto"``); when that itself is ``"auto"``, fall back to
``"cuda"`` if available else ``"cpu"``. This lets per-component overrides win
where set, while keeping a single cluster-level default.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def resolve_device(spec_device: str, base_device: str) -> str:
    """Resolve a Spec's ``device`` against the cluster-level ``base_device``.

    Precedence: explicit ``cpu``/``cuda`` on the spec wins; ``auto`` falls
    through to ``base_device``; if that is also ``auto``, pick ``cuda``
    when available else ``cpu``. ``cuda`` requested without availability
    falls back to ``cpu`` with a warning.
    """
    chosen = _resolve_one(spec_device)
    if chosen == "auto":
        chosen = _resolve_one(base_device)
    if chosen == "auto":
        chosen = "cuda" if torch.cuda.is_available() else "cpu"
    if chosen == "cuda" and not torch.cuda.is_available():
        logger.warning("Reward Spec requested cuda but CUDA is not available; falling back to cpu.")
        return "cpu"
    return chosen


def _resolve_one(value: str) -> str:
    pref = str(value or "").strip().lower()
    if pref in {"cpu", "cuda", "auto"}:
        return pref
    logger.warning(
        "Unknown device pref %r; falling back to cpu.",
        value,
    )
    return "cpu"


__all__ = ["resolve_device"]
