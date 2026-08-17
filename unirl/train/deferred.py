"""Deferred-ops bookkeeping for build-time structural injection."""

from __future__ import annotations

from typing import Callable, List

from torch import nn


def _stamp(model: nn.Module, op: Callable[[nn.Module], None]) -> None:
    if not hasattr(model, "_deferred_ops"):
        model._deferred_ops: List[Callable[[nn.Module], None]] = []
    model._deferred_ops.append(op)


def apply_deferred_ops(model: nn.Module) -> None:
    """Drain ``_deferred_ops`` after materialize.  Feature-agnostic."""
    for op in getattr(model, "_deferred_ops", []):
        op(model)
    model._deferred_ops = []


__all__ = ["apply_deferred_ops"]
