"""Post-materialization callbacks for model construction (see ``../README.md``)."""

from __future__ import annotations

from typing import Callable, List

from torch import nn

_AC_WRAP_SEGMENT = "_checkpoint_wrapped_module."


def canonical_param_name(name: str) -> str:
    """Strip activation-checkpoint wrapper segments from a parameter name (see ``../README.md``)."""
    return name.replace(_AC_WRAP_SEGMENT, "")


def defer_after_materialize(model: nn.Module, op: Callable[[nn.Module], None]) -> None:
    """Register ``op`` to run once ``model`` has real storage and loaded weights."""
    if not hasattr(model, "_deferred_ops"):
        model._deferred_ops: List[Callable[[nn.Module], None]] = []
    model._deferred_ops.append(op)


def apply_deferred_ops(model: nn.Module) -> None:
    """Run and clear all post-materialization callbacks registered on ``model``."""
    for op in getattr(model, "_deferred_ops", []):
        op(model)
    model._deferred_ops = []


__all__ = ["apply_deferred_ops", "canonical_param_name", "defer_after_materialize"]
