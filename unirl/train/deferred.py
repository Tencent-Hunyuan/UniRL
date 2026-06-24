"""Deferred-ops bookkeeping for build-time structural injection.

Injection functions (``inject_lora`` / ``inject_nft`` / ``inject_mirror``)
mutate the nn.Module tree while it may still be on meta device and stamp
``model._deferred_ops`` with post-materialize work.  A single call to
:func:`apply_deferred_ops` drains them all — feature-agnostic.
"""

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
