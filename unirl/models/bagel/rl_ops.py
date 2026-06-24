"""Navit-forward adapter over the PRISTINE official Bagel modeling.

Thin bridge from the official ``Bagel._forward_flow`` (packed navit + 3 KV contexts,
upstream ``@torch.no_grad``) to UniRL's runtime: ``forward_flow`` (grad-capable via
``__wrapped__``) + ``disable_inference_cache`` (TaylorSeer off for replay determinism).
Everything else (SDE/schedule/noise) is UniRL's, so ``vendor/`` stays byte-pristine.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "forward_flow",
    "disable_inference_cache",
]


def disable_inference_cache(model: Any) -> None:
    """Turn off the TaylorSeer cache for the RL path (per-step replay determinism).

    Best-effort; ignored if the attribute path is absent (e.g. a fake test model).
    """
    try:
        model.language_model.model.enable_taylorseer = False
    except AttributeError:
        pass


def _raw_forward_flow(model: Any):
    """The undecorated ``Bagel._forward_flow`` (bypasses upstream ``@torch.no_grad``)."""
    fn = type(model)._forward_flow
    return getattr(fn, "__wrapped__", fn)


def forward_flow(model: Any, **kwargs: Any) -> Any:
    """CFG-combined velocity via the pristine ``Bagel._forward_flow``.

    Bypasses upstream's ``@torch.no_grad`` (via ``__wrapped__``) so gradients flow in
    replay; under an outer ``torch.no_grad()`` it stays grad-free. CFG is done inside.
    """
    return _raw_forward_flow(model)(model, **kwargs)
