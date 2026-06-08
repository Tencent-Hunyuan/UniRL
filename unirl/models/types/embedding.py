"""Conditioning embedding stage interface.

``EmbedStage[P, C]``: ``Primitive → Condition``. Same shape as
``EncodeStage``, separate name for the text-encoder flavor of the operation.

The legacy ``SampleStage`` has been removed — its rollout-level role is
subsumed by typed ``DiffusionStage`` / ``ARStage``.
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

P = TypeVar("P")
C = TypeVar("C")


@runtime_checkable
class EmbedStage(Protocol[P, C]):
    """Embed a primitive into its condition form (e.g. text → text-condition)."""

    def embed(self, p: P) -> C: ...


__all__ = ["EmbedStage"]
