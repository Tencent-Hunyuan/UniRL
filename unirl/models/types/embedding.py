"""Conditioning embedding stage interface.

``EmbedStage[P, C]``: ``Primitive → Condition``. Same shape as
``EncodeStage``, separate name for the text-encoder flavor of the operation.

``ImageConditionedEmbedStage[P, ImageP, C]`` covers multimodal text encoders whose
embedding additionally depends on an image primitive.

The legacy ``SampleStage`` has been removed — its rollout-level role is
subsumed by typed ``DiffusionStage`` / ``ARStage``.
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

P = TypeVar("P")
ImageP = TypeVar("ImageP")
C = TypeVar("C")


@runtime_checkable
class EmbedStage(Protocol[P, C]):
    """Embed a primitive into its condition form (e.g. text → text-condition)."""

    def embed(self, p: P) -> C: ...


@runtime_checkable
class ImageConditionedEmbedStage(Protocol[P, ImageP, C]):
    """Embed a primitive with optional image context into its condition form.

    ``images=None`` means text-only encoding under the same stage (e.g. Edit-Plus
    keeps the edit chat template; it does not fall back to a different model
    family's text-only stage).
    """

    def embed(self, p: P, images: ImageP | None = None) -> C: ...


__all__ = ["EmbedStage", "ImageConditionedEmbedStage"]
