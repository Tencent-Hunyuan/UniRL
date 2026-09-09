"""Conditioning embedding stage interface."""

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
    """Embed a primitive with optional image context into its condition form."""

    def embed(self, p: P, images: ImageP | None = None) -> C: ...


__all__ = ["EmbedStage", "ImageConditionedEmbedStage"]
