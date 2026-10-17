"""Conditioning embedding stage protocols and batch prompt deduplication helper."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Protocol, TypeVar, runtime_checkable

P = TypeVar("P")
ImageP = TypeVar("ImageP")
C = TypeVar("C")
K = TypeVar("K", bound=Hashable)


def dedup_prompts(items: Sequence[K]) -> tuple[list[K], list[int] | None]:
    """Deduplicate prompt keys in first-seen order and return ``(uniq, inverse_or_none)``."""
    index_of: dict[K, int] = {}
    inverse: list[int] = []
    uniq: list[K] = []
    for item in items:
        i = index_of.get(item)
        if i is None:
            i = len(uniq)
            index_of[item] = i
            uniq.append(item)
        inverse.append(i)
    return uniq, inverse if len(uniq) != len(items) else None


@runtime_checkable
class EmbedStage(Protocol[P, C]):
    """Embed a primitive into its condition form (e.g. text → text-condition)."""

    def embed(self, p: P) -> C: ...


@runtime_checkable
class ImageConditionedEmbedStage(Protocol[P, ImageP, C]):
    """Embed a primitive with optional image context into its condition form."""

    def embed(self, p: P, images: ImageP | None = None) -> C: ...


__all__ = ["EmbedStage", "ImageConditionedEmbedStage", "dedup_prompts"]
