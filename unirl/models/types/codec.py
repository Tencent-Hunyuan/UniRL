"""Latent/media codec interfaces — typed pipeline stages between Primitive and Condition/Segment.

``EncodeStage[P, C]``: ``Primitive → Condition``. e.g. VAE encoder for an
image-conditioning slot.

``DecodeStage[S, P]``: ``Segment → Primitive``. e.g. VAE decoder turning a
``LatentSegment`` back into an ``Image``.
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

P = TypeVar("P")
C = TypeVar("C")


@runtime_checkable
class EncodeStage(Protocol[P, C]):
    """Encode a primitive into its condition form."""

    def encode(self, p: P) -> C: ...


@runtime_checkable
class DecodeStage(Protocol[C, P]):
    """Decode a condition back into a primitive."""

    def decode(self, c: C) -> P: ...


__all__ = ["DecodeStage", "EncodeStage"]
