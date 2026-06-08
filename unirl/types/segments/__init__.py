"""Segment types — SoA batched containers for generation outputs.

A ``Segment`` is always batched (SoA); rows are 1:1 with the enclosing
track's samples, so per-sample access is plain row indexing. Each
modality has its own subclass:

- ``LatentSegment`` covers image / video / audio diffusion rollouts.
- ``TextSegment`` covers AR token rollouts (varlen-packed).
"""

from __future__ import annotations

from unirl.types.segments.base import Segment, SegmentStatus
from unirl.types.segments.latent import (
    LatentSegment,
    make_audio_segment,
    make_image_segment,
    make_video_segment,
)
from unirl.types.segments.text import TextSegment

__all__ = [
    "LatentSegment",
    "Segment",
    "SegmentStatus",
    "TextSegment",
    "make_audio_segment",
    "make_image_segment",
    "make_video_segment",
]
