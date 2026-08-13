"""Segment types — SoA batched containers for generation outputs."""

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
