"""Geometry resolution and packed-row layout for MiniMax-H3 t2va.

MiniMax-H3 runs its transformer over ONE packed 1-D sequence holding every
modality at once. For t2va the row order is::

    [ text (L) | target audio (A) | target video (V) ]

(``fl2va`` inserts keyframe conditioning rows between text and audio; that is
Track B and deliberately not built here, though the vendored builder already
supports it via ``keyframe_anchors``.)

This module is a thin resolver over the vendored builders -- the row geometry,
the float64 rotary clock and the tag values are checkpoint contracts, so they
are USED from ``vendor.packing`` rather than reimplemented. What lives here is
only the mapping from UniRL's ``DiffusionSamplingParams`` to the arguments those
builders want, plus the sigma -> timestep conversion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from .config import (
    MINIMAX_H3_LATENT_CHANNELS,
    MINIMAX_H3_PATCH_SIZE,
    MINIMAX_H3_SPATIAL_COMPRESSION,
)
from .vendor import (
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_CANVAS_MULTIPLE,
    MINIMAX_H3_FPS,
    MINIMAX_H3_MAX_ASPECT_RATIO,
    MINIMAX_H3_MAX_PIXELS,
    MINIMAX_H3_MIN_ASPECT_RATIO,
    MINIMAX_H3_SHORT_EDGE,
    MINIMAX_H3_TEXT_TAG,
    MiniMaxH3PackedSequence,
    align_num_frames,
    audio_latent_num_frames,
    build_packed_sequence,
    build_row_timesteps,
    video_latent_num_frames,
)


@dataclass(frozen=True)
class MiniMaxH3Geometry:
    """The resolved shape of one request. Depends only on SHARED sampling params.

    Every field here is a pure function of ``(aspect, duration)``, never of
    per-sample data. That is load-bearing: ``LatentSegment`` stores latents in a
    ``FieldKind.CONCAT`` tensor, so ``torch.cat`` across the batch requires
    identical non-batch dims. Per-sample geometry would need a varlen segment
    UniRL does not have.
    """

    height: int
    width: int
    num_frames: int
    num_latent_frames: int
    latent_height: int
    latent_width: int
    num_audio_latents: int

    @property
    def rows_per_frame(self) -> int:
        _, patch_h, patch_w = MINIMAX_H3_PATCH_SIZE
        return (self.latent_height // patch_h) * (self.latent_width // patch_w)

    @property
    def num_video_rows(self) -> int:
        return self.num_latent_frames * self.rows_per_frame

    @property
    def num_audio_rows(self) -> int:
        return self.num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS

    @property
    def video_token_dim(self) -> int:
        patch_t, patch_h, patch_w = MINIMAX_H3_PATCH_SIZE
        return MINIMAX_H3_LATENT_CHANNELS * patch_t * patch_h * patch_w

    @property
    def latent_shape(self) -> Tuple[int, int, int, int]:
        """Per-sample UNPACKED video latent shape ``(C, T_lat, H_lat, W_lat)``."""
        return (
            MINIMAX_H3_LATENT_CHANNELS,
            self.num_latent_frames,
            self.latent_height,
            self.latent_width,
        )

    @classmethod
    def resolve(cls, *, height: int, width: int, num_frames: int) -> "MiniMaxH3Geometry":
        """Validate a requested ``(height, width, num_frames)`` against H3.

        Three structural canvas constraints: both axes a multiple of 32, aspect
        within 1:4..4:1, and area no larger than ``768*1344``. The multiple of 32
        is the load-bearing one, since it is what keeps ``height/16`` divisible
        by the transformer's patch of 2 -- a violation misaligns the packed rows
        rather than merely degrading quality.

        The 768 pixel short edge is not among them. ``resolve_canvas_size`` is a
        defaults helper answering "given only an aspect ratio, what canvas?", and
        reading it as a validator (as this once did) turned its starting choice
        into a floor that rejected every smaller canvas.

        Frame count is a genuine snap -- the video VAE encodes 17 pixel frames
        per chunk and drops 3 trailing latent frames, so only ``17n + 5`` counts
        round-trip -- but this raises with the nearest legal value rather than
        re-resolving, since training at a geometry the recipe does not state is
        undetectable later.
        """
        h, w = int(height), int(width)
        multiple = MINIMAX_H3_CANVAS_MULTIPLE
        if h % multiple or w % multiple:
            raise ValueError(
                f"MiniMaxH3Geometry: height={h} width={w} must both be multiples of {multiple}. That is what keeps "
                f"the latent dims (axis/{MINIMAX_H3_SPATIAL_COMPRESSION}) divisible by the transformer patch of 2; "
                f"nearest legal canvas is height={max(multiple, round(h / multiple) * multiple)} "
                f"width={max(multiple, round(w / multiple) * multiple)}."
            )
        ratio = w / h
        if not MINIMAX_H3_MIN_ASPECT_RATIO <= ratio <= MINIMAX_H3_MAX_ASPECT_RATIO:
            raise ValueError(
                f"MiniMaxH3Geometry: aspect ratio {w}:{h} ({ratio:g}) is outside the 1:4..4:1 range MiniMax-H3 "
                f"supports."
            )
        if h * w > MINIMAX_H3_MAX_PIXELS:
            raise ValueError(
                f"MiniMaxH3Geometry: height={h} width={w} is {h * w} pixels, above H3's {MINIMAX_H3_MAX_PIXELS} "
                f"area cap ({MINIMAX_H3_SHORT_EDGE}x1344). Reduce either axis; the cap is an upper bound only."
            )
        canvas_height, canvas_width = h, w
        aligned = align_num_frames(int(num_frames))
        if aligned != int(num_frames):
            raise ValueError(
                f"MiniMaxH3Geometry: num_frames={num_frames} does not round-trip through the video VAE, which maps "
                f"`17n + 5` pixel frames to `5n + 2` latent frames. Nearest legal value: {aligned} "
                f"({aligned / MINIMAX_H3_FPS:.2f}s at {MINIMAX_H3_FPS} fps; H3 supports 5-15s)."
            )
        return cls(
            height=canvas_height,
            width=canvas_width,
            num_frames=aligned,
            num_latent_frames=video_latent_num_frames(aligned),
            latent_height=canvas_height // MINIMAX_H3_SPATIAL_COMPRESSION,
            latent_width=canvas_width // MINIMAX_H3_SPATIAL_COMPRESSION,
            num_audio_latents=audio_latent_num_frames(aligned),
        )

    @classmethod
    def from_params(cls, params) -> "MiniMaxH3Geometry":
        """Resolve from a ``DiffusionSamplingParams``-shaped object."""
        return cls.resolve(height=int(params.height), width=int(params.width), num_frames=int(params.num_frames))


def build_t2va_layout(geometry: MiniMaxH3Geometry, num_text_tokens: int) -> MiniMaxH3PackedSequence:
    """Build the ``[text | audio | video]`` layout for a t2va request.

    All text rows carry the text tag. (``fl2va`` tags the rows of a keyframe's
    vision block as VIDEO instead -- that distinction only exists once keyframes
    do, so t2va passes a uniform tag vector.)
    """
    text_token_tags = torch.full((int(num_text_tokens),), MINIMAX_H3_TEXT_TAG, dtype=torch.long)
    return build_packed_sequence(
        text_token_tags=text_token_tags,
        num_latent_frames=geometry.num_latent_frames,
        latent_height=geometry.latent_height,
        latent_width=geometry.latent_width,
        num_audio_latents=geometry.num_audio_latents,
        patch_size=MINIMAX_H3_PATCH_SIZE,
        keyframe_anchors=(),
    )


def row_timestep_plan(
    layout: MiniMaxH3PackedSequence,
    *,
    video_sigma: torch.Tensor,
    audio_sigma: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(unique_timesteps, timestep_indices)`` for one denoising step.

    MiniMax-H3 conditions its AdaLN on ``t = 1 - sigma`` in ``[0, 1]``, UNSCALED
    (no x1000) and with ``t = 1`` meaning clean -- the opposite direction to
    every other model in this repo. Text rows never reach an output head and
    inherit the video timestep.

    t2va has no conditioning rows, so the condition timesteps are dead
    arguments; they are passed as the video timestep so they cannot introduce a
    spurious entry into the unique set if a future caller does add rows without
    revisiting this.
    """
    video_t = float(1.0 - float(video_sigma))
    audio_t = float(1.0 - float(audio_sigma))
    return build_row_timesteps(
        layout,
        video_timestep=video_t,
        audio_timestep=audio_t,
        condition_video_timestep=video_t,
        condition_audio_timestep=video_t,
    )


__all__ = [
    "MiniMaxH3Geometry",
    "build_t2va_layout",
    "row_timestep_plan",
]
