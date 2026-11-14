"""Geometry resolution and packed-row layout for MiniMax-H3 t2va / fl2va."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from unirl.config.require import require

from .conditions import MiniMaxH3Conditions
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
    MINIMAX_H3_MAX_DURATION,
    MINIMAX_H3_MAX_PIXELS,
    MINIMAX_H3_MIN_ASPECT_RATIO,
    MINIMAX_H3_MIN_DURATION,
    MINIMAX_H3_SHORT_EDGE,
    MINIMAX_H3_TEXT_TAG,
    MINIMAX_H3_VIDEO_TAG,
    MiniMaxH3PackedSequence,
    align_num_frames,
    audio_latent_num_frames,
    build_packed_sequence,
    build_row_timesteps,
    video_latent_num_frames,
)

MINIMAX_H3_KEYFRAME_ANCHORS: Tuple[Tuple[str, ...], ...] = ((), ("first",), ("last",), ("first", "last"))


@dataclass(frozen=True)
class MiniMaxH3Geometry:
    """The resolved shape of one request. Depends only on SHARED sampling params."""

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
        """Validate a requested ``(height, width, num_frames)`` against H3."""
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
        duration_seconds = aligned / MINIMAX_H3_FPS
        if not MINIMAX_H3_MIN_DURATION <= duration_seconds <= MINIMAX_H3_MAX_DURATION:
            raise ValueError(
                f"MiniMaxH3Geometry: num_frames={num_frames} aligns to {aligned} frames "
                f"({duration_seconds:.2f}s at {MINIMAX_H3_FPS} fps), outside H3's supported "
                f"{MINIMAX_H3_MIN_DURATION:g}-{MINIMAX_H3_MAX_DURATION:g}s range."
            )
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
        return cls.resolve(height=params.height, width=params.width, num_frames=params.num_frames)


def build_layout(geometry: MiniMaxH3Geometry, conditions: MiniMaxH3Conditions) -> MiniMaxH3PackedSequence:
    """Build the ``[text | keyframe conditions | audio | video]`` layout of one trimmed batch-1 request."""
    num_text_tokens = int(conditions.text.embeds.shape[1])
    keyframes = conditions.keyframes
    if keyframes is None:
        text_token_tags = torch.full((num_text_tokens,), MINIMAX_H3_TEXT_TAG, dtype=torch.long)
        keyframe_anchors: Tuple[str, ...] = ()
    else:
        require(
            keyframes.batch_size == 1,
            f"build_layout: one packed layout per forward, got {keyframes.batch_size} keyframe samples",
        )
        text_token_tags = keyframes.text_token_tags[0].to("cpu")
        keyframe_anchors = keyframes.anchors[0]
        require(
            keyframe_anchors in MINIMAX_H3_KEYFRAME_ANCHORS,
            f"build_layout: keyframe anchors {keyframe_anchors!r} must be one of {MINIMAX_H3_KEYFRAME_ANCHORS!r}",
        )
        require(
            text_token_tags.dtype == torch.long
            and tuple(text_token_tags.shape) == (num_text_tokens,)
            and bool(((text_token_tags == MINIMAX_H3_TEXT_TAG) | (text_token_tags == MINIMAX_H3_VIDEO_TAG)).all()),
            f"build_layout: text_token_tags must be a [{num_text_tokens}] long tensor of 0/1 tags matching the "
            f"trimmed text, got {text_token_tags.dtype} {tuple(text_token_tags.shape)}",
        )
        require(
            bool(keyframe_anchors) or bool((text_token_tags == MINIMAX_H3_TEXT_TAG).all()),
            f"build_layout: a t2va entry (anchors ()) must tag every text token {MINIMAX_H3_TEXT_TAG}",
        )
        rows = keyframes.rows[0]
        expected = (len(keyframe_anchors) * geometry.rows_per_frame, geometry.video_token_dim)
        require(
            rows.dtype == torch.float32 and tuple(rows.shape) == expected,
            f"build_layout: keyframe rows must be fp32 {expected} for {keyframe_anchors!r} at this canvas, got "
            f"{rows.dtype} {tuple(rows.shape)}",
        )
    return build_packed_sequence(
        text_token_tags=text_token_tags,
        num_latent_frames=geometry.num_latent_frames,
        latent_height=geometry.latent_height,
        latent_width=geometry.latent_width,
        num_audio_latents=geometry.num_audio_latents,
        patch_size=MINIMAX_H3_PATCH_SIZE,
        keyframe_anchors=keyframe_anchors,
    )


def row_timestep_plan(
    layout: MiniMaxH3PackedSequence,
    *,
    video_sigma: torch.Tensor,
    audio_sigma: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(unique_timesteps, timestep_indices)`` for one denoising step."""
    if video_sigma.numel() == 1 and audio_sigma.numel() == 1:
        video_t = 1.0 - float(video_sigma)
        audio_t = 1.0 - float(audio_sigma)
        return build_row_timesteps(
            layout,
            video_timestep=video_t,
            audio_timestep=audio_t,
            condition_video_timestep=video_t,
            condition_audio_timestep=video_t,
        )

    video_t = (1.0 - video_sigma.detach().reshape(-1).to(dtype=torch.float64)).to(dtype=torch.float32)
    audio_t = (1.0 - audio_sigma.detach().reshape(-1).to(dtype=torch.float64)).to(
        device=video_t.device, dtype=torch.float32
    )
    if video_t.shape != audio_t.shape:
        raise ValueError(
            "row_timestep_plan: video_sigma and audio_sigma must have matching batch shapes, got "
            f"{tuple(video_sigma.shape)} and {tuple(audio_sigma.shape)}"
        )

    audio_target_cols = layout.audio_indices[layout.num_condition_audio_rows :].to(device=video_t.device)
    row_timesteps = video_t[:, None].expand(-1, layout.sequence_length).clone()
    row_timesteps[:, audio_target_cols] = audio_t[:, None]
    return torch.unique(row_timesteps, sorted=True, return_inverse=True)


__all__ = [
    "MiniMaxH3Geometry",
    "build_layout",
    "row_timestep_plan",
]
