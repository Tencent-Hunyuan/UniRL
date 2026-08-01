"""Shared bounded video sampling for Qwen3-Omni rollout and replay."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import torch


def _validated_sampling_args(fps: float, max_frames: Optional[int]) -> Tuple[float, Optional[int]]:
    resolved_fps = float(fps)
    if resolved_fps <= 0.0:
        raise ValueError(f"Qwen3-Omni video fps must be > 0, got {fps!r}")
    if max_frames is None:
        return resolved_fps, None
    resolved_max_frames = int(max_frames)
    if resolved_max_frames < 1:
        raise ValueError(f"Qwen3-Omni video_max_frames must be >= 1, got {max_frames!r}")
    return resolved_fps, resolved_max_frames


def limit_video_frames(
    frames: Any,
    *,
    fps: float,
    max_frames: Optional[int],
) -> Tuple[Any, float]:
    """Uniformly cap an already-decoded clip and return its effective fps."""
    resolved_fps, resolved_max_frames = _validated_sampling_args(fps, max_frames)
    if resolved_max_frames is None:
        return frames, resolved_fps

    num_frames = len(frames)
    if num_frames <= resolved_max_frames:
        return frames, resolved_fps

    positions = torch.linspace(0, num_frames - 1, steps=resolved_max_frames).round().long()
    if isinstance(frames, torch.Tensor):
        sampled = frames.index_select(0, positions.to(device=frames.device))
    else:
        indices = positions.tolist()
        try:
            sampled = frames[indices]
        except (IndexError, TypeError):
            sampled = [frames[index] for index in indices]

    if resolved_max_frames > 1 and num_frames > 1:
        effective_fps = resolved_fps * float(resolved_max_frames - 1) / float(num_frames - 1)
    else:
        effective_fps = resolved_fps
    return sampled, effective_fps


def sample_video_frames_pyav(
    path: str,
    *,
    target_fps: float,
    max_frames: Optional[int],
) -> Tuple[torch.Tensor, float]:
    """Decode a clip with bounded retained-frame memory.

    The decoder walks the stream once. When the retained buffer grows beyond
    ``2 * max_frames``, it keeps every other frame and doubles the stride. This
    preserves full-video coverage without materializing every sampled frame.
    """
    target_fps, resolved_max_frames = _validated_sampling_args(target_fps, max_frames)

    import av

    container = av.open(path)
    try:
        stream = container.streams.video[0]
        src_fps = float(stream.average_rate) if stream.average_rate else target_fps
        if src_fps <= 0.0:
            src_fps = target_fps
        decode_stride = max(1, round(src_fps / target_fps))
        buffer_limit = 2 * resolved_max_frames if resolved_max_frames is not None else None

        frames = []
        source_indices = []
        for index, frame in enumerate(container.decode(video=0)):
            if index % decode_stride != 0:
                continue
            array = frame.to_ndarray(format="rgb24")
            frames.append(torch.from_numpy(array).permute(2, 0, 1).contiguous())
            source_indices.append(index)

            if buffer_limit is not None and len(frames) > buffer_limit:
                frames = frames[::2]
                source_indices = source_indices[::2]
                decode_stride *= 2
    finally:
        container.close()

    if not frames:
        raise ValueError(f"pyav decoded no frames from video: {path}")

    if resolved_max_frames is not None and len(frames) > resolved_max_frames:
        positions = torch.linspace(0, len(frames) - 1, steps=resolved_max_frames).round().long().tolist()
        frames = [frames[position] for position in positions]
        source_indices = [source_indices[position] for position in positions]

    if len(source_indices) > 1 and source_indices[-1] > source_indices[0]:
        effective_fps = src_fps * float(len(source_indices) - 1) / float(source_indices[-1] - source_indices[0])
    else:
        effective_fps = min(src_fps, target_fps)
    return torch.stack(frames, dim=0), effective_fps


__all__ = ["limit_video_frames", "sample_video_frames_pyav"]
