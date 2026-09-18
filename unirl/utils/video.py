"""Shared video frame loading and bounded sampling helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import torch

_REMOTE_URI_PREFIXES = ("http://", "https://", "s3://", "gs://")
_TENSOR_VIDEO_SUFFIXES = frozenset({".pt", ".pth", ".npy", ".npz"})


def load_video(uri: str, *, max_frames: Optional[int] = None) -> torch.Tensor:
    """Load local video media as RGB ``[T, 3, H, W]`` float32 in ``[0, 1]``."""
    if uri.startswith(_REMOTE_URI_PREFIXES):
        raise NotImplementedError(
            f"Remote video URI {uri!r} is not supported; materialize it to local/shared storage first."
        )
    resolved_max_frames = _validate_max_frames(max_frames)
    suffix = Path(uri).suffix.lower()
    if suffix in _TENSOR_VIDEO_SUFFIXES:
        frames = _load_tensor_frames(uri)
        if resolved_max_frames is not None:
            frames, _ = limit_video_frames(frames, fps=1.0, max_frames=resolved_max_frames)
    else:
        frames, _ = sample_video_frames_pyav(uri, target_fps=None, max_frames=resolved_max_frames)
    return _normalize_video_frames(frames, uri=uri)


def _load_tensor_frames(uri: str) -> torch.Tensor:
    suffix = Path(uri).suffix.lower()
    if suffix in {".pt", ".pth"}:
        return torch.load(uri, map_location="cpu", weights_only=True)

    import numpy as np

    loaded = np.load(uri)
    if suffix == ".npy":
        return torch.from_numpy(loaded)
    with loaded as archive:
        if "frames" not in archive:
            raise ValueError(f"Video fixture {uri!r} must contain a 'frames' array.")
        return torch.from_numpy(archive["frames"])


def _normalize_video_frames(frames: Any, *, uri: str) -> torch.Tensor:
    frames = torch.as_tensor(frames)
    if frames.ndim != 4 or int(frames.shape[1]) != 3:
        raise ValueError(f"Expected video frames [T, 3, H, W], got {tuple(frames.shape)} from {uri!r}.")
    if int(frames.shape[0]) < 1:
        raise ValueError(f"Decoded no video frames from {uri!r}.")
    if frames.dtype == torch.uint8:
        return frames.to(dtype=torch.float32).div_(255.0)
    return frames.to(dtype=torch.float32).clamp_(0.0, 1.0)


def _validate_max_frames(max_frames: Optional[int]) -> Optional[int]:
    if max_frames is None:
        return None
    resolved_max_frames = int(max_frames)
    if resolved_max_frames < 1:
        raise ValueError(f"max_frames must be >= 1, got {max_frames!r}")
    return resolved_max_frames


def _validated_sampling_args(fps: float, max_frames: Optional[int]) -> Tuple[float, Optional[int]]:
    resolved_fps = float(fps)
    if resolved_fps <= 0.0:
        raise ValueError(f"video fps must be > 0, got {fps!r}")
    return resolved_fps, _validate_max_frames(max_frames)


def limit_video_frames(frames: Any, *, fps: float, max_frames: Optional[int]) -> Tuple[Any, float]:
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
    target_fps: Optional[float],
    max_frames: Optional[int],
) -> Tuple[torch.Tensor, float]:
    """Decode a clip with optional FPS sampling and bounded full-clip coverage."""
    resolved_max_frames = _validate_max_frames(max_frames)

    import av

    container = av.open(path)
    try:
        stream = container.streams.video[0]
        src_fps = float(stream.average_rate) if stream.average_rate else 0.0
        if src_fps <= 0.0:
            src_fps = float(target_fps) if target_fps is not None else 1.0
        if target_fps is None:
            decode_stride = 1
        else:
            target_fps, _ = _validated_sampling_args(target_fps, resolved_max_frames)
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
        raise ValueError(f"PyAV decoded no frames from video: {path}")

    if resolved_max_frames is not None and len(frames) > resolved_max_frames:
        positions = torch.linspace(0, len(frames) - 1, steps=resolved_max_frames).round().long().tolist()
        frames = [frames[position] for position in positions]
        source_indices = [source_indices[position] for position in positions]

    if len(source_indices) > 1 and source_indices[-1] > source_indices[0]:
        effective_fps = src_fps * float(len(source_indices) - 1) / float(source_indices[-1] - source_indices[0])
    else:
        effective_fps = min(src_fps, target_fps) if target_fps is not None else src_fps
    return torch.stack(frames, dim=0), effective_fps


__all__ = ["limit_video_frames", "load_video", "sample_video_frames_pyav"]
