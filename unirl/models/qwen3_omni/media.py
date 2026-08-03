"""Shared media helpers for Qwen3-Omni audio-in-video processing."""

from __future__ import annotations

from typing import Optional

import numpy as np


def load_audio_pyav(path: str, target_sr: int = 16000) -> Optional[np.ndarray]:
    """Decode the first audio stream to a mono float32 waveform."""
    import av

    container = av.open(path)
    try:
        if not container.streams.audio:
            return None
        resampler = av.AudioResampler(format="flt", layout="mono", rate=int(target_sr))
        chunks = []
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray())
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray())
    finally:
        container.close()

    if not chunks:
        return None
    waveform = np.concatenate(chunks, axis=-1).reshape(-1).astype(np.float32)
    return waveform if waveform.size else None


def extract_audio_from_video_pyav(path: str, target_sr: int = 16000) -> Optional[np.ndarray]:
    """Decode a video's embedded audio track."""
    return load_audio_pyav(path, target_sr)


__all__ = ["extract_audio_from_video_pyav", "load_audio_pyav"]
