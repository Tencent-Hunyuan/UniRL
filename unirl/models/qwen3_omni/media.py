"""Shared media helpers for Qwen3-Omni prompt processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import PIL.Image

from .video import limit_video_frames, sample_video_frames_pyav


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


def load_qwen3_audio(path: str, target_sr: int) -> Optional[Tuple[np.ndarray, int]]:
    """Return UniRL's canonical waveform plus its explicit sampling rate.

    The waveform is decoded once through the already-aligned PyAV path. HF
    consumes the array while vLLM consumes ``(array, sample_rate)``; vLLM then
    skips resampling because the declared rate equals its processor target.
    """
    waveform = load_audio_pyav(path, target_sr)
    if waveform is None:
        return None
    return np.ascontiguousarray(waveform, dtype=np.float32), int(target_sr)


def load_image_rgb(path: str) -> PIL.Image.Image:
    """Load one prompt image in the RGB mode expected by Qwen processors."""
    source: object = path
    if path.startswith(("http://", "https://")):
        from io import BytesIO
        from urllib.request import urlopen

        with urlopen(path, timeout=30) as response:
            source = BytesIO(response.read())
    elif path.startswith(("s3://", "gs://")):
        raise NotImplementedError(
            f"Qwen3-Omni image URI scheme is not supported: {path!r}; materialize to a local path or HTTP(S) URL first."
        )
    with PIL.Image.open(source) as image:
        return image.convert("RGB")


@dataclass
class PreparedOmniMedia:
    messages: list[dict]
    image: Optional[object]
    video_frames: Optional[object]
    effective_fps: float
    audio_waveform: Optional[np.ndarray]
    audio_sample_rate: Optional[int]
    audio_in_video: bool


def prepare_omni_media(
    messages: list[dict],
    *,
    sample_rate: int,
    video_fps: float,
    video_max_frames: Optional[int],
    use_audio_in_video: bool,
) -> PreparedOmniMedia:
    """Decode one row's typed URI media without duplicating consumer logic."""
    prepared: list[dict] = []
    image: Optional[object] = None
    video_frames: Optional[object] = None
    effective_fps = float(video_fps)
    audio_waveform: Optional[np.ndarray] = None
    audio_sample_rate: Optional[int] = None
    audio_in_video = False
    has_standalone_audio = any(
        isinstance(message.get("content"), list) and any(block.get("type") == "audio" for block in message["content"])
        for message in messages
    )

    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            prepared.append(dict(message))
            continue
        blocks: list[dict] = []
        for raw_block in content:
            block = dict(raw_block)
            block_type = block.get("type")
            if block_type == "image":
                if image is not None:
                    raise ValueError("Qwen3-Omni currently supports at most one image per conversation.")
                raw_image = block.get("image")
                image = load_image_rgb(raw_image) if isinstance(raw_image, str) else raw_image
                if image is None:
                    raise ValueError("Qwen3-Omni image block has no payload.")
                block["image"] = image
            elif block_type == "audio":
                if audio_waveform is not None:
                    raise ValueError("Qwen3-Omni currently supports at most one audio per conversation.")
                raw_audio = block.get("audio")
                if not isinstance(raw_audio, str):
                    raise ValueError(
                        "Qwen3-Omni prompt audio must be a URI-backed MediaRef; "
                        "waveform inputs have no source sampling-rate contract."
                    )
                loaded = load_qwen3_audio(raw_audio, sample_rate)
                if loaded is None:
                    raise ValueError(f"Qwen3-Omni decoded no audio samples from {raw_audio!r}.")
                audio_waveform, audio_sample_rate = loaded
                block["audio"] = audio_waveform
            elif block_type == "video":
                if video_frames is not None:
                    raise ValueError("Qwen3-Omni currently supports at most one video per conversation.")
                raw_video = block.get("video")
                if isinstance(raw_video, str):
                    video_frames, effective_fps = sample_video_frames_pyav(
                        raw_video,
                        target_fps=video_fps,
                        max_frames=video_max_frames,
                    )
                    if use_audio_in_video and not has_standalone_audio:
                        loaded = load_qwen3_audio(raw_video, sample_rate)
                        if loaded is not None:
                            audio_waveform, audio_sample_rate = loaded
                            audio_in_video = True
                else:
                    video_frames, effective_fps = limit_video_frames(
                        raw_video,
                        fps=video_fps,
                        max_frames=video_max_frames,
                    )
                block["video"] = video_frames
            blocks.append(block)
        copied = dict(message)
        copied["content"] = blocks
        prepared.append(copied)

    return PreparedOmniMedia(
        messages=prepared,
        image=image,
        video_frames=video_frames,
        effective_fps=effective_fps,
        audio_waveform=audio_waveform,
        audio_sample_rate=audio_sample_rate,
        audio_in_video=audio_in_video,
    )


__all__ = [
    "PreparedOmniMedia",
    "extract_audio_from_video_pyav",
    "load_audio_pyav",
    "load_image_rgb",
    "load_qwen3_audio",
    "prepare_omni_media",
]
