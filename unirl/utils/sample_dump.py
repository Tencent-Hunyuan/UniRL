"""Best-effort local sample observation sink, called on the driver."""

from __future__ import annotations

import json
import logging
import wave
from pathlib import Path
from typing import List

import torch

from unirl.types.sample_record import SampleRecord

logger = logging.getLogger(__name__)


def dump_sample_records(root: str, records: List[SampleRecord]) -> None:
    """Write a step's JSONL index and PNG, MP4, or WAV files with relative paths."""
    if not records:
        return
    try:
        directory = Path(root) / records[0].phase / f"step_{records[0].step:06d}"
        directory.mkdir(parents=True, exist_ok=True)
        rows = []
        for index, record in enumerate(records):
            row = {**vars(record), "media": None}
            media = record.media
            if media is not None:
                paths = {}
                stem = f"{record.track}_{index}"
                if media.image is not None:
                    path = directory / f"{stem}.png"
                    media.image.save(path)
                    paths["image"] = path.name
                if media.video is not None:
                    import av

                    frames = media.video.clamp(0, 1).mul(255).byte().permute(1, 2, 3, 0).numpy()  # [T, H, W, C]
                    path = directory / f"{stem}.mp4"
                    with av.open(str(path), mode="w") as container:
                        stream = container.add_stream("libx264", rate=8)
                        stream.width, stream.height = frames.shape[2], frames.shape[1]
                        stream.pix_fmt = "yuv420p"
                        for pixels in frames:
                            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                                container.mux(packet)
                        for packet in stream.encode():
                            container.mux(packet)
                    paths["video"] = path.name
                if media.audio is not None:
                    if media.audio_sample_rate is None:
                        raise ValueError("Audio observation requires primitive_metadata['audio']['sample_rate']")
                    audio = media.audio.detach().cpu().float()
                    if audio.ndim == 1:
                        audio = audio[:, None]
                    elif audio.shape[0] in (1, 2):
                        audio = audio.T
                    path = directory / f"{stem}.wav"
                    with wave.open(str(path), "wb") as output:
                        output.setnchannels(audio.shape[1])
                        output.setsampwidth(2)
                        output.setframerate(media.audio_sample_rate)
                        output.writeframes(audio.clamp(-1, 1).mul(32767).to(torch.int16).contiguous().numpy().tobytes())
                    paths["audio"] = path.name
                    paths["audio_sample_rate"] = media.audio_sample_rate
                row["media"] = paths
            rows.append(json.dumps(row, ensure_ascii=False))
        (directory / "samples.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    except Exception as exc:
        logger.warning("Local sample observation failed: %s", exc)
