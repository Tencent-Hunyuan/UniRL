"""Best-effort local sample observation sink, called on the driver."""

from __future__ import annotations

import json
import logging
import os
import wave
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
from uuid import uuid4

import torch

from unirl.types.sample_record import SampleRecord
from unirl.utils.media import normalize_waveform

logger = logging.getLogger(__name__)


def _decode_record(record: SampleRecord, stem: str) -> Tuple[str, Dict[str, Any]]:
    """Decode and serialize one record, before any of its files exist, so failures leave no media."""
    row = {**vars(record), "media": None}
    decoded: Dict[str, Any] = {}
    media = record.media
    if media is not None:
        paths: Dict[str, Any] = {}
        if media.image is not None:
            decoded["image"] = media.image
            paths["image"] = f"{stem}.png"
        if media.video is not None:
            video = media.video.detach().cpu()
            if video.ndim != 4 or video.shape[0] != 3 or video.shape[1] == 0:
                raise ValueError(f"Expected video [3, T, H, W] with T > 0, got {tuple(video.shape)}")
            decoded["video"] = video.clamp(0, 1).mul(255).byte().permute(1, 2, 3, 0).numpy()  # [T, H, W, C]
            paths["video"] = f"{stem}.mp4"
        if media.audio is not None:
            rate = media.audio_sample_rate
            if not isinstance(rate, int) or rate <= 0:
                raise ValueError(
                    f"Audio observation requires a positive primitive_metadata['audio']['sample_rate'], got {rate!r}"
                )
            decoded["audio"] = normalize_waveform(media.audio).mul(32767).to(torch.int16).contiguous().numpy()
            decoded["audio_sample_rate"] = rate
            paths["audio"] = f"{stem}.wav"
            paths["audio_sample_rate"] = rate
        row["media"] = paths
    return json.dumps(row, ensure_ascii=False, default=str), decoded


def _write_media(directory: Path, stem: str, decoded: Dict[str, Any], written: List[Path]) -> None:
    """Write one record's decoded media, recording each path in ``written`` before it is created."""
    if "image" in decoded:
        path = directory / f"{stem}.png"
        written.append(path)
        decoded["image"].save(path)
    if "video" in decoded:
        import av

        frames = decoded["video"]
        path = directory / f"{stem}.mp4"
        written.append(path)
        with av.open(str(path), mode="w") as container:
            stream = container.add_stream("libx264", rate=8)
            stream.width, stream.height = frames.shape[2], frames.shape[1]
            stream.pix_fmt = "yuv420p"
            for pixels in frames:
                for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
    if "audio" in decoded:
        samples = decoded["audio"]
        path = directory / f"{stem}.wav"
        written.append(path)
        with wave.open(str(path), "wb") as output:
            output.setnchannels(samples.shape[1])
            output.setsampwidth(2)
            output.setframerate(decoded["audio_sample_rate"])
            output.writeframes(samples.tobytes())


def _indexed_media(index: Path) -> set[Path]:
    """Return the local media paths owned by the previous index."""
    try:
        lines = index.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return set()
    paths = set()
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            logger.warning("Local sample observation could not read %s: %s", index, exc)
            continue
        media = row.get("media") if isinstance(row, dict) else None
        if not isinstance(media, dict):
            continue
        for key, suffix in (("image", ".png"), ("video", ".mp4"), ("audio", ".wav")):
            name = media.get(key)
            if isinstance(name, str) and Path(name).name == name and Path(name).suffix == suffix:
                paths.add(index.parent / name)
    return paths


def _remove_sample_files(paths: Iterable[Path]) -> None:
    """Best-effort cleanup for files written by the sample sink."""
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Local sample observation could not remove %s: %s", path, exc)


def dump_sample_records(root: str, records: List[SampleRecord]) -> None:
    """Write a step's JSONL index and PNG, MP4, or WAV files with relative paths."""
    if not records:
        return
    directory = Path(root) / records[0].phase / f"step_{records[0].step:06d}"
    index_path = directory / "samples.jsonl"
    write_id = uuid4().hex
    staged = directory / f".samples-{write_id}.jsonl"
    written: List[Path] = []
    try:
        directory.mkdir(parents=True, exist_ok=True)
        previous = _indexed_media(index_path)
        rows = []
        for index, record in enumerate(records):
            stem = f"{record.track}_{index}-{write_id}"
            record_files: List[Path] = []
            try:
                row, decoded = _decode_record(record, stem)
                _write_media(directory, stem, decoded, record_files)
            except Exception as exc:
                _remove_sample_files(record_files)
                logger.warning("Local sample observation skipped %s: %s", record.sample_id, exc)
                continue
            written.extend(record_files)
            rows.append(row)
        if not rows:
            logger.warning("Local sample observation kept %s: no record could be written.", index_path)
            return
        staged.write_text("".join(f"{row}\n" for row in rows), encoding="utf-8")
        os.replace(staged, index_path)
    except Exception as exc:
        _remove_sample_files([*written, staged])
        logger.warning("Local sample observation failed: %s", exc)
        return
    _remove_sample_files(previous)
