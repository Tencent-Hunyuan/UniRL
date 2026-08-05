#!/usr/bin/env python3
"""Convert gijs/dcase2025-audio-qa parquet shards to UniRL audio MCQA JSONL.

The source dataset embeds WAV bytes in parquet. This converter extracts each
clip once, normalizes four-way answers to A/B/C/D, and maps the labeled source
``test`` split to UniRL validation. The unlabeled source ``eval`` split is not
emitted.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

LETTERS = "ABCD"
DEFAULT_OUT_DIR = Path("datasets/dcase2025_audio_qa")


def _answer_letter(answer: Any, choices: list[str]) -> str | None:
    text = str(answer or "").strip()
    if len(choices) != 4 or len(text) < 2:
        return None
    letter = text[0].upper()
    if letter not in LETTERS or text[1] not in ".) :":
        return None
    return letter


def _answer_value(answer: Any) -> str:
    text = str(answer or "").strip()
    if len(text) >= 2 and text[0].upper() in LETTERS and text[1] in ".) :":
        return text[2:].strip()
    return text


def _build_prompt(question: str, choices: list[str]) -> str:
    options = "\n".join(f"{letter}. {choice}" for letter, choice in zip(LETTERS, choices))
    return (
        "Listen to the audio carefully, then answer the following multiple-choice question:\n\n"
        f"{question}\n\n{options}\n\n"
        "Reason step by step, then provide the final answer in the exact format: "
        "The answer is [X]."
    )


def _iter_rows(paths: Iterable[Path]) -> Iterable[tuple[Path, dict[str, Any]]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to convert the DCASE parquet shards") from exc

    columns = [
        "audio",
        "question_text",
        "choices",
        "answer",
        "id",
        "audio_url",
        "question_type",
        "subset",
    ]
    for path in paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=32, columns=columns):
            for row in batch.to_pylist():
                yield path, row


def _write_audio(path: Path, payload: bytes) -> bool:
    """Atomically write embedded audio and return whether a file was created."""
    if path.is_file() and path.stat().st_size == len(payload):
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return True


def _convert_split(
    source_dir: Path,
    out_dir: Path,
    *,
    source_split: str,
    output_name: str,
) -> Counter[str]:
    paths = sorted(source_dir.glob(f"{source_split}-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no {source_split}-*.parquet files found under {source_dir}")

    stats: Counter[str] = Counter()
    output_path = out_dir / output_name
    temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")
    audio_dir = out_dir / "audio" / source_split

    with temporary_output.open("w", encoding="utf-8") as output:
        for source_path, row in _iter_rows(paths):
            stats["source_rows"] += 1
            choices = [str(choice).strip() for choice in row.get("choices") or []]
            answer = row.get("answer")
            letter = _answer_letter(answer, choices)
            question = str(row.get("question_text") or "").strip()
            audio = row.get("audio") or {}
            payload = audio.get("bytes")
            audio_name = Path(str(audio.get("path") or "")).name

            if len(choices) != 4:
                stats["skipped_non_four_choice"] += 1
                continue
            if letter is None:
                stats["skipped_non_abcd_answer"] += 1
                continue
            if not question or not payload or not audio_name:
                stats["skipped_invalid_row"] += 1
                continue

            audio_path = audio_dir / audio_name
            if _write_audio(audio_path, payload):
                stats["audio_files_written"] += 1
                stats["audio_bytes_written"] += len(payload)
            else:
                stats["audio_files_reused"] += 1

            source_id = str(row.get("id") or audio_name).strip()
            record = {
                "prompt": _build_prompt(question, choices),
                "prompt_id": f"dcase2025:{source_split}:{source_id}:{stats['source_rows']}",
                "media_refs": [
                    {
                        "modality": "audio",
                        "role": "prompt",
                        "uri": str(Path("audio") / source_split / audio_name),
                    }
                ],
                "metadata": {
                    "answer": letter,
                    "answer_value": _answer_value(answer),
                    "choices": choices,
                    "source_id": source_id,
                    "source_split": source_split,
                    "question_type": row.get("question_type"),
                    "subset": row.get("subset"),
                    "audio_url": row.get("audio_url"),
                    "source_file": source_path.name,
                },
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            stats["written_rows"] += 1

    temporary_output.replace(output_path)
    return stats


def convert(snapshot: Path, out_dir: Path) -> dict[str, Any]:
    snapshot = snapshot.expanduser().resolve(strict=True)
    out_dir = out_dir.expanduser().resolve()
    source_dir = snapshot / "data"
    if not source_dir.is_dir():
        raise FileNotFoundError(f"dataset snapshot has no data directory: {source_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    split_stats = {
        "train": _convert_split(source_dir, out_dir, source_split="train", output_name="train.jsonl"),
        "val": _convert_split(source_dir, out_dir, source_split="test", output_name="val.jsonl"),
    }
    manifest = {
        "format": "UniRL prompt-first standalone-audio MCQA JSONL",
        "source": "gijs/dcase2025-audio-qa",
        "source_snapshot": snapshot.name,
        "split_mapping": {"train": "train", "val": "test"},
        "excluded_source_split": {"eval": "competition split has no public labels"},
        "stats": {split: dict(sorted(stats.items())) for split, stats in split_stats.items()},
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True, help="local Hugging Face dataset snapshot")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    manifest = convert(args.snapshot, args.out_dir)
    print(json.dumps(manifest["stats"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
