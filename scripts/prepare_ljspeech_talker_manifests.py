"""Convert an extracted LJSpeech release into deterministic Talker SFT manifests."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List


def _write_jsonl(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    tmp.replace(path)


def _load_rows(root: Path, *, speaker: str, language: str) -> List[Dict[str, object]]:
    metadata_path = root / "metadata.csv"
    wav_dir = root / "wavs"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"LJSpeech metadata is missing: {metadata_path}")
    if not wav_dir.is_dir():
        raise FileNotFoundError(f"LJSpeech wav directory is missing: {wav_dir}")

    rows: List[Dict[str, object]] = []
    with metadata_path.open(encoding="utf-8") as handle:
        # LJSpeech uses a literal pipe separator, not RFC CSV quoting. Some raw
        # transcripts begin with an unmatched quote, so csv.reader would merge
        # the raw/normalized fields. Split the first two pipes literally.
        for line_number, line in enumerate(handle, 1):
            fields = line.rstrip("\r\n").split("|", 2)
            if len(fields) != 3:
                raise ValueError(
                    f"{metadata_path}:{line_number}: expected id|raw|normalized, got {len(fields)} fields"
                )
            utterance_id, raw_text, normalized_text = (field.strip() for field in fields)
            text = normalized_text or raw_text
            wav_path = (wav_dir / f"{utterance_id}.wav").resolve()
            if not utterance_id or not text:
                raise ValueError(f"{metadata_path}:{line_number}: empty utterance id or transcript")
            if not wav_path.is_file():
                raise FileNotFoundError(f"{metadata_path}:{line_number}: missing {wav_path}")
            rows.append(
                {
                    "sample_id": utterance_id,
                    "text": text,
                    "audio": str(wav_path),
                    "speaker": speaker,
                    "language": language,
                    "metadata": {
                        "speaker": speaker,
                        "language": language,
                        "source": "LJSpeech-1.1",
                        "raw_transcript": raw_text,
                    },
                }
            )
    if len(rows) != 13_100:
        raise ValueError(f"Expected 13,100 LJSpeech rows, found {len(rows)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ljspeech_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--speaker", default="Ethan")
    parser.add_argument("--language", default="en")
    parser.add_argument("--val_size", type=int, default=256)
    parser.add_argument("--smoke_train_size", type=int, default=32)
    parser.add_argument("--smoke_val_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()

    rows = _load_rows(
        Path(args.ljspeech_dir).expanduser().resolve(),
        speaker=str(args.speaker),
        language=str(args.language),
    )
    if not 1 <= args.val_size < len(rows):
        raise ValueError(f"val_size must lie in [1, {len(rows) - 1}]")
    rng = random.Random(int(args.seed))
    rng.shuffle(rows)
    val_rows = rows[: args.val_size]
    train_rows = rows[args.val_size :]
    if not 1 <= args.smoke_train_size <= len(train_rows):
        raise ValueError("smoke_train_size is outside the train split")
    if not 1 <= args.smoke_val_size <= len(val_rows):
        raise ValueError("smoke_val_size is outside the validation split")

    output = Path(args.output_dir).expanduser().resolve()
    manifests = {
        "raw_train.jsonl": train_rows,
        "raw_val.jsonl": val_rows,
        "raw_smoke_train.jsonl": train_rows[: args.smoke_train_size],
        "raw_smoke_val.jsonl": val_rows[: args.smoke_val_size],
    }
    for name, split_rows in manifests.items():
        _write_jsonl(output / name, split_rows)

    print(
        json.dumps(
            {
                "output_dir": str(output),
                "train_rows": len(train_rows),
                "val_rows": len(val_rows),
                "smoke_train_rows": args.smoke_train_size,
                "smoke_val_rows": args.smoke_val_size,
                "speaker": args.speaker,
                "language": args.language,
                "seed": args.seed,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
