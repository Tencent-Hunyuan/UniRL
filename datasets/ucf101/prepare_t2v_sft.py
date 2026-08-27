"""Build UniRL target-video SFT manifests from an extracted UCF-101 tree."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

DEFAULT_CLASSES = (
    "Archery",
    "Basketball",
    "BasketballDunk",
    "Biking",
    "CliffDiving",
    "CricketBowling",
    "CricketShot",
    "Diving",
    "GolfSwing",
    "HorseRiding",
    "Kayaking",
    "Skiing",
    "Skijet",
    "SoccerJuggling",
    "Surfing",
    "TennisSwing",
    "VolleyballSpiking",
    "WalkingWithDog",
)


def _class_words(name: str) -> str:
    return re.sub(r"(?<!^)([A-Z])", r" \1", name).lower()


def _rows(data_root: Path, classes: tuple[str, ...]) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {}
    for class_name in classes:
        class_rows = []
        for path in sorted((data_root / class_name).glob("*.avi")):
            class_rows.append(
                {
                    "sample_id": path.stem,
                    "prompt": f"a person is {_class_words(class_name)}",
                    "media": [
                        {
                            "modality": "video",
                            "role": "target",
                            "uri": str(path.resolve()),
                        }
                    ],
                    "metadata": {"source": "UCF-101", "ucf101_class": class_name},
                }
            )
        if not class_rows:
            raise FileNotFoundError(f"no .avi videos found for class {class_name!r} under {data_root}")
        rows[class_name] = class_rows
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows):6d} rows -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--classes", nargs="+", default=list(DEFAULT_CLASSES))
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--min-val-per-class", type=int, default=4)
    parser.add_argument("--max-val-samples", type=int, default=128, help="0 = no cap")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0.0 < args.val_fraction < 1.0:
        parser.error("--val-fraction must be in (0, 1)")
    if args.min_val_per_class < 1:
        parser.error("--min-val-per-class must be >= 1")
    if args.max_val_samples < 0:
        parser.error("--max-val-samples must be >= 0")

    classes = tuple(args.classes)
    by_class = _rows(args.data_root, classes)
    rng = random.Random(args.seed)
    for class_rows in by_class.values():
        rng.shuffle(class_rows)

    val: list[dict[str, Any]] = []
    train: list[dict[str, Any]] = []
    for class_name in classes:
        class_rows = by_class[class_name]
        count = max(args.min_val_per_class, round(len(class_rows) * args.val_fraction))
        if count >= len(class_rows):
            parser.error(
                f"class {class_name!r} has {len(class_rows)} samples but the validation split requests {count}; "
                "reduce --val-fraction/--min-val-per-class so each class keeps at least one training sample"
            )
        val.extend(class_rows[:count])
        train.extend(class_rows[count:])
    rng.shuffle(val)
    if args.max_val_samples > 0:
        train.extend(val[args.max_val_samples :])
        val = val[: args.max_val_samples]
    rng.shuffle(train)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.out_dir / "train.jsonl", train)
    _write_jsonl(args.out_dir / "val.jsonl", val)
    print(f"classes={len(classes)} train={len(train)} val={len(val)} seed={args.seed}")


if __name__ == "__main__":
    main()
