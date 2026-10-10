"""Convert official DyRef metadata into shared UniRL SFT/RL manifests."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable


def _iter_records(path: Path) -> Iterable[dict[str, Any]]:
    """Read an official JSON array or JSONL metadata file."""
    if path.suffix == ".json":
        with path.open(encoding="utf-8") as source:
            records = json.load(source)
        if not isinstance(records, list):
            raise TypeError(f"{path}: expected a JSON list, got {type(records).__name__}.")
        yield from records
        return
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
        return
    raise ValueError(f"{path}: expected a .json or .jsonl metadata file.")


def _source_path(raw: Any, *, data_root: Path, context: str) -> Path:
    """Resolve and validate one official media path."""
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{context}: expected a non-empty path string, got {raw!r}.")
    path = Path(raw.strip())
    path = path if path.is_absolute() else data_root / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{context}: media file not found: {path}")
    return path


def _manifest_uri(path: Path, *, out_dir: Path) -> str:
    """Write a path relative to the generated manifest directory."""
    return Path(os.path.relpath(path, out_dir.resolve())).as_posix()


def _style_reference_index(paths: list[Path], *, declared: Any, context: str) -> int | None:
    """Locate the single CSD style reference from the official path layout."""
    matches = [index for index, path in enumerate(paths) if "/style/reference" in f"/{path.as_posix().lower()}"]
    if len(matches) > 1:
        raise ValueError(f"{context}: found multiple style/reference images at indices {matches}.")
    if declared is True and not matches:
        raise ValueError(f"{context}: is_style=true but no style/reference image was found.")
    if declared is False and matches:
        raise ValueError(f"{context}: is_style=false but a style/reference image exists.")
    return matches[0] if matches else None


def _reference_type(path: Path, *, context: str) -> str:
    """Infer one official DyRef reference type from its directory layout."""
    value = f"/{path.as_posix().lower()}"
    patterns = (
        ("/transfer_subjects/", "subject"),
        ("/background/", "background"),
        ("/style/reference", "style"),
        ("/lighting/", "lighting"),
        ("/pose/", "pose"),
    )
    matches = [kind for pattern, kind in patterns if pattern in value]
    if len(matches) != 1:
        raise ValueError(f"{context}: cannot infer one DyRef reference type from {path}.")
    return matches[0]


def _convert_record(record: Any, *, data_root: Path, out_dir: Path, position: int) -> dict[str, Any]:
    """Convert one official ``image/edit_image`` row without reordering refs."""
    if not isinstance(record, dict):
        raise TypeError(f"record {position}: expected an object, got {type(record).__name__}.")
    prompt = record.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"record {position}: missing non-empty prompt.")
    raw_references = record.get("edit_image")
    if not isinstance(raw_references, list) or not raw_references:
        raise ValueError(f"record {position}: edit_image must be a non-empty path list.")

    index = record.get("index", position)
    sample_id = f"dyref:{index}"
    references = [
        _source_path(path, data_root=data_root, context=f"record {position} edit_image[{ref_index}]")
        for ref_index, path in enumerate(raw_references)
    ]
    target = _source_path(record.get("image"), data_root=data_root, context=f"record {position} image")
    style_index = _style_reference_index(
        references,
        declared=record.get("is_style"),
        context=f"record {position}",
    )
    reference_types = [
        _reference_type(path, context=f"record {position} edit_image[{index}]") for index, path in enumerate(references)
    ]

    dyref_metadata: dict[str, Any] = {
        "index": index,
        "reference_count": len(references),
        "reference_types": reference_types,
        "style_reference_index": style_index,
        "is_style": style_index is not None,
    }
    for key in ("category", "subjects"):
        if record.get(key) is not None:
            dyref_metadata[key] = record[key]

    media = [
        {
            "modality": "image",
            "role": "condition",
            "uri": _manifest_uri(path, out_dir=out_dir),
        }
        for path in references
    ]
    media.append(
        {
            "modality": "image",
            "role": "target",
            "uri": _manifest_uri(target, out_dir=out_dir),
        }
    )
    return {
        "sample_id": sample_id,
        "prompt_id": sample_id,
        "prompt": prompt.strip(),
        "media": media,
        "metadata": {"source": "dyref", "dyref": dyref_metadata},
    }


def _convert_file(
    path: Path,
    *,
    data_root: Path,
    out_dir: Path,
    max_samples: int | None,
    reference_count: int | None,
) -> list[dict[str, Any]]:
    """Convert and validate one metadata source."""
    rows = []
    seen_ids: set[str] = set()
    for position, record in enumerate(_iter_records(path)):
        converted = _convert_record(record, data_root=data_root, out_dir=out_dir, position=position)
        if reference_count is not None and converted["metadata"]["dyref"]["reference_count"] != reference_count:
            continue
        sample_id = converted["sample_id"]
        if sample_id in seen_ids:
            raise ValueError(f"{path}: duplicate sample identity {sample_id!r}.")
        seen_ids.add(sample_id)
        rows.append(converted)
        if max_samples is not None and len(rows) >= max_samples:
            break
    if not rows:
        raise ValueError(f"{path}: no usable DyRef records.")
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write normalized rows as UTF-8 JSONL."""
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    """Run the DyRef converter CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Official train_data.json or DyRef RL JSONL.")
    parser.add_argument("--data-root", type=Path, required=True, help="Root joined with official image paths.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--val-input", type=Path, help="Optional official validation/test metadata.")
    parser.add_argument("--val-data-root", type=Path, help="Data root for --val-input; defaults to --data-root.")
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--reference-count", type=int, help="Keep only rows with exactly N ordered references.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.max_samples is not None and args.max_samples < 1:
        parser.error("--max-samples must be >= 1.")
    if args.reference_count is not None and args.reference_count < 1:
        parser.error("--reference-count must be >= 1.")
    if args.val_input is None and not 0.0 < args.val_fraction < 1.0:
        parser.error("--val-fraction must lie in (0, 1) when --val-input is absent.")

    out_dir = args.out_dir.resolve()
    train_rows = _convert_file(
        args.input.resolve(),
        data_root=args.data_root.resolve(),
        out_dir=out_dir,
        max_samples=args.max_samples,
        reference_count=args.reference_count,
    )
    if args.val_input is not None:
        val_rows = _convert_file(
            args.val_input.resolve(),
            data_root=(args.val_data_root or args.data_root).resolve(),
            out_dir=out_dir,
            max_samples=None,
            reference_count=args.reference_count,
        )
    else:
        random.Random(args.seed).shuffle(train_rows)
        val_size = max(1, round(len(train_rows) * args.val_fraction))
        val_rows, train_rows = train_rows[:val_size], train_rows[val_size:]
        if not train_rows:
            raise ValueError("Validation split consumed every row; provide more records or a smaller fraction.")

    print(f"validated {len(train_rows)} train and {len(val_rows)} validation rows")
    if args.dry_run:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out_dir / "train.jsonl", train_rows)
    _write_jsonl(out_dir / "val.jsonl", val_rows)
    print(f"wrote {out_dir / 'train.jsonl'}")
    print(f"wrote {out_dir / 'val.jsonl'}")


if __name__ == "__main__":
    main()
