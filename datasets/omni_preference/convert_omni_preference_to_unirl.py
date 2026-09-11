#!/usr/bin/env python
"""Convert Omni-Preference ``final_rl_data.jsonl`` into UniRL preference manifests."""

from __future__ import annotations

import argparse
import html
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

AUDIO_FILE_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")
VIDEO_FILE_EXTENSIONS = (".mp4", ".webm", ".mkv", ".mov", ".avi")

# The rubric prompt ends with a "### Context" block holding the media path, question and both candidates.
CONTEXT_PATTERN = re.compile(
    r"(?:Image|Video|Audio) file:\s*(?P<media>.+?)\s+Question:\s*(?P<question>.+?)\s+"
    r"Candidate A:\s*(?P<candidate_a>.+?)\s+Candidate B:\s*(?P<candidate_b>.+?)\s*\Z",
    re.DOTALL,
)

MODALITY_CONFIGS = {
    "image": {
        "jsonl_relpath": "dataset_jsonl/image/final_rl_data.jsonl",
        "media_field": "images",
        "media_subdir": None,
        "placeholder": "<image>",
    },
    "video": {
        "jsonl_relpath": "dataset_jsonl/video/final_rl_data.jsonl",
        "media_field": "videos",
        "media_subdir": "video-dataset",
        "placeholder": "<video>",
    },
    "audio": {
        "jsonl_relpath": "dataset_jsonl/audio/final_rl_data.jsonl",
        "media_field": "audios",
        "media_subdir": "audio_files",
        "placeholder": "<audio>",
    },
}


def _dataset_media_rel(media_path: str) -> str:
    """Strip the dataset's ``/data/`` prefix and the trailing ``_`` artifact seen in the context block."""
    normalized = media_path.replace("\\", "/").strip().rstrip("_")
    if normalized.startswith("/data/"):
        return normalized[len("/data/") :]
    return normalized.lstrip("/")


def _parse_context(content: str) -> Optional[Dict[str, str]]:
    if "### Context" not in content:
        return None
    match = CONTEXT_PATTERN.search(content.split("### Context", 1)[1].strip())
    if match is None:
        return None
    return {
        "media": match.group("media").strip().rstrip("_"),
        "question": match.group("question").strip(),
        "candidate_a": match.group("candidate_a").strip(),
        "candidate_b": match.group("candidate_b").strip(),
    }


def _basename_variants(name: str) -> str:
    """Fold the jsonl's escaped/underscored basenames onto the on-disk spelling (``&amp;``, ``_`` for space)."""
    return html.unescape(name).replace("_", " ").strip().lower()


def _build_basename_index(root: str, extensions: tuple) -> Dict[str, str]:
    """Shortest absolute path per basename, keyed both exactly and folded — the dataset nests media unevenly."""
    index: Dict[str, str] = {}
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if Path(filename).suffix.lower() not in extensions:
                continue
            abs_path = os.path.abspath(os.path.join(dirpath, filename))
            for key in (filename, _basename_variants(filename)):
                existing = index.get(key)
                if existing is None or len(abs_path) < len(existing):
                    index[key] = abs_path
    return index


def _resolve_media(dataset_root: str, modality: str, media_rel: str, basename_index: Dict[str, str]) -> Optional[str]:
    subdir = MODALITY_CONFIGS[modality]["media_subdir"]
    name = Path(media_rel).name
    for key in (name, _basename_variants(name)):
        indexed = basename_index.get(key)
        if indexed is not None:
            return indexed
    candidates = []
    if subdir:
        candidates.append(os.path.join(dataset_root, subdir, media_rel))
    candidates.append(os.path.join(dataset_root, media_rel))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
        if not Path(media_rel).suffix:
            for extension in VIDEO_FILE_EXTENSIONS:
                if os.path.isfile(candidate + extension):
                    return os.path.abspath(candidate + extension)
    return None


def _normalize(record: Dict[str, Any], modality: str, index: int) -> Optional[Dict[str, Any]]:
    """One raw RL row → the intermediate preference dict, or None when it must be dropped."""
    media_items = record.get(MODALITY_CONFIGS[modality]["media_field"]) or []
    messages = record.get("messages") or []
    if not media_items or not messages:
        return None
    try:
        solution = json.loads(record["solution"])
    except (KeyError, json.JSONDecodeError, TypeError):
        return None
    better = str(solution.get("better", "")).strip()
    if better not in ("A", "B"):
        return None
    context = _parse_context(str(messages[0].get("content", "")))
    if context is None:
        return None
    try:
        score_a = float(solution.get("score_A", 0))
        score_b = float(solution.get("score_B", 0))
    except (TypeError, ValueError):
        return None
    if better == "A":
        chosen, rejected, win, lose = context["candidate_a"], context["candidate_b"], score_a, score_b
    else:
        chosen, rejected, win, lose = context["candidate_b"], context["candidate_a"], score_b, score_a
    if not context["question"] or not chosen or not rejected:
        return None
    return {
        "modality": modality,
        "dataset_media_rel": _dataset_media_rel(str(media_items[0]).strip()),
        "question": context["question"],
        "chosen": chosen,
        "rejected": rejected,
        "win_score": win,
        "lose_score": lose,
        "better": better,
        "sample_id": f"omni_pref_{modality}_{index}",
    }


def _test_media_keys(records: List[Dict[str, Any]], test_ratio: float, seed: int) -> set:
    """Hold out whole media files so one medium never appears in both splits."""
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[Path(record["dataset_media_rel"]).name].append(record)
    keys = sorted(grouped)
    random.Random(seed).shuffle(keys)
    n_test = int(round(len(keys) * test_ratio))
    return set(keys[:n_test])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, help="Omni-Preference clone or HF snapshot directory")
    parser.add_argument("--modality", required=True, choices=sorted(MODALITY_CONFIGS))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=0, help="0 = no cap; useful for smoke manifests")
    args = parser.parse_args()

    modality = args.modality
    jsonl_path = os.path.join(args.snapshot, MODALITY_CONFIGS[modality]["jsonl_relpath"])
    if not os.path.isfile(jsonl_path):
        raise SystemExit(f"Missing source jsonl: {jsonl_path}")

    raw_rows = [json.loads(line) for line in open(jsonl_path, encoding="utf-8") if line.strip()]
    normalized = [n for i, row in enumerate(raw_rows) if (n := _normalize(row, modality, i)) is not None]

    extensions = AUDIO_FILE_EXTENSIONS if modality == "audio" else VIDEO_FILE_EXTENSIONS
    search_root = os.path.join(args.snapshot, MODALITY_CONFIGS[modality]["media_subdir"] or "")
    basename_index = _build_basename_index(search_root, extensions) if modality != "image" else {}

    resolved: List[Dict[str, Any]] = []
    missing = 0
    for record in normalized:
        media_abs = _resolve_media(args.snapshot, modality, record["dataset_media_rel"], basename_index)
        if media_abs is None:
            missing += 1
            continue
        record["media_abs"] = media_abs
        resolved.append(record)

    if args.max_rows:
        resolved = resolved[: args.max_rows]

    test_keys = _test_media_keys(resolved, args.test_ratio, args.seed)
    placeholder = MODALITY_CONFIGS[modality]["placeholder"]
    os.makedirs(args.out_dir, exist_ok=True)
    counts = {"train": 0, "val": 0}
    handles = {split: open(os.path.join(args.out_dir, f"{split}.jsonl"), "w", encoding="utf-8") for split in counts}
    try:
        for record in resolved:
            split = "val" if Path(record["dataset_media_rel"]).name in test_keys else "train"
            row = {
                "sample_id": record["sample_id"],
                # The placeholder marks where the chat stage splices the medium into the rendered prompt.
                "prompt": f"{placeholder}{record['question']}",
                "chosen": record["chosen"],
                "rejected": record["rejected"],
                "media_refs": [{"modality": modality, "role": "prompt", "uri": record["media_abs"]}],
                "metadata": {
                    "modality": modality,
                    "win_score": record["win_score"],
                    "lose_score": record["lose_score"],
                    "better": record["better"],
                },
            }
            handles[split].write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[split] += 1
    finally:
        for handle in handles.values():
            handle.close()

    manifest = {
        "source": "Omni-RRM/Omni-Preference",
        "modality": modality,
        "raw_rows": len(raw_rows),
        "normalized": len(normalized),
        "skipped_missing_media": missing,
        "dropped_not_a_or_b": len(raw_rows) - len(normalized),
        "train": counts["train"],
        "val": counts["val"],
        "test_ratio": args.test_ratio,
        "seed": args.seed,
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
