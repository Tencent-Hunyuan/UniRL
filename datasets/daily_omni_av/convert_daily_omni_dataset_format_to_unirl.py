#!/usr/bin/env python3
"""Convert Daily-Omni audio/video MCQA data to UniRL JSONL."""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List

ANSWER_INSTRUCTION = (
    "Watch the video and listen to its audio, then answer the multiple-choice question.\n"
    "Reason step by step, then end your reply with the exact phrase: The answer is [X]"
)


def _answer(raw: Any) -> str:
    answer = str(raw).strip().upper()
    if len(answer) == 3 and answer[0] == "[" and answer[-1] == "]":
        answer = answer[1]
    if answer not in {"A", "B", "C", "D"}:
        raise ValueError(f"invalid ground-truth answer: {answer!r}")
    return answer


def _prompt(row: Dict[str, Any]) -> str:
    question = str(row["Question"]).strip()
    choices = [str(choice).strip() for choice in row["Choice"]]
    if not question or not choices:
        raise ValueError("row has no question or no choices")
    return "\n".join([question, *choices, ANSWER_INSTRUCTION])


def _group_by_video(qa_json: str, videos_root: str, keep_missing: bool) -> Dict[str, List[Dict[str, Any]]]:
    with open(qa_json, encoding="utf-8") as source:
        rows = json.load(source)

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    missing = 0
    for index, row in enumerate(rows):
        try:
            video_id = str(row["video_id"])
            video = os.path.abspath(os.path.join(videos_root, video_id, f"{video_id}_video.mp4"))
            if not keep_missing and not os.path.isfile(video):
                missing += 1
                continue
            grouped.setdefault(video_id, []).append(
                {
                    "prompt": _prompt(row),
                    "media_refs": [{"modality": "video", "role": "prompt", "uri": video}],
                    "metadata": {
                        "answer": _answer(row["Answer"]),
                        "video_id": video_id,
                        "qa_type": row.get("Type"),
                    },
                }
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{qa_json}[{index}]: {exc}") from exc

    if missing:
        print(f"[stats] skipped {missing} rows whose mp4 is not on disk (pass --keep-missing to emit them)")
    return grouped


def _split(grouped: Dict[str, List[Dict[str, Any]]], val_ratio: float, seed: int) -> Dict[str, List[Dict[str, Any]]]:
    video_ids = sorted(grouped)
    random.Random(seed).shuffle(video_ids)

    n_val = 0
    if val_ratio > 0 and len(video_ids) > 1:
        n_val = max(1, min(int(len(video_ids) * val_ratio), len(video_ids) - 1))
    val_ids = set(video_ids[:n_val])

    splits: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": []}
    for video_id in video_ids:
        name = "val" if video_id in val_ids else "train"
        for payload in grouped[video_id]:
            splits[name].append(
                {
                    "prompt": payload["prompt"],
                    "prompt_id": f"daily_omni_av:{name}:{len(splits[name]):06d}:{video_id}",
                    "media_refs": payload["media_refs"],
                    "metadata": payload["metadata"],
                }
            )
    return splits


def _write_split(records: List[Dict[str, Any]], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[write] {output_path}: {len(records)} rows")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--qa-json", required=True, help="official Daily-Omni qa.json")
    parser.add_argument("--videos-root", help="unpacked Videos/ tree (default: Videos/ next to qa.json)")
    parser.add_argument("--out-dir", default="datasets/daily_omni_av")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="fraction of videos held out for val.jsonl")
    parser.add_argument("--seed", type=int, default=42, help="shuffle seed (deterministic split)")
    parser.add_argument("--keep-missing", action="store_true", help="keep rows whose video is unavailable locally")
    args = parser.parse_args()

    qa_json = os.path.abspath(os.path.expanduser(args.qa_json))
    videos_root = (
        os.path.abspath(os.path.expanduser(args.videos_root))
        if args.videos_root
        else os.path.join(os.path.dirname(qa_json), "Videos")
    )

    grouped = _group_by_video(qa_json, videos_root, args.keep_missing)
    if not grouped:
        raise SystemExit(f"No usable rows. Is {videos_root} unpacked? (tar -xf Videos.tar)")

    splits = _split(grouped, args.val_ratio, args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    _write_split(splits["train"], os.path.join(args.out_dir, "train.jsonl"))
    if splits["val"]:
        _write_split(splits["val"], os.path.join(args.out_dir, "val.jsonl"))


if __name__ == "__main__":
    main()
