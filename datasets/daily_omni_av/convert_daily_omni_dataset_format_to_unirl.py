#!/usr/bin/env python3
"""Convert Daily-Omni audio/video MCQA data to UniRL JSONL.

The output contains only a video media reference. Qwen3-Omni extracts the
embedded audio track from that same file when ``use_audio_in_video=true``.

Example:
    python datasets/daily_omni_av/convert_daily_omni_dataset_format_to_unirl.py \
      --train-input /path/to/daily_omni_av_train.jsonl \
      --val-input /path/to/daily_omni_av_val.jsonl \
      --out-dir datasets/daily_omni_av
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable


def _user_text(row: Dict[str, Any]) -> str:
    messages = row.get("prompt") or []
    user = next((message for message in reversed(messages) if message.get("role") == "user"), None)
    if user is None:
        raise ValueError("row has no user message")
    content = user.get("content", "")
    if isinstance(content, str):
        return content.strip()
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            return str(part.get("text", "")).strip()
    raise ValueError("user message has no text content")


def _video_uri(row: Dict[str, Any]) -> str:
    videos = row.get("videos") or []
    if videos:
        first = videos[0]
        uri = first.get("video") if isinstance(first, dict) else first
        if uri:
            return os.path.abspath(os.path.expanduser(str(uri)))

    messages = row.get("prompt") or []
    for message in reversed(messages):
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "video" and part.get("video"):
                return os.path.abspath(os.path.expanduser(str(part["video"])))
    raise ValueError("row has no video path")


def _answer(row: Dict[str, Any]) -> str:
    answer = str((row.get("reward_model") or {}).get("ground_truth", "")).strip().upper()
    if len(answer) == 3 and answer[0] == "[" and answer[-1] == "]":
        answer = answer[1]
    if answer not in {"A", "B", "C", "D"}:
        raise ValueError(f"invalid ground-truth answer: {answer!r}")
    return answer


def _iter_records(path: str, split: str, keep_missing: bool) -> Iterable[Dict[str, Any]]:
    with open(path, encoding="utf-8") as source:
        output_index = 0
        for source_index, line in enumerate(source):
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                video = _video_uri(row)
                if not keep_missing and not os.path.isfile(video):
                    continue
                extra = row.get("extra_info") or {}
                video_id = str(extra.get("video_id") or source_index)
                yield {
                    "prompt": _user_text(row),
                    "prompt_id": f"daily_omni_av:{split}:{output_index:06d}:{video_id}",
                    "media_refs": [{"modality": "video", "role": "prompt", "uri": video}],
                    "metadata": {
                        "answer": _answer(row),
                        "video_id": video_id,
                        "qa_type": extra.get("qa_type"),
                    },
                }
                output_index += 1
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{source_index + 1}: {exc}") from exc


def _write_split(input_path: str, output_path: str, split: str, keep_missing: bool) -> int:
    count = 0
    with open(output_path, "w", encoding="utf-8") as output:
        for record in _iter_records(input_path, split, keep_missing):
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    print(f"[write] {output_path}: {count} rows")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-input", required=True, help="Daily-Omni training JSONL")
    parser.add_argument("--val-input", help="Daily-Omni validation JSONL")
    parser.add_argument("--out-dir", default="datasets/daily_omni_av")
    parser.add_argument("--keep-missing", action="store_true", help="keep rows whose video is unavailable locally")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    _write_split(
        args.train_input,
        os.path.join(args.out_dir, "train.jsonl"),
        "train",
        args.keep_missing,
    )
    if args.val_input:
        _write_split(
            args.val_input,
            os.path.join(args.out_dir, "val.jsonl"),
            "val",
            args.keep_missing,
        )


if __name__ == "__main__":
    main()
