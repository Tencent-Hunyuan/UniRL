"""Build a Qwen3-Omni video-MCQA jsonl (the format the video GSPO recipe expects).

Reads a manifest of MCQA cases and emits the train/val jsonl + a ``videos/``
directory of symlinks that ``examples/ar/qwen3_omni_video_gspo_lora.yaml``
consumes. This is a generic template — point ``--manifest`` at your own dataset;
it is intentionally not bound to any specific corpus.

Manifest (``--manifest``): a jsonl with one object per case::

    {"video": "/abs/or/rel/path/to/clip.mp4",
     "question": "<question text including the A/B/C/D options>",
     "answer": "B"}

``question`` should already contain the multiple-choice options and any answer-
format instruction; ``answer`` is the letter scored by mc_exact_match.

Output (under ``--out-dir``)::

    train.jsonl / val.jsonl   # {prompt, prompt_id, media_refs:[(video,prompt,uri)], metadata:{answer}}
    videos/<id>.mp4           # symlinks to the source video (uri is relative to the jsonl dir)

The ``(video, prompt)`` media-ref role hands the raw video path to the Qwen3-Omni
processor, which samples frames itself (fps-driven) so video_grid_thw /
second_per_grid match its TMRoPE (see unirl/data/data_source.py::_load_prompt_videos).

Usage::

    python scripts/build_qwen3_omni_video_mcqa.py \
        --manifest my_cases.jsonl --out-dir datasets/my_video_mcqa --val-count 1
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List


def _load_manifest(manifest_path: str) -> List[Dict]:
    """Read the manifest jsonl into validated ``{id, question, answer, video}`` cases."""
    cases: List[Dict] = []
    with open(manifest_path, encoding="utf-8") as f:
        for lineno, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            video = obj.get("video")
            question = obj.get("question")
            answer = obj.get("answer")
            if not (video and question and answer):
                print(f"[skip] line {lineno}: missing video / question / answer")
                continue
            if not os.path.isfile(video):
                print(f"[skip] line {lineno}: video not found: {video}")
                continue
            cases.append(
                {
                    "id": str(obj.get("id", lineno)),
                    "question": str(question),
                    "answer": str(answer).strip(),
                    "video_path": video,
                }
            )
    return cases


def _write_split(cases: List[Dict], jsonl_path: str, videos_dir: str) -> None:
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for c in cases:
            link = os.path.join(videos_dir, f"{c['id']}.mp4")
            if os.path.islink(link) or os.path.exists(link):
                os.remove(link)
            os.symlink(os.path.abspath(c["video_path"]), link)
            row = {
                "prompt": c["question"],
                "prompt_id": f"video_mcqa:{c['id']}",
                # (video, prompt): hand the raw path to the processor (it samples frames).
                "media_refs": [{"modality": "video", "role": "prompt", "uri": f"videos/{c['id']}.mp4"}],
                "metadata": {"answer": c["answer"]},
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[write] {jsonl_path}: {len(cases)} rows")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="jsonl manifest: {video, question, answer} per line")
    ap.add_argument("--out-dir", required=True, help="output dataset dir")
    ap.add_argument("--val-count", type=int, default=1, help="last N cases go to val.jsonl (0 = reuse train as val)")
    args = ap.parse_args()

    cases = _load_manifest(args.manifest)
    if not cases:
        raise SystemExit(f"no usable cases found in {args.manifest}")

    videos_dir = os.path.join(args.out_dir, "videos")
    os.makedirs(videos_dir, exist_ok=True)

    n_val = max(0, min(int(args.val_count), len(cases) - 1))
    train_cases = cases[: len(cases) - n_val] if n_val else cases
    val_cases = cases[len(cases) - n_val :] if n_val else cases  # reuse train as val when val_count=0

    _write_split(train_cases, os.path.join(args.out_dir, "train.jsonl"), videos_dir)
    _write_split(val_cases, os.path.join(args.out_dir, "val.jsonl"), videos_dir)
    print(f"[done] {len(cases)} cases → {args.out_dir} (train={len(train_cases)}, val={len(val_cases)})")


if __name__ == "__main__":
    main()
