"""Build the local geo3k_mc dataset that the ``qwen_vl_grpo_geo3k_mc_*`` recipes train on.

The multimodal RL data loader (``unirl/data/data_source.py`` + ``unirl/data/datasets.py``)
reads a **local** jsonl of records::

    {"prompt": <str>,
     "prompt_id": "geo3k:<id>",
     "media_refs": [{"modality": "image", "role": "condition", "uri": "images/<id>.png"}],
     "metadata": {"answer": "<A|B|C|D>", "answer_value": <str>, "choices": [...], ...}}

plus an ``images/`` subdir next to the jsonl — relative ``uri``\\ s are resolved against
the jsonl's directory (``datasets._resolve_media_uri``). It does not accept HuggingFace
dataset ids. This tool converts the native multiple-choice Geometry3K dataset into that
layout so the recipes are reproducible without any private cache:

  - source : ``xyliu6/geometry3k`` — Geometry3K in its native 4-way multiple-choice form
             (diagram image + ``problem`` + ``choices`` + ``ground_truth`` letter).
  - splits : ``train`` -> train.jsonl, ``validation`` -> val.jsonl, ``test`` -> test.jsonl.
             The recipes read ``DATA_PATH=.../train.jsonl`` and ``EVAL_DATA_PATH=.../val.jsonl``.

The reward (``mc_exact_match``) only needs ``metadata.answer`` (the letter); the other
metadata fields are informational. The diagram rides on ``media_refs`` (not inline), so
the ``<image>`` placeholder is stripped from the prompt text.

Usage:
  python -m unirl.utils.prepare_geo3k_mc --out-dir data/geo3k_mc
  # -> data/geo3k_mc/{train.jsonl, val.jsonl, test.jsonl} + data/geo3k_mc/images/

  QWEN_VL_PATH=Qwen/Qwen2.5-VL-7B-Instruct \
  DATA_PATH=data/geo3k_mc/train.jsonl EVAL_DATA_PATH=data/geo3k_mc/val.jsonl \
  python -m unirl.train_ar --config-name=ar/qwen_vl_grpo_geo3k_mc_4x8 num_devices=32

The HF id / split names below are sensible defaults; override with the flags if your
source differs. The extractor is schema-tolerant (problem/question, ground_truth/answer).
"""

from __future__ import annotations

import argparse
import json
import os

_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]
PROMPT_PREFIX = "Look at the geometry diagram. "
PROMPT_SUFFIX = "Answer with the letter only."


def _clean_problem(text: str) -> str:
    """Drop the inline ``<image>`` placeholder; the diagram is delivered via media_refs."""
    return (text or "").replace("<image>", "").strip()


def _build_prompt(problem: str, choices: list) -> str:
    """``<prefix><problem>\\n\\nA) ..\\nB) ..\\n\\n<suffix>`` — matches the on-disk format."""
    lines = [f"{_LETTERS[i]}) {c}" for i, c in enumerate(choices)]
    return f"{PROMPT_PREFIX}{_clean_problem(problem)}\n\n" + "\n".join(lines) + f"\n\n{PROMPT_SUFFIX}"


def _answer_letter(row: dict, choices: list) -> str | None:
    """The ground-truth choice letter (schema-tolerant)."""
    for key in ("ground_truth", "answer_letter", "correct"):
        gt = row.get(key)
        if gt is not None and str(gt).strip()[:1].upper() in _LETTERS[: len(choices)]:
            return str(gt).strip()[:1].upper()
    # fall back: match the answer value against the choices
    ans = row.get("answer")
    if ans is not None and choices:
        ans = str(ans).strip()
        for i, c in enumerate(choices):
            if str(c).strip() == ans:
                return _LETTERS[i]
    return None


def _convert(hf_id: str, split: str, out_path: str, images_dir: str) -> int:
    from datasets import load_dataset

    ds = load_dataset(hf_id, split=split)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)
    n = 0
    with open(out_path, "w") as f:
        for idx, row in enumerate(ds):
            choices = [str(c) for c in (row.get("choices") or [])]
            problem = row.get("problem") or row.get("question") or ""
            images = row.get("images") or ([row["image"]] if row.get("image") is not None else [])
            letter = _answer_letter(row, choices)
            if not choices or not problem or not images or letter is None:
                continue  # skip rows we cannot turn into a verifiable MC example
            sid = row.get("id")
            sid = int(sid) if sid is not None else idx
            images[0].convert("RGB").save(os.path.join(images_dir, f"{sid}.png"))
            record = {
                "prompt": _build_prompt(problem, choices),
                "prompt_id": f"geo3k:{sid}",
                "media_refs": [{"modality": "image", "role": "condition", "uri": f"images/{sid}.png"}],
                "metadata": {
                    "answer": letter,
                    "answer_value": str(row.get("answer", "")).strip(),
                    "choices": choices,
                    "source_id": sid,
                    "split": split,
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
    print(f"  wrote {n} records -> {out_path}  (+ images in {images_dir}/)  from {hf_id}:{split}")
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="data/geo3k_mc", help="output directory for the jsonl files + images/")
    ap.add_argument("--hf", default="xyliu6/geometry3k", help="HF id for the native multiple-choice Geometry3K")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="validation")
    ap.add_argument("--test-split", default="test", help="set to '' to skip the test split")
    args = ap.parse_args()

    try:
        import datasets  # noqa: F401
    except ImportError:
        raise SystemExit("This tool needs `datasets` + `pillow`: pip install datasets pillow")

    images_dir = os.path.join(args.out_dir, "images")
    splits = [(args.train_split, "train.jsonl"), (args.val_split, "val.jsonl")]
    if args.test_split:
        splits.append((args.test_split, "test.jsonl"))
    for split, fname in splits:
        print(f"Building {fname} (split={split}):")
        _convert(args.hf, split, os.path.join(args.out_dir, fname), images_dir)


if __name__ == "__main__":
    main()
