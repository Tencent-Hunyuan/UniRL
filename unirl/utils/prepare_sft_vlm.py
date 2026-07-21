"""Build a local VLM SFT manifest (image + prompt → response) from an HF dataset.

Converts a LLaVA-style visual-instruct dataset into the supervised manifest
layout (``unirl/data/sft.py``)::

    {"prompt": <str>, "response": <str>,
     "media": [{"modality": "image", "role": "condition", "uri": "images/<id>.png"}]}

plus an ``images/`` subdir next to the jsonl. Default source is
``HuggingFaceH4/llava-instruct-mix-vsft`` (messages + PIL images); rows are
flattened to the FIRST user→assistant round (the supervised source is
single-turn v1). Any dataset with the same ``messages``/``images`` schema works.

Usage:
  python -m unirl.utils.prepare_sft_vlm --out-dir data/sft_vlm --max-samples 4000
  # -> data/sft_vlm/{train.jsonl, val.jsonl} + data/sft_vlm/images/

Set HF_ENDPOINT for a mirror.
"""

from __future__ import annotations

import argparse
import json
import os
import random


def _content_text(content) -> str:
    """Flatten a messages content field (str or list of typed parts) to text."""
    if isinstance(content, str):
        return content.strip()
    parts = []
    for part in content or []:
        if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
            parts.append(str(part["text"]))
    return "\n".join(parts).replace("<image>", "").strip()


def _first_round(messages) -> tuple:
    """(user_text, assistant_text) of the first user→assistant exchange."""
    user_text, assistant_text = None, None
    for msg in messages or []:
        role = msg.get("role")
        if role == "user" and user_text is None:
            user_text = _content_text(msg.get("content"))
        elif role == "assistant" and user_text is not None:
            assistant_text = _content_text(msg.get("content"))
            break
    return user_text, assistant_text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="HuggingFaceH4/llava-instruct-mix-vsft")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=4000)
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from datasets import load_dataset

    ds = load_dataset(args.dataset, split=args.split, streaming=True)
    images_dir = os.path.join(args.out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    rows = []
    for i, row in enumerate(ds):
        prompt, response = _first_round(row.get("messages"))
        images = row.get("images") or ([row["image"]] if row.get("image") is not None else [])
        if not prompt or not response or not images:
            continue
        image_name = f"{len(rows):06d}.png"
        images[0].convert("RGB").save(os.path.join(images_dir, image_name))
        rows.append(
            {
                "sample_id": f"vlm_sft:{i}",
                "prompt": prompt,
                "response": response,
                "media": [{"modality": "image", "role": "condition", "uri": f"images/{image_name}"}],
            }
        )
        if len(rows) >= args.max_samples:
            break
    if len(rows) < 2:
        raise SystemExit(f"prepare_sft_vlm: only {len(rows)} usable rows — check the dataset schema.")

    random.Random(args.seed).shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_fraction))
    for name, split_rows in (("train.jsonl", rows[n_val:]), ("val.jsonl", rows[:n_val])):
        path = os.path.join(args.out_dir, name)
        with open(path, "w") as fh:
            for r in split_rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(split_rows):6d} rows -> {path}")


if __name__ == "__main__":
    main()
