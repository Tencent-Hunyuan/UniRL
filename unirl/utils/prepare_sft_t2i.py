"""Build a local T2I SFT manifest (caption + target image) from an HF dataset.

Converts an image-caption dataset into the supervised manifest layout
(``unirl/data/sft.py``) the diffusion SFT recipes read::

    {"prompt": <caption>,
     "media": [{"modality": "image", "role": "target", "uri": "images/<id>.png"}]}

plus an ``images/`` subdir next to the jsonl. Default source is
``lambdalabs/naruto-blip-captions`` (~1.2k pairs — the classic small
finetune set); any dataset with ``image`` + ``text``/``caption`` columns works.

Usage:
  python -m unirl.utils.prepare_sft_t2i --out-dir data/sft_t2i
  # -> data/sft_t2i/{train.jsonl, val.jsonl} + data/sft_t2i/images/

Set HF_ENDPOINT for a mirror.
"""

from __future__ import annotations

import argparse
import json
import os
import random


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="lambdalabs/naruto-blip-captions")
    parser.add_argument("--split", default="train")
    parser.add_argument("--caption-key", default="text")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=4000)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from datasets import load_dataset

    ds = load_dataset(args.dataset, split=args.split)
    images_dir = os.path.join(args.out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    rows = []
    for i, row in enumerate(ds):
        caption = str(row.get(args.caption_key) or row.get("caption") or "").strip()
        image = row.get("image")
        if not caption or image is None:
            continue
        image_name = f"{len(rows):06d}.png"
        image.convert("RGB").save(os.path.join(images_dir, image_name))
        rows.append(
            {
                "sample_id": f"t2i_sft:{i}",
                "prompt": caption,
                "media": [{"modality": "image", "role": "target", "uri": f"images/{image_name}"}],
            }
        )
        if len(rows) >= args.max_samples:
            break
    if len(rows) < 2:
        raise SystemExit(f"prepare_sft_t2i: only {len(rows)} usable rows — check --caption-key.")

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
