"""Build a local LLM SFT manifest (``{"prompt", "response"}`` jsonl) from an HF dataset.

The supervised data layer (``unirl/data/sft.py``) reads local jsonl manifests,
not HF dataset ids. This tool converts an instruction dataset into that layout:

  - default source: ``yahma/alpaca-cleaned`` (instruction / input / output);
    any dataset whose columns map onto (instruction[, input], output) works via
    the ``--prompt-key/--input-key/--response-key`` flags.
  - splits: one source split, sliced into train.jsonl + val.jsonl
    (``--val-fraction``, deterministic tail split after a seeded shuffle).

Usage:
  python -m unirl.utils.prepare_sft_text --out-dir data/sft_alpaca
  # -> data/sft_alpaca/{train.jsonl, val.jsonl}

  SFT_DATA=data/sft_alpaca/train.jsonl SFT_EVAL_DATA=data/sft_alpaca/val.jsonl \
  python -m unirl.train_sft --config-name=sft/qwen3_sft

Set HF_ENDPOINT for a mirror; --max-samples caps rows for smoke runs.
"""

from __future__ import annotations

import argparse
import json
import os
import random


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="yahma/alpaca-cleaned")
    parser.add_argument("--split", default="train")
    parser.add_argument("--prompt-key", default="instruction")
    parser.add_argument("--input-key", default="input")
    parser.add_argument("--response-key", default="output")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=20000)
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from datasets import load_dataset

    ds = load_dataset(args.dataset, split=args.split)
    rows = []
    for i, row in enumerate(ds):
        prompt = str(row.get(args.prompt_key) or "").strip()
        extra = str(row.get(args.input_key) or "").strip() if args.input_key else ""
        response = str(row.get(args.response_key) or "").strip()
        if not prompt or not response:
            continue
        if extra:
            prompt = f"{prompt}\n\n{extra}"
        rows.append({"sample_id": f"{os.path.basename(args.dataset)}:{i}", "prompt": prompt, "response": response})
        if len(rows) >= args.max_samples:
            break
    if len(rows) < 2:
        raise SystemExit(f"prepare_sft_text: only {len(rows)} usable rows — check the column flags.")

    random.Random(args.seed).shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_fraction))
    os.makedirs(args.out_dir, exist_ok=True)
    for name, split_rows in (("train.jsonl", rows[n_val:]), ("val.jsonl", rows[:n_val])):
        path = os.path.join(args.out_dir, name)
        with open(path, "w") as fh:
            for r in split_rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(split_rows):6d} rows -> {path}")


if __name__ == "__main__":
    main()
