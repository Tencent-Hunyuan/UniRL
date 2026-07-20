"""Fetch GPQA-Diamond (gated; NEVER commit the data — it carries a canary string).

Prerequisites: accept the conditions at https://huggingface.co/datasets/Idavidrein/gpqa
while logged in, then run with a token: ``HF_TOKEN=... python benchmarks/text/gpqa/fetch.py``.

Writes ``data/gpqa_diamond.jsonl`` with ``{id, problem, answer}``: the problem embeds
the four options (A-D) shuffled with a fixed per-question seed, mirroring the official
baseline's seeded shuffle (github.com/idavidrein/gpqa baselines/utils.py).
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
LETTERS = "ABCD"


def main() -> None:
    from huggingface_hub import hf_hub_download  # ships with the repo's transformers/diffusers stack

    csv_path = hf_hub_download(repo_id="Idavidrein/gpqa", filename="gpqa_diamond.csv", repo_type="dataset")
    DATA_DIR.mkdir(exist_ok=True)
    out_path = DATA_DIR / "gpqa_diamond.jsonl"
    with open(csv_path, newline="") as f, open(out_path, "w") as out:
        rows = list(csv.DictReader(f))
        for i, row in enumerate(rows):
            options = [
                row["Correct Answer"].strip(),
                row["Incorrect Answer 1"].strip(),
                row["Incorrect Answer 2"].strip(),
                row["Incorrect Answer 3"].strip(),
            ]
            random.Random(i).shuffle(options)
            answer = LETTERS[options.index(row["Correct Answer"].strip())]
            problem = (
                row["Question"].strip()
                + "\n\n"
                + "\n".join(f"({letter}) {opt}" for letter, opt in zip(LETTERS, options))
            )
            out.write(json.dumps({"id": f"diamond-{i}", "problem": problem, "answer": answer}) + "\n")
    print(f"wrote {len(rows)} questions to {out_path} (do NOT commit this file)")


if __name__ == "__main__":
    main()
