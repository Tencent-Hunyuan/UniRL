"""Prepare ASearcher deep-research training data as jsonl (LIN-519).

Writes one jsonl line per example in the schema ``MultimodalRLDataSource`` expects —
``{"prompt": <question>, "metadata": {"answer": <reference>}}`` (the data source reads
the per-row ``metadata`` dict verbatim; a top-level ``answer`` is NOT lifted). The
trainer then reads ``metadata["answer"]`` as the gold for the LLM-judge reward. Sibling
of ``prepare_dapo_math.py``.

  # download from HF (inclusionAI/ASearcher-train-data) and convert:
  python -m unirl.utils.prepare_asearcher --out-dir data/asearcher
  # or convert a local jsonl (tolerant of field names):
  python -m unirl.utils.prepare_asearcher --source path/to/asearcher.jsonl --out-dir data/asearcher
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterator, Optional, Sequence

_QUESTION_KEYS = ("question", "prompt", "query")
_ANSWER_KEYS = ("answer", "gt", "golden_answers", "ground_truth", "label")


def _first(row: Dict[str, Any], keys: Sequence[str]) -> Optional[Any]:
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return None


def _norm_answer(a: Any) -> Optional[str]:
    if isinstance(a, (list, tuple)):
        a = a[0] if a else None
    return None if a is None else str(a).strip()


def _iter_source(source: Optional[str], hf_name: str, split: str) -> Iterator[Dict[str, Any]]:
    if source and os.path.exists(source):
        with open(source, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return
    from datasets import load_dataset  # lazy: only needed when downloading

    # streaming=True yields raw rows and skips the arrow schema cast that fails when
    # building the full DatasetDict (the sibling ASearcherLRM35k split has an
    # inconsistent schema), and avoids downloading the whole set up front.
    for row in load_dataset(hf_name, split=split, streaming=True):
        yield dict(row)


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare ASearcher {prompt, answer} jsonl.")
    ap.add_argument("--out-dir", default="data/asearcher")
    ap.add_argument("--source", default=None, help="local jsonl (else download from HF)")
    ap.add_argument("--hf-name", default="inclusionAI/ASearcher-train-data")
    # ASearcherBase35k loads cleanly; ASearcherLRM35k (AReaL's exact split) has an
    # inconsistent HF schema (extra id/idx cols, scalar answer) that `datasets`
    # can't auto-cast — download its raw jsonl and pass it via --source to use it.
    ap.add_argument("--split", default="ASearcherBase35k")
    ap.add_argument("--limit", type=int, default=0, help="cap kept rows (0 = all)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "train.jsonl")
    seen = kept = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for row in _iter_source(args.source, args.hf_name, args.split):
            seen += 1
            question = _first(row, _QUESTION_KEYS)
            answer = _norm_answer(_first(row, _ANSWER_KEYS))
            if not question or not answer:
                continue
            out.write(
                json.dumps(
                    {"prompt": str(question).strip(), "metadata": {"answer": answer}},
                    ensure_ascii=False,
                )
                + "\n"
            )
            kept += 1
            if args.limit and kept >= args.limit:
                break
    print(f"wrote {kept}/{seen} examples -> {out_path}")


if __name__ == "__main__":
    main()
