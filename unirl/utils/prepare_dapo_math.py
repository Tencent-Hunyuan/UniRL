"""Build the local jsonl that ``qwen3_drpo_4b_base_dapo_sglang`` trains on.

The AR data loader (``unirl/data/data_source.py``) reads a **local** jsonl of
``{"prompt": <str>, "metadata": {"answer": <str>}}`` records — it does not accept
HuggingFace dataset ids. This tool converts the raw competition-math datasets into
that format so the recipe is reproducible without any private cache:

  - train : DAPO-Math-17k (rule-verifiable competition math), the reference
            ``data/dapo_filter.parquet`` source.
  - eval  : AIME 2024 + AIME 2025 (the paper's avg@16 benchmark), concatenated.

Usage:
  python scripts/prepare_dapo_math.py --out-dir data/dapo_math
  # → data/dapo_math/train.jsonl  (+ aime_eval.jsonl)

  DATA_PATH=data/dapo_math/train.jsonl EVAL_DATA_PATH=data/dapo_math/aime_eval.jsonl \
  python -m unirl.train_ar --config-name=ar/qwen3_drpo_4b_base_dapo_sglang num_devices=64

The HF ids below are sensible defaults; override with the flags if your source
differs. The extractor handles the common verl RL schema (``prompt`` chat list +
``reward_model.ground_truth``) with fallbacks for plain problem/answer columns.
"""

from __future__ import annotations

import argparse
import json
import os

# Boxed-answer instruction (matches the math_boxed reward, which extracts \boxed{}).
BOXED_SUFFIX = "\n\nLet's think step by step and output the final answer within \\boxed{}."


def _extract_prompt(row: dict, append_boxed: bool) -> str:
    """Pull the user-facing prompt text from a raw row (schema-tolerant)."""
    p = row.get("prompt")
    if isinstance(p, list) and p:  # verl chat format: [{"role","content"}, ...]
        text = (p[-1] or {}).get("content", "") if isinstance(p[-1], dict) else str(p[-1])
    elif isinstance(p, str):
        text = p
    else:  # plain math datasets (AIME_2024 capitalizes its columns)
        text = row.get("question") or row.get("problem") or row.get("Problem") or ""
    text = (text or "").strip()
    if append_boxed and "\\boxed" not in text:
        text = text + BOXED_SUFFIX
    return text


def _extract_answer(row: dict) -> str | None:
    """Pull the ground-truth answer (schema-tolerant)."""
    rm = row.get("reward_model")
    if isinstance(rm, dict) and rm.get("ground_truth") is not None:
        return str(rm["ground_truth"]).strip()
    ei = row.get("extra_info")
    if isinstance(ei, dict) and ei.get("answer") is not None:
        return str(ei["answer"]).strip()
    for k in ("answer", "Answer", "solution", "final_answer", "gt"):
        if row.get(k) is not None:
            return str(row[k]).strip()
    return None


def _convert(hf_id: str, split: str, out_path: str, append_boxed: bool, retry_columns: list[str] | None = None) -> int:
    from datasets import load_dataset

    try:
        ds = load_dataset(hf_id, split=split)
    except Exception as e:
        # Shard-inconsistent side columns (e.g. DPAO_filter's extra_info is
        # {index: string} in one shard and struct<index,split> in another) make
        # schema unification fail. Retry reading only the columns we use — the
        # column set is source-specific (verl-style sources carry
        # prompt/reward_model; the AIME sets don't), so callers opt in.
        if retry_columns is None:
            raise
        print(f"  full-schema load failed ({type(e).__name__}); retrying with pruned columns")
        ds = load_dataset(hf_id, split=split, columns=retry_columns)
    n = 0
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        for row in ds:
            prompt = _extract_prompt(row, append_boxed)
            answer = _extract_answer(row)
            if not prompt or answer is None:
                continue  # skip rows we cannot rule-verify
            f.write(json.dumps({"prompt": prompt, "metadata": {"answer": answer}}, ensure_ascii=False) + "\n")
            n += 1
    print(f"  wrote {n} records -> {out_path}  (from {hf_id}:{split})")
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="data/dapo_math", help="output directory for the jsonl files")
    ap.add_argument("--dapo-hf", default="BytedTsinghua-SIA/DAPO-Math-17k", help="HF id for the train set")
    ap.add_argument("--dapo-split", default="train")
    ap.add_argument("--aime24-hf", default="Maxwell-Jia/AIME_2024", help="HF id for AIME 2024 (eval)")
    ap.add_argument("--aime25-hf", default="yentinglin/aime_2025", help="HF id for AIME 2025 (eval)")
    ap.add_argument("--aime-split", default="train")
    ap.add_argument(
        "--append-boxed-template",
        action="store_true",
        help="append the \\boxed{} instruction to prompts that lack it (set if the raw "
        "prompts don't already request a boxed answer)",
    )
    args = ap.parse_args()

    try:
        import datasets  # noqa: F401
    except ImportError:
        raise SystemExit("This tool needs `datasets`: pip install datasets")

    print("Building train set:")
    _convert(
        args.dapo_hf,
        args.dapo_split,
        os.path.join(args.out_dir, "train.jsonl"),
        args.append_boxed_template,
        retry_columns=["prompt", "reward_model"],  # verl schema; pruning drops the conflicting extra_info
    )

    print("Building AIME eval set (2024 + 2025):")
    eval_path = os.path.join(args.out_dir, "aime_eval.jsonl")
    total = 0
    os.makedirs(args.out_dir, exist_ok=True)
    with open(eval_path, "w") as out:
        for hf_id in (args.aime24_hf, args.aime25_hf):
            tmp = eval_path + ".part"
            total += _convert(hf_id, args.aime_split, tmp, args.append_boxed_template)
            with open(tmp) as part:
                out.write(part.read())
            os.remove(tmp)
    print(f"  combined AIME eval: {total} problems -> {eval_path}")


if __name__ == "__main__":
    main()
