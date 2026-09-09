"""Prepare a local OpenAI-style agent SFT manifest."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from typing import Any, Dict, Iterable, List, Sequence

from unirl.data.sft import normalize_supervised_example
from unirl.models.types.conversations import tokenize_agent_target

_HELP = """Prepare a local OpenAI-style agent SFT manifest.

The default source is the Apache-2.0
``pyromind/agentic-tool-call-dataset-12k`` short split. Each trajectory is
expanded into one example per assistant turn so tool calls and post-tool final
answers are both supervised. The split is trajectory-level to keep turns from
the same conversation out of both train and validation.

Usage:
  python datasets/sft_manifests/prepare_sft_agent.py \
    --out-dir data/sft_agent_toolcall_12k
"""


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path) as fh:
        for line_num, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_num}: invalid JSON — {exc}") from exc
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_num}: expected an object, got {type(row).__name__}.")
            rows.append(row)
    return rows


def expand_trajectory(row: Dict[str, Any], *, trajectory_id: str) -> List[Dict[str, Any]]:
    """Expand one trajectory into final-assistant-target training records."""
    messages = row.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"{trajectory_id}: missing OpenAI-style 'messages' list.")
    tools = row.get("tools")
    examples: List[Dict[str, Any]] = []
    seen_user = False
    for turn, message in enumerate(messages):
        if isinstance(message, dict) and message.get("role") == "user":
            seen_user = True
        if not isinstance(message, dict) or message.get("role") != "assistant" or not seen_user:
            continue
        example: Dict[str, Any] = {
            "sample_id": f"{trajectory_id}:assistant-{turn}",
            "messages": messages[: turn + 1],
            "metadata": {"source_trajectory": trajectory_id, "assistant_turn": turn},
        }
        if tools is not None:
            example["tools"] = tools
        normalize_supervised_example(example, default_sample_id=example["sample_id"])
        examples.append(example)
    if not examples:
        raise ValueError(f"{trajectory_id}: no assistant target turn after a user message.")
    return examples


def _expand(rows: Sequence[Dict[str, Any]], *, dataset_name: str) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    prefix = dataset_name.rsplit("/", 1)[-1]
    for index, row in enumerate(rows):
        examples.extend(expand_trajectory(row, trajectory_id=f"{prefix}:{index}"))
    return examples


def _write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> int:
    count = 0
    with open(path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _filter_overlong_targets(
    examples: Sequence[Dict[str, Any]],
    *,
    tokenizer: Any,
    max_response_length: int,
    enable_thinking: bool,
) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Drop targets that the matching AR track builder would have to truncate."""
    eos_id = tokenizer.eos_token_id
    if isinstance(eos_id, (list, tuple)):
        eos_id = eos_id[0] if eos_id else None
    if eos_id is None:
        raise ValueError("The selected tokenizer has no eos_token_id, but the recipe appends EOS.")

    kept: List[Dict[str, Any]] = []
    dropped = {"tool_call": 0, "final_answer": 0}
    for example in examples:
        target_ids = tokenize_agent_target(
            example,
            tokenizer=tokenizer,
            enable_thinking=enable_thinking,
        )
        if not target_ids:
            raise ValueError(f"{example['sample_id']}: final assistant target tokenized to zero tokens.")
        target_length = len(target_ids) + (target_ids[-1] != eos_id)
        if target_length <= max_response_length:
            kept.append(example)
            continue
        target = example["messages"][-1]
        kind = "tool_call" if target.get("tool_calls") else "final_answer"
        dropped[kind] += 1
    return kept, dropped


def main() -> None:
    parser = argparse.ArgumentParser(description=_HELP)
    parser.add_argument("--dataset", default="pyromind/agentic-tool-call-dataset-12k")
    parser.add_argument("--filename", default="agent_short_10k.jsonl")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-trajectories", type=int, default=10000)
    parser.add_argument("--tokenizer", default=os.environ.get("QWEN3_PATH", "Qwen/Qwen3-8B"))
    parser.add_argument("--max-response-length", type=int, default=1024)
    parser.add_argument(
        "--disable-thinking",
        action="store_true",
        help="Render targets with enable_thinking=False; use only with a matching pipeline setting.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 0.0 < args.val_fraction < 1.0:
        raise SystemExit("--val-fraction must be between 0 and 1.")
    if args.max_response_length < 1:
        raise SystemExit("--max-response-length must be at least 1.")

    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    source_path = hf_hub_download(repo_id=args.dataset, filename=args.filename, repo_type="dataset")
    rows = _read_jsonl(source_path)
    random.Random(args.seed).shuffle(rows)
    if args.max_trajectories > 0:
        rows = rows[: args.max_trajectories]
    if len(rows) < 2:
        raise SystemExit(f"prepare_sft_agent: only {len(rows)} trajectories found.")

    n_val = min(len(rows) - 1, max(1, round(len(rows) * args.val_fraction)))
    val_rows, train_rows = rows[:n_val], rows[n_val:]
    train_examples = _expand(train_rows, dataset_name=args.dataset)
    val_examples = _expand(val_rows, dataset_name=args.dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    filter_kwargs = {
        "tokenizer": tokenizer,
        "max_response_length": args.max_response_length,
        "enable_thinking": not args.disable_thinking,
    }
    train_examples, train_dropped = _filter_overlong_targets(train_examples, **filter_kwargs)
    val_examples, val_dropped = _filter_overlong_targets(val_examples, **filter_kwargs)
    if not train_examples or not val_examples:
        raise SystemExit(
            "prepare_sft_agent: length filtering left an empty "
            f"{'train' if not train_examples else 'validation'} split."
        )

    os.makedirs(args.out_dir, exist_ok=True)
    raw_path = os.path.join(args.out_dir, args.filename)
    if os.path.abspath(source_path) != os.path.abspath(raw_path):
        shutil.copyfile(source_path, raw_path)
    train_path = os.path.join(args.out_dir, "train.jsonl")
    val_path = os.path.join(args.out_dir, "val.jsonl")
    n_train = _write_jsonl(train_path, train_examples)
    n_val_examples = _write_jsonl(val_path, val_examples)
    print(f"source: {len(rows)} trajectories ({len(train_rows)} train / {len(val_rows)} val)")
    print(
        f"filtered targets over {args.max_response_length} tokens: "
        f"{train_dropped['tool_call'] + val_dropped['tool_call']} tool calls, "
        f"{train_dropped['final_answer'] + val_dropped['final_answer']} final answers"
    )
    print(f"wrote {n_train:6d} assistant targets -> {train_path}")
    print(f"wrote {n_val_examples:6d} assistant targets -> {val_path}")
    print(f"kept raw source -> {raw_path}")


if __name__ == "__main__":
    main()
