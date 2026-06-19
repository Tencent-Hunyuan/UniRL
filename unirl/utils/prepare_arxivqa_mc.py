"""Build the local arxivqa_mc dataset that the ``bagel_grpo_arxivqa_mc_*`` recipe trains on.

Same on-disk layout as :mod:`unirl.utils.prepare_geo3k_mc` (so the multimodal RL data
loader ``unirl/data/data_source.py`` + ``unirl/data/datasets.py`` consumes it unchanged):
a **local** jsonl of records::

    {"prompt": <str — the question with its A/B/C/D options inline>,
     "prompt_id": "<config>:<split>:<n>",
     "media_refs": [{"modality": "image", "role": "condition", "uri": "images/<...>.png"}],
     "metadata": {"answer": "<A|B|C|D>", "data_source": <str>}}

plus an ``images/`` subdir next to the jsonl — relative ``uri``\\ s resolve against the
jsonl's directory (``datasets._resolve_media_uri``). The reward (``mc_exact_match``) only
needs ``metadata.answer`` — the single ground-truth letter.

  - source : ``zlab-princeton/Vero-600k`` config ``chart_ocr-arxivqa_formatted`` — arxivqa
             scientific-figure QA in Vero's verl-style schema (``reward_model.ground_truth``
             + ``extra_info.reward_type``/``question``). Single-domain (image+text->text)
             chart/figure reasoning.
  - splits : ``train`` -> train.jsonl, ``val`` -> val.jsonl (Vero's own held-out val, no
             leakage). The recipe reads ``DATA_PATH=.../train.jsonl`` and
             ``EVAL_DATA_PATH=.../val.jsonl``.
  - keep   : ONLY ``reward_type == "multiple_choice"`` rows whose gold is a single A-D
             letter (``mc_exact_match`` supports A-D; the rare >D-choice rows are dropped).
             No subsampling — every qualifying row is kept (~12k train).

The rows are read straight from the config's parquet shards with pyarrow
(``row = {k: column[i].as_py()}``), i.e. the exact extraction the dataset was built
with — no ``datasets`` row-structure reshaping of the nested ``prompt`` / ``reward_model``
/ ``extra_info`` structs. The arxivqa ``question`` already contains the
``A) .. B) .. C) .. D) ..`` options inline, so we only wrap it with the look-at-image
preamble + the answer-with-a-letter instruction. Images are downscaled to ``--max-edge``
(BAGEL's ViT ceiling, 980) preserving aspect ratio.

Usage:
  python -m unirl.utils.prepare_arxivqa_mc --out-dir data/arxivqa_mc
  # -> data/arxivqa_mc/{train.jsonl, val.jsonl} + data/arxivqa_mc/images/

  BAGEL_PATH=/root/BAGEL-7B-MoT \\
  DATA_PATH=data/arxivqa_mc/train.jsonl EVAL_DATA_PATH=data/arxivqa_mc/val.jsonl \\
  ENTRY=train_ar bash examples/run_experiment_multinode_taiji.sh ar/bagel_grpo_arxivqa_mc_2x8_lora

The repo / config / split names below are sensible defaults; override with the flags if
your source differs. Needs ``pyarrow`` + ``huggingface_hub`` + ``pillow``:
pip install pyarrow huggingface_hub pillow
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re

LETTERS = {"A", "B", "C", "D"}
IMAGE_TOKEN = re.compile(r"^\s*<image>\s*\n?")


def to_pil(cell):
    """A PIL image from a Vero ``image`` cell.

    Parquet stores the HF Image feature as ``{bytes, path}`` (what ``.as_py()`` yields);
    also tolerant of raw ``bytes`` or a 1-element list, for robustness.
    """
    from PIL import Image

    if cell is None:
        return None
    if isinstance(cell, list) and cell:
        return to_pil(cell[0])
    if isinstance(cell, dict):
        if cell.get("bytes"):
            return Image.open(io.BytesIO(cell["bytes"]))
        if cell.get("path"):
            return Image.open(cell["path"])
        return None
    if isinstance(cell, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(cell)))
    return None


def question_text(prompt_cell, extra) -> str:
    """Question text, schema-tolerant: ``prompt.content`` (list/str) else ``extra_info.question``."""
    if isinstance(prompt_cell, dict):
        content = prompt_cell.get("content")
        if isinstance(content, list) and content:
            return str(content[-1])
        if isinstance(content, str):
            return content
    return str((extra or {}).get("question") or "")


def gold_letter(row: dict) -> str:
    """The single A-D ground-truth letter (``reward_model.ground_truth`` else ``extra_info.answer``)."""
    rm = row.get("reward_model") or {}
    extra = row.get("extra_info") or {}
    return str(rm.get("ground_truth", extra.get("answer", ""))).strip().upper()


def is_multiple_choice(row: dict) -> bool:
    rm = row.get("reward_model") or {}
    extra = row.get("extra_info") or {}
    return (extra.get("reward_type") or rm.get("style")) == "multiple_choice"


def shard_paths(repo: str, config: str, split: str) -> list:
    """Download (to the HF cache) + return the config's ``{split}-*.parquet`` shard paths."""
    from huggingface_hub import HfApi, hf_hub_download

    files = HfApi().list_repo_files(repo, repo_type="dataset")
    want = sorted(
        f
        for f in files
        if f.startswith(f"{config}/") and os.path.basename(f).startswith(f"{split}-") and f.endswith(".parquet")
    )
    if not want:
        raise SystemExit(f"prepare_arxivqa_mc: no {split}-*.parquet found under {repo}/{config}")
    return [hf_hub_download(repo, f, repo_type="dataset") for f in want]


def iter_rows(paths: list):
    """Yield each parquet row as a plain dict (``column[i].as_py()`` — the build's exact read)."""
    import pyarrow.parquet as pq

    for path in paths:
        table = pq.read_table(path)
        cols = table.column_names
        for i in range(table.num_rows):
            yield {k: table.column(k)[i].as_py() for k in cols}


def convert_split(rows, split: str, out_dir: str, fname: str, max_edge: int, config: str) -> int:
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None  # Vero has legit multi-thousand-px figures.
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    out_path = os.path.join(out_dir, fname)
    n = 0
    with open(out_path, "w") as f:
        for row in rows:
            if not is_multiple_choice(row):
                continue
            gold = gold_letter(row)
            if gold not in LETTERS:
                continue  # mc_exact_match supports A-D only
            question = IMAGE_TOKEN.sub("", question_text(row.get("prompt"), row.get("extra_info"))).strip()
            pil = to_pil(row.get("image"))
            if not question or pil is None:
                continue  # skip rows we cannot turn into a verifiable MC example
            rel = f"images/{config}_{split}_{n}.png"
            pil = pil.convert("RGB")
            pil.thumbnail((max_edge, max_edge))  # downscale-only, preserves aspect
            pil.save(os.path.join(out_dir, rel))
            record = {
                "prompt": f"Look at the image. {question}\n\nAnswer with the letter only.",
                "prompt_id": f"{config}:{split}:{n}",
                "media_refs": [{"modality": "image", "role": "condition", "uri": rel}],
                "metadata": {"answer": gold, "data_source": row.get("data_source")},
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
    print(f"  wrote {n} records -> {out_path}  (+ images in {images_dir}/)")
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="data/arxivqa_mc", help="output dir for the jsonl files + images/")
    ap.add_argument("--repo", default="zlab-princeton/Vero-600k", help="HF dataset id")
    ap.add_argument("--config", default="chart_ocr-arxivqa_formatted", help="Vero config (subdirectory) to convert")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--val-split", default="val")
    ap.add_argument("--max-edge", type=int, default=980, help="downscale long edge to this (BAGEL ViT ceiling)")
    args = ap.parse_args()

    try:
        import huggingface_hub  # noqa: F401
        import PIL  # noqa: F401
        import pyarrow.parquet  # noqa: F401
    except ImportError:
        raise SystemExit(
            "This tool needs pyarrow + huggingface_hub + pillow: pip install pyarrow huggingface_hub pillow"
        )

    for split, fname in ((args.train_split, "train.jsonl"), (args.val_split, "val.jsonl")):
        print(f"Building {fname} (split={split}):")
        paths = shard_paths(args.repo, args.config, split)
        convert_split(iter_rows(paths), split, args.out_dir, fname, args.max_edge, args.config)


if __name__ == "__main__":
    main()
