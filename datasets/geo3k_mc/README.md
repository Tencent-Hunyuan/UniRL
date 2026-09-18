# Geometry3K multiple-choice (geo3k_mc)

Diagram + 4-way multiple-choice geometry QA, converted into the local jsonl + `images/`
layout the multimodal RL data loader reads. The reward is `mc_exact_match` on the gold
letter.

- Recipes: [`examples/ar/qwen_vl_grpo_geo3k_mc_4x8.yaml`](../../examples/ar/qwen_vl_grpo_geo3k_mc_4x8.yaml)
  and its `_lora` / `sglang_` variants,
  [`examples/ar/qwen3_5_grpo_9b_geo3k_mc_sglang.yaml`](../../examples/ar/qwen3_5_grpo_9b_geo3k_mc_sglang.yaml)
- Converter: [`prepare_geo3k_mc.py`](prepare_geo3k_mc.py) — `--help` carries the full record contract
- Loader contract: [`unirl/data/data_source.py`](../../unirl/data/data_source.py) + [`unirl/data/datasets.py`](../../unirl/data/datasets.py)

Generated jsonl and images are local artifacts and must not be committed.

## Source

[`xyliu6/geometry3k`](https://huggingface.co/datasets/xyliu6/geometry3k) — Geometry3K in its
native 4-way multiple-choice form (diagram image + `problem` + `choices` + `ground_truth`
letter). Downloaded automatically through `datasets.load_dataset` into your HF cache; set
`HF_ENDPOINT` for a mirror.

## Cook

Needs the converter dependencies (`pip install -e '.[dataset-prep]'`, see
[`../README.md`](../README.md#install)).

```bash
python datasets/geo3k_mc/prepare_geo3k_mc.py --out-dir data/geo3k_mc
```

Writes `train.jsonl`, `val.jsonl`, `test.jsonl` and an `images/` subdir under `--out-dir`
(image `uri`s are relative to the jsonl). Override `--hf` / `--train-split` / `--val-split`
/ `--test-split` if your source differs; `--test-split ''` skips the test split.

## Train

The model defaults to the HF id `Qwen/Qwen2.5-VL-7B-Instruct`; set `QWEN_VL_PATH` to a local
dir to use a cache. `DATA_PATH` is required.

```bash
DATA_PATH=data/geo3k_mc/train.jsonl EVAL_DATA_PATH=data/geo3k_mc/val.jsonl \
python -m unirl.train_ar --config-name=ar/qwen_vl_grpo_geo3k_mc_4x8
```
