# ArxivQA multiple-choice (arxivqa_mc)

Scientific-figure QA in 4-way multiple-choice form, converted into the same local jsonl +
`images/` layout as [`../geo3k_mc`](../geo3k_mc). The reward is `mc_exact_match` on the gold
letter.

- Recipe: [`examples/ar/bagel_grpo_arxivqa_mc_2x8_lora.yaml`](../../examples/ar/bagel_grpo_arxivqa_mc_2x8_lora.yaml)
- Converter: [`prepare_arxivqa_mc.py`](prepare_arxivqa_mc.py) — `--help` carries the full record contract
- Loader contract: [`unirl/data/data_source.py`](../../unirl/data/data_source.py) + [`unirl/data/datasets.py`](../../unirl/data/datasets.py)

Generated jsonl and images are local artifacts and must not be committed.

## Source

[`zlab-princeton/Vero-600k`](https://huggingface.co/datasets/zlab-princeton/Vero-600k),
config `chart_ocr-arxivqa_formatted` — arxivqa figure QA in Vero's verl-style schema. Rows are
read straight from the config's parquet shards with `pyarrow` (no `datasets` row reshaping of
the nested structs), pulled through `huggingface_hub`.

Only `reward_type == "multiple_choice"` rows whose gold is a single A–D letter are kept
(~12k train, no subsampling). Vero's own held-out `val` split is used, so there is no leakage.

## Cook

Needs the converter dependencies (`pip install -e '.[dataset-prep]'`, see
[`../README.md`](../README.md#install)) — this one reads the parquet shards through `pyarrow`
rather than `datasets`.

```bash
python datasets/arxivqa_mc/prepare_arxivqa_mc.py --out-dir data/arxivqa_mc
```

Writes `train.jsonl`, `val.jsonl` and an `images/` subdir under `--out-dir`. Images are
downscaled to `--max-edge` (default 980, BAGEL's ViT ceiling) preserving aspect ratio.

## Train

```bash
BAGEL_PATH=/root/BAGEL-7B-MoT \
DATA_PATH=data/arxivqa_mc/train.jsonl EVAL_DATA_PATH=data/arxivqa_mc/val.jsonl \
ENTRY=train_ar bash examples/run_experiment_multinode.sh ar/bagel_grpo_arxivqa_mc_2x8_lora
```
