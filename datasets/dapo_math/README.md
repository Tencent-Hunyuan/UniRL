# DAPO-Math + AIME (dapo_math)

Rule-verifiable competition math, converted into the local
`{"prompt", "metadata": {"answer"}}` jsonl the AR data loader reads — it does not accept
HuggingFace dataset ids.

- Recipes: every `examples/ar/qwen3_*_dapo_sglang.yaml` (GRPO / DRPO / DrGRPO / DPPO / PPO /
  CPPO / vanilla), plus [`CPPO/`](../../CPPO) and [`DRPO/`](../../DRPO)
- Converter: [`prepare_dapo_math.py`](prepare_dapo_math.py) — `--help` carries the full contract
- Loader contract: [`unirl/data/data_source.py`](../../unirl/data/data_source.py)

Generated jsonl is a local artifact and must not be committed.

## Source

| Split | Dataset |
|---|---|
| train | [`BytedTsinghua-SIA/DAPO-Math-17k`](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k) |
| eval | [`Maxwell-Jia/AIME_2024`](https://huggingface.co/datasets/Maxwell-Jia/AIME_2024) + [`yentinglin/aime_2025`](https://huggingface.co/datasets/yentinglin/aime_2025), concatenated |

Downloaded automatically through `datasets.load_dataset` into your HF cache; set `HF_ENDPOINT`
for a mirror. The extractor handles the common verl RL schema (`prompt` chat list +
`reward_model.ground_truth`) with fallbacks for plain problem/answer columns, so
`--dapo-hf` / `--aime24-hf` / `--aime25-hf` can point at an equivalent source.

## Cook

Needs the converter dependencies (`pip install -e '.[dataset-prep]'`, see
[`../README.md`](../README.md#install)).

```bash
python datasets/dapo_math/prepare_dapo_math.py --out-dir data/dapo_math
```

Writes `train.jsonl` and `aime_eval.jsonl` under `--out-dir`.

## Train

The model defaults to the HF id `Qwen/Qwen3-4B-Base`; set `QWEN3_PATH` to a local checkpoint
dir to use a cache. `DATA_PATH` is required.

```bash
DATA_PATH=data/dapo_math/train.jsonl EVAL_DATA_PATH=data/dapo_math/aime_eval.jsonl \
python -m unirl.train_ar --config-name=ar/qwen3_drpo_4b_base_dapo_sglang num_devices=32
```
