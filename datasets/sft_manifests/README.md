# Generic SFT manifest builders (sft_manifests)

Four converters that turn an arbitrary HuggingFace dataset into the local supervised manifest
layout [`unirl/data/sft.py`](../../unirl/data/sft.py) reads — the SFT recipes take local jsonl,
not HF dataset ids.

Unlike the other `datasets/` folders these are **not tied to one dataset**: each ships a
sensible default source and is parameterized by `--dataset` (plus column-key flags), so the
same script cooks any dataset with a compatible schema. That is why they live in one folder
rather than under a dataset name.

| Script | Modality | Default source | Recipes |
|---|---|---|---|
| [`prepare_sft_text.py`](prepare_sft_text.py) | text → text | [`yahma/alpaca-cleaned`](https://huggingface.co/datasets/yahma/alpaca-cleaned) | [`sft/qwen3_sft`](../../examples/sft/qwen3_sft.yaml) |
| [`prepare_sft_vlm.py`](prepare_sft_vlm.py) | image + text → text | [`HuggingFaceH4/llava-instruct-mix-vsft`](https://huggingface.co/datasets/HuggingFaceH4/llava-instruct-mix-vsft) | [`sft/qwen_vl_sft`](../../examples/sft/qwen_vl_sft.yaml) |
| [`prepare_sft_t2i.py`](prepare_sft_t2i.py) | text → image | [`lambdalabs/naruto-blip-captions`](https://huggingface.co/datasets/lambdalabs/naruto-blip-captions) | [`sft/sd3_sft_lora`](../../examples/sft/sd3_sft_lora.yaml), [`sft/bagel_sft_lora`](../../examples/sft/bagel_sft_lora.yaml) |
| [`prepare_sft_agent.py`](prepare_sft_agent.py) | agent trajectories | [`pyromind/agentic-tool-call-dataset-12k`](https://huggingface.co/datasets/pyromind/agentic-tool-call-dataset-12k) | [`sft/qwen3_agent_sft_lora`](../../examples/sft/qwen3_agent_sft_lora.yaml) |

Each script's `--help` carries its record schema and flag list. Generated manifests and
extracted images are local artifacts and must not be committed.

Sources are downloaded automatically into your HF cache; set `HF_ENDPOINT` for a mirror.

## Cook

Needs the converter dependencies (`pip install -e '.[dataset-prep]'`, see
[`../README.md`](../README.md#install)) — all four read the Hub through `datasets`, and
`prepare_sft_agent.py` additionally imports `unirl`.

```bash
python datasets/sft_manifests/prepare_sft_text.py --out-dir data/sft_alpaca
python datasets/sft_manifests/prepare_sft_vlm.py  --out-dir data/sft_vlm --max-samples 4000
python datasets/sft_manifests/prepare_sft_t2i.py  --out-dir data/sft_t2i
python datasets/sft_manifests/prepare_sft_agent.py --out-dir data/sft_agent_toolcall_12k
```

All four write `train.jsonl` + `val.jsonl` under `--out-dir` (`--val-fraction` / `--seed`
control the split). The VLM and T2I builders also write an `images/` subdir next to the
jsonl; image `uri`s are relative to the manifest.

`prepare_sft_agent.py` expands each trajectory into one example per assistant turn, so tool
calls and post-tool final answers are both supervised, and splits at trajectory level to keep
turns from one conversation out of both sides. It validates every row through
`unirl.data.sft.normalize_supervised_example` at prep time — install the package
(`pip install -e .`) before cooking.

## Train

```bash
SFT_DATA=data/sft_alpaca/train.jsonl SFT_EVAL_DATA=data/sft_alpaca/val.jsonl \
python -m unirl.train_sft --config-name=sft/qwen3_sft
```

Point `SFT_DATA` / `SFT_EVAL_DATA` at the matching manifest for the other three.
