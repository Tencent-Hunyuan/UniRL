# Installation

UniRL ships two mutually exclusive inference engines (`vllm` and `sglang`) — install each in its own virtual environment.

| Engine | CUDA | glibc |
|---|---|---|
| **vllm-omni** | 12.9 | ≥ 2.28 |
| **sglang** | 13.0 | ≥ 2.34 |

## vllm-omni

```bash
uv venv --python 3.12 --seed .venv && source .venv/bin/activate
export VLLM_USE_PRECOMPILED=1   # else 30+ min CUDA build
uv pip install -e ".[vllm,train,infer]"
```

## sglang

```bash
uv venv --python 3.12 --seed .venv-sglang && source .venv-sglang/bin/activate
uv pip install -e ".[sglang,train,infer]" --prerelease=allow
```

## Extras

| Extra | Adds | Use when |
|---|---|---|
| `vllm` | `vllm`, `vllm-omni`, torch +cu129 stack | Running any vllm-omni-based example |
| `sglang` | `sglang[diffusion]`, `flash-attn-4`, torch +cu130 stack | Running VLM/LLM examples or `sd3_sglang_*` |
| `train` | `wandb`, `aiohttp` | Training runs (almost always wanted) |
| `infer` | `accelerate` | HunyuanImage3 and similar models |
| `eval` | `torchvision`, `easyocr` | OCR-based reward components |
| `dev` | `pytest`, `ruff`, `pre-commit` | Local development |

For development tools (lint and tests):

```bash
uv pip install -e ".[vllm,train,infer,eval,dev]"
# or, for the sglang engine:
uv pip install -e ".[sglang,train,infer,eval,dev]" --prerelease=allow
```

## Environment

Example configs read cluster-local paths, checkpoints, data, and W&B settings from
environment variables via `${oc.env:...}`. Common variables:

| Variable | Purpose |
|---|---|
| `PRETRAINED_MODEL` | Base model checkpoint path |
| `DATA_PATH` | Training data / prompt-list path |
| `EVAL_DATA_PATH` | Evaluation data path |
| `HF_TOKEN` | Hugging Face token for gated models (e.g. SD3.5) |
| `REPORT_TO_WANDB` | Enable W&B logging (`true` / `false`) |
| `WANDB_PROJECT` | W&B project name |
| `WANDB_ENTITY` | W&B entity / team |

Sample prompt lists are committed under `datasets/`.

Once installed, see the [launch guide](examples/README.md#running-a-recipe) to run an experiment.
