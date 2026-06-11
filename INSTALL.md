# Installation

UniRL ships two mutually exclusive inference engines (`vllm` and `sglang`) — install each in its own virtual environment.
DGX Spark / GB10 / linux-aarch64 users should start with the `spark` smoke-test
extra before attempting engine extras; see [DGX Spark bring-up](docs/dgx-spark-aarch64.md).

| Engine / stack | CUDA | glibc | Arch |
|---|---|---|---|
| **spark smoke test** | 13.0 | ≥ 2.39 | linux/aarch64 |
| **vllm-omni** | 12.9 | ≥ 2.28 | linux/x86_64 verified |
| **sglang** | 13.0 | ≥ 2.34 | linux/x86_64 verified |

## DGX Spark / GB10 / linux-aarch64

```bash
python3 -m venv .venv-spark && source .venv-spark/bin/activate
python -m pip install -U pip uv
uv pip install -e ".[spark,train,infer,dev]"
python scripts/dgx_spark_probe.py
```

The probe validates the layer order that usually finds the first Arm/CUDA break:
PyTorch CUDA → Diffusers/Transformers/PEFT/Ray → UniRL core → optional rollout
engines. Optional failures for `sglang`, `vllm`, `flash_attn`, and
`sglang_kernel` are expected until their linux-aarch64 CUDA 13 wheels or local
source-build recipes are confirmed.

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
