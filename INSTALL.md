# Installation

UniRL ships two mutually exclusive inference engines (`vllm` and `sglang`) — install each in its own virtual environment.

| Engine | CUDA | glibc |
|---|---|---|
| **vllm-omni** | 13.0 | ≥ 2.28 |
| **sglang** | 13.0 | ≥ 2.34 |

Both engines are CUDA 13 now. On the driver-535 fleet that means NVIDIA's
`cuda-compat-13-0` forward-compat layer has to be on `LD_LIBRARY_PATH`; the
launchers find it under `.cuda-compat-13/` or `/usr/local/cuda-13.*/`, or take an
explicit `CUDA_COMPAT_DIR`.

## vllm-omni

```bash
uv venv --python 3.12 --seed .venv && source .venv/bin/activate
uv pip install -e ".[vllm,train,infer]" --prerelease=allow
```

## sglang

```bash
uv venv --python 3.12 --seed .venv-sglang && source .venv-sglang/bin/activate
uv pip install -e ".[sglang,train,infer]" --prerelease=allow
```

This extra reaches `causal-conv1d` through `flash-linear-attention[conv1d]`,
which has no wheel and compiles a CUDA extension. Torch refuses to build one
against a different CUDA major than its own, so a CUDA 12 `nvcc` on `PATH` fails
the install with a version-mismatch `RuntimeError`. Install the CUDA 13 compiler
wheels first and point `CUDA_HOME` at them — the same toolkit SGLang's runtime
JIT uses:

```bash
uv pip install "nvidia-cuda-nvcc==13.0.*" "nvidia-cuda-crt==13.0.*" \
    "nvidia-nvvm==13.0.*" "nvidia-cuda-cccl==13.0.*" "nvidia-cuda-runtime==13.0.*"
export CUDA_HOME="$VIRTUAL_ENV"/lib/python3.12/site-packages/nvidia/cu13
export PATH="$CUDA_HOME/bin:$PATH"
```

The compile is memory-hungry; set `MAX_JOBS` to something a machine with 15 GB
can survive.

## Extras

| Extra | Adds | Use when |
|---|---|---|
| `vllm` | `vllm`, `vllm-omni`, torch +cu130 stack, PyAV | Running any vllm-omni-based example |
| `sglang` | `sglang[diffusion]`, `flash-attn-4`, torch +cu130 stack, PyAV | Running VLM/LLM examples or `sd3_sglang_*` |
| `train` | `wandb`, `aiohttp` | Training runs (almost always wanted) |
| `infer` | `accelerate` | HunyuanImage3 and similar models |
| `eval` | `torchvision`, `easyocr` | OCR-based reward components |
| `dev` | `pytest`, `ruff`, `pre-commit` | Local development |
| `dataset-prep` | `datasets`, `pandas`, `pyarrow`, PyAV | Cooking a dataset with a converter under [`datasets/`](datasets/README.md) |

`dataset-prep` is independent of the engine extras — it carries no torch, so cooking in a
bare venv works for every converter except `datasets/droid100/`, which needs torch as well
(any engine extra supplies it; a plain `uv pip install torch` is enough for CPU-only prep).

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
