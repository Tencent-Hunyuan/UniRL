# Installation

Install from the repository root with uv, using [`pyproject.toml`](pyproject.toml)
for versions and extras. Python **>=3.12,<3.14**, Linux x86_64, NVIDIA GPUs.
The commands below use Python 3.12. uv applies the CUDA index and dependency
overrides in `pyproject.toml`; plain pip ignores `[tool.uv]`.

`vllm` and `sglang` pin incompatible PyTorch stacks — one engine extra per
virtualenv, never `--all-extras`. `train` and `infer` do not pull a rollout engine.

| Engine extra | PyTorch | CUDA |
|---|---|---|
| `vllm` (vLLM + vLLM-Omni) | `2.13.0+cu130` | 13.0 |
| `sglang` | `2.13.0+cu130` | 13.0 |

SGLang's wheel needs glibc >= 2.34. Put NVIDIA's CUDA 13 forward-compat
libraries on `LD_LIBRARY_PATH` before launch; the launchers do not do this.

## vllm-omni

```bash
uv venv --python 3.12 --seed .venv && source .venv/bin/activate
uv pip install -e ".[vllm,train,infer]"
```

To use a Diffusers FlashAttention backend (for example `flash_varlen`) in
diffusion training:

```bash
MAX_JOBS=8 uv pip install -e ".[flash-attn]" --no-build-isolation
```

## sglang

```bash
uv venv --python 3.12 --seed .venv-sglang && source .venv-sglang/bin/activate
```

The `sglang` extra reaches `causal-conv1d` through `flash-linear-attention[conv1d]`,
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

Limit build parallelism with `MAX_JOBS` if host RAM is constrained, then install
the SGLang extra:

```bash
uv pip install -e ".[sglang,train,infer]"
```

## Extras

| Extra | Adds | Use when |
|---|---|---|
| `vllm` | `vllm`, `vllm-omni`, torch +cu130 stack, PyAV | vLLM and vLLM-Omni recipes |
| `flash-attn` | FlashAttention 2 | Diffusers `flash*` attention backends in diffusion training |
| `sglang` | `sglang[diffusion]`, `checkpoint-engine`, `flash-attn-4`, `flash-linear-attention[conv1d]`, torch +cu130 stack, PyAV | SGLang-based AR/VLM and diffusion recipes |
| `fastvideo` | FastVideo pinned to an upstream Git commit | WAN 2.1 / 2.2 rollout; [the extra does not currently resolve](#fastvideo-installation-blocker) |
| `train` | `wandb`, `aiohttp`, `math-verify` | Training runs and local math-answer scoring |
| `cosmos3` | `diffusers>=0.39` | [Cosmos3 SFT](unirl/models/cosmos3/README.md); uv's `diffusers==0.40.0` override already satisfies this extra |
| `infer` | `accelerate`, `timm` | HunyuanImage3, Janus-Pro, and similar models |
| `eval` | `torchvision`, `paddlepaddle`, `paddleocr`, `python-Levenshtein` | OCR-based reward components |
| `veomni` | `veomni` | Recipes using the [VeOmni training backend](unirl/train/backend/veomni/) |
| `dev` | `pytest`, `pytest-cov`, `ruff`, `pre-commit` | Local development |
| `dataset-prep` | `datasets`, `pandas`, `pyarrow`, PyAV | Cooking a dataset with a converter under [`datasets/`](datasets/README.md) |

`dataset-prep` is independent of the engine extras — it carries no torch, so cooking in a
bare venv works for every converter except `datasets/droid100/`, which needs torch as well
(any engine extra supplies it; a plain `uv pip install torch` is enough for CPU-only prep).

`eval` pulls PaddlePaddle for the PP-OCRv5 models behind
[`unirl.reward.local.ocr`](unirl/reward/local/ocr.py). It is not part of the
engine extras; install it when you need OCR rewards.

For development tools (lint and tests):

```bash
uv pip install -e ".[vllm,train,infer,eval,dev]"
# or, for the sglang engine:
uv pip install -e ".[sglang,train,infer,eval,dev]"
```

Prefer these extras over the legacy [`requirements.txt`](requirements.txt) and
`setup.py` paths, which do not match the engine stacks or uv's CUDA index.

### FastVideo installation blocker

The `fastvideo` extra pins
[hao-ai-lab/FastVideo@2095477](https://github.com/hao-ai-lab/FastVideo/blob/2095477eac7e289c7a7ab13acb367ca60687c304/pyproject.toml),
which requires `transformers==4.57.3` and `wandb>=0.21.0`. The transformers pin
conflicts with UniRL's `transformers>=5.12,<5.13`, so `.[fastvideo]` does not
resolve — a separate venv does not help, because UniRL's base deps still apply.
Adding `train` also conflicts on `wandb`. Use `$FASTVIDEO_PATH` as in the
[FastVideo engine README](unirl/rollout/engine/fastvideo/README.md) until the extra
is solvable.

### Cosmos3

`cosmos3` asks for `diffusers>=0.39`. The uv override pins `diffusers==0.40.0`,
which already satisfies that floor, so install it as a normal extra:

```bash
uv pip install -e ".[vllm,train,infer,cosmos3]"
```

## Environment

Recipes read cluster-local paths and W&B settings from `${oc.env:...}`. A
variable only affects fields that reference it; use a Hydra override for
literal values. Common names:

| Variable | Purpose |
|---|---|
| `PRETRAINED_MODEL` | Base model checkpoint path |
| `QWEN3_PATH` / `QWEN_VL_PATH` | Model-specific checkpoint paths in Qwen recipes |
| `DATA_PATH` | Training data / prompt-list path |
| `EVAL_DATA_PATH` | Evaluation data path |
| `SFT_DATA` / `SFT_EVAL_DATA` | Training / evaluation manifests in SFT recipes |
| `HF_TOKEN` | Hugging Face token for gated models (e.g. SD3.5) |
| `REPORT_TO_WANDB` | Enable W&B logging (`true` / `false`) |
| `WANDB_PROJECT` | W&B project name |
| `WANDB_ENTITY` | W&B entity / team |

Recipes also use model-specific names such as `BAGEL_PATH` and `LLM_MODEL`;
check the selected YAML. Sample prompt lists are committed under `datasets/`.

Once installed, see the [launch guide](examples/README.md#running-a-recipe) to run an experiment.
