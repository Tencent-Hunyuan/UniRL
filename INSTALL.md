# Installation

Use [`pyproject.toml`](pyproject.toml) as the source of truth for dependency
versions and extras. It requires Python **>=3.12,<3.14** and configures uv for
**Linux x86_64**; the training examples below target NVIDIA GPUs. Run installation
commands from the repository root. The examples use Python 3.12.

The `vllm` and `sglang` extras have mutually exclusive PyTorch stacks — use a
separate virtual environment for each and do not install `--all-extras`.
FastVideo is a declared engine extra with a
[known installation blocker](#fastvideo-installation-blocker).

| Engine extra | PyTorch | CUDA |
|---|---|---|
| `vllm` (vLLM + vLLM-Omni) | `2.13.0+cu130` | 13.0 |
| `sglang` | `2.11.0+cu130` | 13.0 |

SGLang's wheel requires glibc >= 2.34. If your deployment uses NVIDIA's CUDA 13
forward-compatibility layer, provision it in the image and add its library
directory to `LD_LIBRARY_PATH` before launching. The launchers do not discover
or load compatibility libraries automatically; setting `CUDA_COMPAT_DIR` alone
does not configure the library search path.

The commands below use uv so that the CUDA wheel index and dependency overrides
in `pyproject.toml` apply. Plain pip does not automatically use `[tool.uv]`
settings. The base dependencies do not pin an engine-specific PyTorch/CUDA
stack. Installing only `train` or `infer` does not establish compatibility with
a particular rollout engine.

## vllm-omni

```bash
uv venv --python 3.12 --seed .venv && source .venv/bin/activate
uv pip install -e ".[vllm,train,infer]" --prerelease=allow
```

## sglang

```bash
uv venv --python 3.12 --seed .venv-sglang && source .venv-sglang/bin/activate
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

Limit build parallelism with `MAX_JOBS` if host RAM is constrained, then install
the SGLang extra:

```bash
uv pip install -e ".[sglang,train,infer]" --prerelease=allow
```

## Extras

| Extra | Adds | Use when |
|---|---|---|
| `vllm` | `vllm`, `vllm-omni`, torch +cu130 stack, PyAV | Running any vllm-omni-based example |
| `sglang` | `sglang[diffusion]`, `checkpoint-engine`, `flash-attn-4`, `flash-linear-attention[conv1d]`, torch +cu130 stack, PyAV | SGLang-based AR/VLM and diffusion recipes |
| `fastvideo` | FastVideo pinned to an upstream Git commit | Declared for WAN 2.1 / 2.2 rollout; [installation currently blocked](#fastvideo-installation-blocker) |
| `train` | `wandb`, `aiohttp`, `math-verify` | Training runs and local math-answer scoring |
| `cosmos3` | `diffusers>=0.39` | [Cosmos3 SFT](unirl/models/cosmos3/README.md); apply the [version constraint](#cosmos3-version-prerequisite) |
| `infer` | `accelerate`, `timm` | HunyuanImage3, Janus-Pro, and similar models |
| `eval` | `torchvision`, `paddlepaddle`, `paddleocr`, `python-Levenshtein` | OCR-based reward components |
| `veomni` | `veomni` | Recipes using the [VeOmni training backend](unirl/train/backend/veomni/) |
| `dev` | `pytest`, `pytest-cov`, `ruff`, `pre-commit` | Local development |
| `dataset-prep` | `datasets`, `pandas`, `pyarrow`, PyAV | Cooking a dataset with a converter under [`datasets/`](datasets/README.md) |

`dataset-prep` is independent of the engine extras — it carries no torch, so cooking in a
bare venv works for every converter except `datasets/droid100/`, which needs torch as well
(any engine extra supplies it; a plain `uv pip install torch` is enough for CPU-only prep).

`eval` pulls PaddlePaddle, a second deep-learning framework needed to run
the PP-OCRv5 detection and recognition models behind
[`unirl.reward.local.ocr`](unirl/reward/local/ocr.py). It is deliberately kept out of
`requirements.txt` for that reason, so install it explicitly when you need OCR rewards.

For development tools (lint and tests):

```bash
uv pip install -e ".[vllm,train,infer,eval,dev]" --prerelease=allow
# or, for the sglang engine:
uv pip install -e ".[sglang,train,infer,eval,dev]" --prerelease=allow
```

Use these pyproject-based installation paths for new environments. The legacy
[`requirements.txt`](requirements.txt) and direct `setup.py` installation paths
are not recommended: their dependency declarations differ from the current
engine extras and do not replace uv's CUDA index and overrides.

### FastVideo installation blocker

The `fastvideo` extra is declared, but its
[pinned upstream revision](https://github.com/hao-ai-lab/FastVideo/blob/2095477eac7e289c7a7ab13acb367ca60687c304/pyproject.toml)
has incompatible dependency requirements: `transformers==4.57.3` conflicts
with UniRL's base `transformers>=5.6,<5.7`, and `wandb>=0.21.0` conflicts with
`train`'s `wandb>=0.16,<0.20`.

Standard dependency resolution is therefore blocked for `.[fastvideo]`, even
without selecting SGLang or vLLM; `.[fastvideo,train]` adds the W&B conflict.
A separate virtual environment does not resolve these metadata conflicts.
This path needs a compatible dependency set or a maintainer-validated
installation procedure before it can be recommended. These conflicts were
identified from dependency metadata, not a full installation attempt.

### Cosmos3 version prerequisite

The `cosmos3` extra declares `diffusers>=0.39`, but the shared uv override
`diffusers>=0.38.0` [replaces dependency requirements](https://docs.astral.sh/uv/concepts/resolution/#dependency-overrides)
rather than intersecting with the extra's stricter minimum. Selecting the extra
alone does not guarantee that an existing diffusers 0.38 installation is upgraded.

Create a constraint file, include `cosmos3` in the selected environment's extras,
and append `--constraint "$COSMOS3_CONSTRAINTS"` to that installation command:

```bash
COSMOS3_CONSTRAINTS="$(mktemp)"
printf '%s\n' 'diffusers>=0.39' > "$COSMOS3_CONSTRAINTS"
```

Keep applying this constraint when resolving dependencies for Cosmos3. After
installation, check the version in the same activated environment:

```bash
python - <<'PY'
from importlib.metadata import version
from packaging.version import Version

installed = version("diffusers")
if Version(installed) < Version("0.39"):
    raise SystemExit(f"Cosmos3 requires diffusers>=0.39; found {installed}")
print(f"diffusers version prerequisite passed: {installed}")
PY
```

This is a version prerequisite check, not a Cosmos3 runtime smoke test. The
constraint prevents a resolution below the required minimum; a complete
Cosmos3/engine installation and training run have not been validated here.

## Environment

Example configs read cluster-local paths, checkpoints, data, and W&B settings from
environment variables via `${oc.env:...}`. Check the selected YAML: a variable
only affects fields that reference it; use Hydra overrides for literal values.
Common variables:

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

Sample prompt lists are committed under `datasets/`.

Once installed, see the [launch guide](examples/README.md#running-a-recipe) to run an experiment.
