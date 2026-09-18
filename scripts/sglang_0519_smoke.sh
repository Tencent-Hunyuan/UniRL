#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PIP_INDEX_URL=https://mirrors.cloud.tencent.com/pypi/simple
export PIP_TRUSTED_HOST=mirrors.cloud.tencent.com

python3 - <<'PY'
import torch

assert torch.__version__.startswith("2.13.0"), torch.__version__
assert torch.version.cuda and torch.version.cuda.startswith("13."), torch.version.cuda
assert torch.cuda.is_available()
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0))
PY

python3 -m pip install \
  --no-deps \
  --force-reinstall \
  "sglang==0.5.19"
python3 -m pip install --no-build-isolation --no-deps -e .

python3 - <<'PY'
from importlib import metadata

assert metadata.version("sglang") == "0.5.19"
assert metadata.version("transformers") == "5.12.1"
assert metadata.version("torch").startswith("2.13.0")

from unirl.rollout.engine.sglang.backends.http import _import_sglang_runtime
from unirl.rollout.engine.sglang.backends.native import _import_sglang_engine
from unirl.rollout.engine.sglang_diffusion.backends.native import (
    _import_sglang_runtime as import_diffusion_runtime,
)

_import_sglang_runtime()
_import_sglang_engine()
import_diffusion_runtime()
print("SGLang 0.5.19 AR and diffusion imports passed")
PY

PYTHONPATH="$PWD" python3 -m pytest -q \
  tests/test_sglang_0519_migration.py \
  tests/test_sglang_ar_0519.py

nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used \
  --format=csv,noheader
