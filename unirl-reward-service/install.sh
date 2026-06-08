#!/usr/bin/env bash
# install.sh — install the RewardService base environment.
#
# This installs only the base layer: ray, fastapi, uvicorn, pillow, and the
# reward_service package itself (editable). Per-scorer dependencies (torch
# versions, transformers, vllm, hpsv2, etc.) are NOT installed here — they
# live in envs/*.txt and are pip-installed automatically by Ray into isolated
# virtualenvs when each scorer actor starts for the first time.
#
# Usage:
#   ./install.sh                       # installs into the active Python env
#
# Assumptions:
#   - You have already activated the target Python >=3.12 environment with
#     torch, nccl, and cuda toolkit pre-installed (these are too large and
#     hardware-specific for pip).
#   - `python3.12` is on PATH, plus either `uv` or `pip`.
#   - Run from the repo root (the directory holding pyproject.toml).

set -euo pipefail

# ─── Locate repo root ──────────────────────────────────────────────────────
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

if [[ ! -f pyproject.toml ]]; then
    echo "install.sh: pyproject.toml not found in ${repo_root}; aborting." >&2
    exit 1
fi

pip_cache="${repo_root}/.pip-cache"
mkdir -p "$pip_cache"

# ─── Verify tooling ─────────────────────────────────────────────────────────
command -v python3.12 >/dev/null || {
    echo "install.sh: python3.12 not on PATH; activate the target env first." >&2
    exit 1
}

if command -v uv >/dev/null; then
    installer_cmd=(uv pip)
    installer_name="uv pip"
elif command -v pip >/dev/null; then
    installer_cmd=(pip)
    installer_name="pip"
else
    echo "install.sh: neither uv nor pip is on PATH; activate the target env first." >&2
    exit 1
fi

echo "== RewardService installer =="
echo "repo root : ${repo_root}"
echo "python    : $(python3.12 --version)"
echo "installer : ${installer_name}"
echo "pip cache : ${pip_cache}"
echo

# ─── Install base environment (server + dev) ───────────────────────────────
echo "[1/2] Removing scorer-level packages that conflict with per-scorer venvs..."
# Per-scorer dependencies must NOT be in the base env: Ray runtime_env
# creates venvs with --system-site-packages, so base packages leak in and
# can cause version conflicts (e.g. diffusers compiled against wrong torch,
# transformers version mismatch). Many Docker images ship these pre-installed.
# Uninstalling from base ensures Ray venvs get a clean slate for each scorer.
pip uninstall -y \
    transformers diffusers accelerate \
    vllm \
    hpsv2 \
    image-reward ImageReward \
    fairscale openai-clip \
    sentence-transformers \
    peft timm \
    ftfy braceexpand \
    flash-attn flash_attn flash-attn-3 flash_attn_3 \
    xformers \
    2>/dev/null || true
echo

echo "[2/2] Installing base dependencies (server + dev)..."
PIP_CACHE_DIR="$pip_cache" "${installer_cmd[@]}" install --cache-dir="$pip_cache" -e ".[server,dev]"
echo

# ─── Sanity check ──────────────────────────────────────────────────────────
echo "== Post-install verification =="
PYTHONPATH="$repo_root" python3.12 - <<'PY'
import importlib

def probe(mod: str) -> None:
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "?")
        print(f"  {mod:15s} {ver}")
    except Exception as e:
        print(f"  {mod:15s} FAIL: {e.__class__.__name__}: {e}")

for mod in ["torch", "ray", "fastapi"]:
    probe(mod)
PY
echo

echo "== Summary =="
echo "Base install: OK"
echo "Per-scorer deps will be installed by Ray runtime_env on first actor start."
echo "See envs/*.txt for each scorer's requirements."
echo "Done."
