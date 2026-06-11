# DGX Spark / GB10 / linux-aarch64 bring-up

This page tracks the supported path for running UniRL on DGX Spark-class hosts:

- CPU/OS architecture: `aarch64`
- GPU: NVIDIA GB10
- CUDA user stack: 13.x
- glibc: 2.39+
- Python: 3.12 aarch64

The CUDA driver/runtime can be present while Python packages and native CUDA
extensions are still the blocker. Treat the stack as layers and validate them in
order.

## Current support boundary

The `spark` extra is the DGX Spark smoke-test stack. It deliberately does not
install rollout engines (`sglang`, `vllm`, `vllm-omni`, `flash-attn`,
`sglang-kernel`) because those packages frequently lag linux-aarch64/GB10/CUDA 13
wheel availability.

Expected first milestone:

1. PyTorch CUDA imports and sees GB10.
2. Diffusers/Transformers/PEFT/Ray import.
3. UniRL core imports.
4. Hydra configs compose.

Only after that should we try the engine extras.

## Install

Use a fresh virtual environment. `uv` is preferred because this project declares
the CUDA torch indexes in `pyproject.toml`.

```bash
python3 -m venv .venv-spark
source .venv-spark/bin/activate
python -m pip install -U pip uv
uv pip install -e ".[spark,train,infer,dev]"
```

If using `pip` directly, pass the CUDA 13 PyTorch index explicitly:

```bash
python3 -m venv .venv-spark
source .venv-spark/bin/activate
python -m pip install -U pip
python -m pip install -e ".[spark,train,infer,dev]" \
  --extra-index-url https://download.pytorch.org/whl/cu130
```

## Validate layers

```bash
source .venv-spark/bin/activate
python scripts/dgx_spark_probe.py
python -m unirl.train_diffusion --config-name=diffusion/sd3_trainside --cfg job --resolve
```

The probe exits non-zero if the required DGX Spark smoke-test layer fails. It also
prints optional failures for `sglang`, `vllm`, `flash_attn`, and
`sglang_kernel`; those are expected until their linux-aarch64 CUDA 13 wheels (or a
working local source-build recipe) are available.

## Engine triage order

After the smoke-test stack passes, test native engines one at a time in separate
venvs:

1. `sglang-kernel`
2. `flash-attn` / `flash-attn-4`
3. `sglang[diffusion]`
4. `vllm`
5. `vllm-omni`

Do not mix `sglang` and `vllm` in the same environment; their CUDA/PyTorch pins
are intentionally conflicting.

For every native-extension failure, record:

- package and version
- whether a linux-aarch64 wheel exists
- Python tag (`cp312` here)
- CUDA version encoded by the wheel or build
- PyTorch version and CUDA ABI
- compiler and CUDA header versions if building from source
- GB10 compute capability reported by `scripts/dgx_spark_probe.py`

## Known risks

- x86_64 wheels are not usable on DGX Spark.
- Docker images for research ML projects often publish only `linux/amd64`.
- CUDA 13 support does not imply GB10/Blackwell tuning support.
- Source builds can fail independently on Python 3.12, PyTorch ABI, CUDA headers,
  compute capability flags, or Arm-specific compiler issues.
