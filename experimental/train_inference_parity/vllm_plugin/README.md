# UniRL train/inference parity vLLM plugin

An opt-in, correctness-first plugin for Qwen3-30B-A3B. It is not a general
performance optimization or a claim of portable bitwise reproducibility.

## Installation

Install from this directory with `pip install -e '.[verified]'`. The `verified`
extra pins vLLM 0.27.0, Torch 2.13.0, and Transformers 5.6.0. UniRL must also be
installed for its native weight-sync worker extension. The parent experiment's
README documents the four-GPU recipe and two-phase verification.

The wheel registers `unirl_train_inference_parity` in `vllm.general_plugins`.
Without `UNIRL_PARITY_ENABLE=1`, registration imports neither Torch nor vLLM and
does not patch anything. An enabled worker requires all of:

```bash
export UNIRL_PARITY_ENABLE=1
export UNIRL_PARITY_PROFILE=public_reference
export UNIRL_PARITY_MODEL=qwen3_moe_30b_a3b
export UNIRL_PARITY_PATCHES=common,qwen3_moe_30b_a3b
export UNIRL_PARITY_STRICT=1
export VLLM_PLUGINS=unirl_train_inference_parity
```

Use the experiment launcher rather than these flags alone for parity validation:
it also configures precision, attention, NCCL, and Ray propagation.

## Ownership

- `common/`: shared numerical providers, precision, attention, and reductions.
- `models/qwen3_moe_30b_a3b/`: Qwen-specific routing, projections, MoE, and reload.
- `compat.py`: pinned versions and private-symbol signature checks.
- `registry.py`: preflight checks and before/after installation evidence.

Training imports the public-reference providers from this package; the plugin
does not import the training experiment. Neither direction may import a sibling
experiment. Core UniRL does not import this plugin.

## Gotchas

- Registration is process-local and idempotent. After installation, changing the
  enable flag does not undo CUDA overrides: start a new process instead.
  A failed installation also requires a restart, because mutation may be partial.
- Preflight checks detect known upstream signature and provider drift. They are
  not proof of numerical equality; rerun both verification phases after upgrades.
- The custom `o_proj` changes row sharding to column sharding. Generic vLLM
  meta-staged reload can infer the wrong layout despite equal element counts.
  Its dedicated subclass therefore bypasses meta staging and loads the full
  canonical weight directly into the existing CuMem allocation.
- Do not wrap the expert Parameter loaders: vLLM introspects them during
  layerwise reload. Derived MoE columns are invalidated at module load and before
  worker sleep so cached allocations never survive weight replacement.
- Keep the TP routing agreement checks before route-dependent collectives.
  Removing them can turn a routing mismatch into a distributed hang.
