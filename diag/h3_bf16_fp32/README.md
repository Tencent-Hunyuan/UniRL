# H3 bf16 vs fp32 gradient diagnosis — launchers, standalone script, configs

Diagnosis-only files for this PR, which is not meant to be merged. AGENTS.md does not allow verification harnesses in the tree, so this folder would be dropped from anything that goes upstream.

- `replay-tools/`: launchers for the fixed-batch replay in `unirl/utils/h3_checkup_*.py`. `capture.py` saves the batch, `replay.py` recomputes gradients for one variant, `runtime.py` builds each variant's config, `chain.py` runs the sequence, `cmp_micro.py` compares two exported gradient sets (overall and per-block cosine, norm ratio, relative L2), `*_cpu.py` are CPU self-checks. `env.sh` needs `TRAIN_ROOT` and `REPLAY_ROOT`.
- `standalone/`: `standalone_micro.py` needs no UniRL (plain PyTorch, real weights, the 50 blocks placed across 8 GPUs). `capture_hook.py` and `capture_supervisor.py` capture one micro-batch from the replay, `cpu_validate.py` is the CPU check, `diag_g1.py` is the forward-only input-layout diagnosis.
- `configs/`: the recipe and overrides as run, and `VARIANTS.md`, the table of replay variants (V1 = bf16 as trained, V2 = bf16 with our accelerations off, V3 = fp32).

`replay.py` is the final host version: it also allows V1 with micro-batch 1 and V2 with micro-batch 8, written to a separate folder through `CHECKUP_OUT_NAME` with `CHECKUP_SPLITS=full`.
