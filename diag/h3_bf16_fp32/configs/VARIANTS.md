# Replay variants

All variants load the same saved batch (192 rows, 24 per rank) and the same step-20 LoRA, and use the Flash-GRPO loss with `old_logp = new_logp.detach()`, KL beta 0.004. Source: `replay-tools/runtime.py` (config changes) and `unirl/utils/h3_checkup_replay.py` (backend flags).

| | V1: bf16 as trained | V2: bf16, speed-ups off | V3: fp32 reference |
|---|---|---|---|
| `bundle.config.model_precision` | bf16 | bf16 | fp32 |
| `backend.fsdp_cfg.param_dtype` / `master_dtype` | bf16 / fp32 | bf16 / fp32 | fp32 / fp32 |
| `backend.fsdp_cfg.mixed_precision` | true | true | false |
| `bundle.config.frozen_adaln_cache` | true | false | false |
| `bundle.config.rollout_keep_unsharded_layers` | 10 | 0 | 0 |
| `forward_prefetch`, `copy_engine_all_gather`, `defer_grad_sync`, `use_torch_compile` | recipe default (false) | false | false |
| TF32 (`allow_tf32`, matmul precision) | PyTorch default | off, `highest` | off, `highest` |
| `allow_bf16_reduced_precision_reduction` | PyTorch default | false | false |
| saved trajectories | fp16 as stored | fp16 as stored | cast to fp32 at replay |
| micro-batch used | 8 (also 1) | 1 (also 8) | 4 (also 2) |

`frozen_adaln_cache` and `rollout_keep_unsharded_layers` are the two speed-ups we added; both default to off in `MiniMaxH3PipelineConfig` (`unirl/models/minimax_h3/config.py:49-50`), so the committed recipe already runs without them. The training run turned them on through the overrides in `long192_overrides.yaml`.

Files here:
- `train_recipe_as_run.yaml`: the recipe the training runs used (HPSv3 reward through `unirl_hps_adapter.py`, paths replaced by placeholders).
- `long192_overrides.yaml`: Hydra overrides of the run whose step-20 checkpoint and batch were replayed.
- `run_train.py`: launcher that mirrors logged metrics to a JSONL file.
