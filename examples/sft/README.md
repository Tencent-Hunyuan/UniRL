# SFT — supervised finetuning domain

The first non-RL training domain in UniRL. Where the GRPO/ReFL domains do
`rollout → reward → advantage → update`, SFT does the plain supervised loop:
`read data → compute loss → backward → update`. No rollout engine, reward,
advantages, or weight sync.

It mirrors the ReFL domain's shape (a `BaseTrainer` + an FSDP-wrapped policy),
the codified template for non-GRPO domains.

## Model-agnostic by design

The skeleton (`unirl/trainer/sft.py`, `unirl/train/sft/`) knows nothing about
any specific model. It resolves a **task adapter** from config
(`task_target` → `get_class(...).from_config(model_config)`) and drives it.
Everything family-specific — which model to load, how to read a record, and
*what the loss is* — lives in the task. That is why the same skeleton serves
two very different families:

| Task | Family | Loss |
|------|--------|------|
| `unirl.models.qwen3.sft_task.Qwen3SFTTask` | autoregressive (Qwen3) | next-token **cross-entropy** (prompt tokens masked) |
| `unirl.models.sd3.sft_task.SD3SFTTask` | diffusion (SD3.5) | flow-matching **velocity MSE** (target `noise - x0`) |

The contract a task implements is codified in `unirl/train/sft/task.py`
(`SFTTask` Protocol / `SFTTaskBase` ABC): `from_config`, `load_record`,
`compute_loss`, `sample`, plus `.bundle` and `.block_class_names`.

## Module split

- `unirl/train_sft.py` — Hydra entry point (pick a recipe with `--config-name`).
- `unirl/trainer/sft.py` — `SFTTrainer`: the driver loop (data → loss → step),
  inherits `BaseTrainer` for checkpoint / wandb / eval.
- `unirl/train/sft/policy.py` — `SFTPolicy`: FSDP-wrapped worker; broadcasts the
  batch, shards it `[dp_rank::dp_size]`, per-sample `compute_loss().backward()`.
- `unirl/train/sft/data.py` — `JsonlSFTDataSource`: epoch-cycling JSONL reader
  (opaque record dicts; `_root` injected for portable relative paths).
- `unirl/train/sft/task.py` — the task-adapter contract.
- `unirl/models/{qwen3,sd3}/sft_task.py` — the two task adapters.

## Quickstart

```bash
ray start --head
# AR (cross-entropy):
RAY_ADDRESS=auto DATA_PATH=<train.jsonl> EVAL_DATA_PATH=<eval.jsonl> \
  python -m unirl.train_sft --config-name sft/qwen3_sft_smoke num_rollouts=6 eval_interval=3
# Diffusion (flow-matching MSE):
RAY_ADDRESS=auto DATA_PATH=<train.jsonl> EVAL_DATA_PATH=<eval.jsonl> \
  python -m unirl.train_sft --config-name sft/sd3_sft_smoke num_rollouts=6 eval_interval=3
```

Success = the loss falls and eval samples improve.

### Manifest formats (one JSON object per line)

- qwen3: `{"sample_id": str, "prompt": str, "response": str}`
- sd3: `{"sample_id": str, "image_path": str, "prompt": str}` (`image_path`
  relative to the manifest dir)

## Adding a new family

Write `unirl/models/<family>/sft_task.py` implementing the `SFTTask` contract
(reuse the family's existing bundle + forward), add an `examples/sft/<family>_*.yaml`
recipe pointing `task_target`/`model_config` at it. No skeleton changes.

## Notes / debug-scale simplifications

- The `*_smoke.yaml` recipes use LoRA to keep the memory footprint light; drop
  `lora_cfg` for full finetuning.
- Qwen3 SFT reuses `_replay_aware_forward` (the AR RL path's dual-mode forward)
  to get chunked, memory-safe per-token log-probs; it requires an sdpa/flash
  attention impl and disables the cuDNN-SDP backward internally.
- SD3 SFT's `sample()` is a minimal few-step Euler flow-match sampler for a
  qualitative eval only — not the training objective.
