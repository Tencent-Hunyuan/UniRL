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
very different families:

| Task | Family | Loss |
|------|--------|------|
| `unirl.models.qwen3.sft_task.Qwen3SFTTask` | autoregressive (Qwen3) | next-token **cross-entropy** (prompt tokens masked) |
| `unirl.models.sd3.sft_task.SD3SFTTask` | diffusion (SD3.5) | flow-matching **velocity MSE** (target `noise - x0`) |
| `unirl.models.cosmos3.sft_task.Cosmos3VideoSFTTask` / `Cosmos3ActionBCTask` | omnimodal MoT (Cosmos3-Nano) | flow-matching **velocity MSE** (video / co-denoised action+video) |

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
- `unirl/models/{qwen3,sd3,cosmos3}/sft_task.py` — the task adapters.

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

---

# Cosmos3 — omnimodal SFT / behavior-cloning

Supervised flow-matching finetuning for **NVIDIA Cosmos3-Nano** (16B omnimodal
MoT world model), including robot **action-trajectory behavior cloning** in the
style of Cosmos3-Nano-Policy-DROID. This is a diffusion-family task on the same
skeleton above; the loss + packing live entirely in the Cosmos3 task/wrapper.

## Cosmos3 module split

| Layer | Code | Role |
|---|---|---|
| Cosmos3 wrapper | `unirl/models/cosmos3/{config,bundle,packing}.py` | Weights + freeze policy (und/AR tower frozen, gen tower trainable), joint-sequence packing that reuses `diffusers.Cosmos3OmniPipeline`'s own helpers, flow-matching sigma/noise/velocity math. |
| Task adapters | `unirl/models/cosmos3/sft_task.py` | `Cosmos3VideoSFTTask` (t2i / t2v / video prediction) and `Cosmos3ActionBCTask` (policy-mode BC: obs + instruction → action chunk + co-denoised future video). |
| Data prep | `unirl/utils/prepare_droid100.py` | LeRobot-v3 → self-contained debug samples (uint8 clips + z-normalized action chunks + JSONL manifests). |

## Requirements

- `diffusers>=0.39` (first release with `Cosmos3OmniTransformer` / `Cosmos3OmniPipeline`).
- The trainer venv only — no inference engine (sglang / vllm-omni) is involved.
- Checkpoint: `nvidia/Cosmos3-Nano` (diffusers layout; ~33 GB for
  `transformer/ vae/ text_tokenizer/ scheduler/`).

## Cosmos3 quickstart

```bash
# 1. Data: droid_100 (0.93 GB, 100 episodes, LeRobot v3) -> debug windows
python -m unirl.utils.prepare_droid100 --root datasets/droid100_debug

# 2. First milestone — video prediction (obs frame + instruction -> 16 frames)
export PRETRAINED_MODEL=/path/to/Cosmos3-Nano
ray start --head
RAY_ADDRESS=auto python -m unirl.train_sft --config-name sft/cosmos3_droid100_videopred

# 3. Action BC (policy mode, the Cosmos3-Nano-Policy-DROID objective)
RAY_ADDRESS=auto python -m unirl.train_sft --config-name sft/cosmos3_droid100_action_bc
```

Each step covers the full chain: dataset → collate (worker-side) → packed MoT
forward → masked velocity MSE → backward → clip + AdamW step → periodic
checkpoint (`save_interval`) → periodic samples (`eval_interval`, written to
`<save_dir>/samples/step_N.pt` as `{"video": [T,C,H,W], "action": [T,D]}`).

## How Cosmos3 training matches inference

- Packing calls the *pipeline's own* `tokenize_prompt` / `_prepare_*_segment`
  helpers, so prompts (chat template + metadata sentences + `<|vision_start|>`),
  mRoPE ids, and sequence layout are bit-identical to `Cosmos3OmniPipeline.__call__`.
- VAE encoding uses `_encode_video` (argmax mode + per-channel mean/std, amp off).
- Velocity target is `v = eps - x0` (the UniPC `flow_prediction` convention:
  `x0_pred = sample - sigma * v`); sigma is logitnormal, warped by the same
  `flow_shift` map `set_timesteps` uses.
- Policy-mode BC: latent frame 0 clean (no timestep embedding, excluded from
  loss), future frames + the whole action chunk noised with one shared sigma —
  the joint objective Cosmos3-Nano-Policy-DROID was post-trained with
  (`action_loss_weight=10`, `flow_shift=5.0` per the upstream action recipes).

## Known Cosmos3 debug-scale simplifications

- **droid_100 actions are LeRobot's collapsed 7-D `action` field** (≈ 6-D
  cartesian velocity + gripper), not the 10-D EE-pose layout the
  `droid_lerobot` domain head (id 8) was pretrained on, and not the 8-D
  absolute joint positions Policy-DROID uses. Fine for a debug BC run — the
  head is being finetuned — but for faithful Policy-DROID reproduction, prep
  `nvidia/Cosmos3-DROID` (success split) with `action.joint_position` +
  `gripper_position` instead.
- No proprioceptive-state stream (the released inference stacks don't consume
  one either); NVIDIA's internal recipe additionally conditions on state.
- Full finetune of the gen stream only; `lora_cfg` is wired through
  `FSDPBackend` if adapter training is preferred.
