# experimental/refl — WAN ReFL/BPTT (differentiable reward backprop)

The first self-contained UniRL recipe: direct reward backprop (ReFL / DRaFT)
for WAN video models. Two colocated roles — a `ReflActorRole` (FSDP WAN +
grad BPTT sampling + optimizer) and a frozen differentiable reward
(`RewardService`) — run a 3-RPC step under the distributed `enable_grad()`
context: generate → score → backward, then optimizer step. No advantages,
no replay, no rollout engine, no weight sync.

## Launch

One command per config; a Ray cluster must be up (`ray start --head`).

```bash
# WAN 2.1 T2V + VideoAlign (Qwen2-VL VQ/MQ/TA) reward
export PRETRAINED_MODEL=/path/to/Wan2.1-T2V-1.3B-Diffusers \
       VIDEOALIGN_MODEL_PATH=/path/to/VideoReward \
       DATA_PATH=/path/to/prompts.txt
RAY_ADDRESS=auto python -m experimental.refl.run --config-name=wan21_t2v_videoalign_refl num_devices=8

# SD3.5 T2I + PickScore reward (core scorer — no package-local reward)
export PRETRAINED_MODEL=/path/to/stable-diffusion-3.5-medium   # or the HF default
RAY_ADDRESS=auto python -m experimental.refl.run --config-name=sd3_pickscore_refl num_devices=8

# WAN 2.2 I2V + Face-identity reward (first frame via (image, condition)
# MediaRef; face reference via per-sample metadata ref_video_path)
pip install -r experimental/refl/reward/face/requirements.txt
export PRETRAINED_MODEL=/path/to/Wan2.2-I2V-A14B-Diffusers \
       FACE_MODEL_PATH=/path/to/antelodev2 \
       DATA_PATH=/path/to/i2v_prompts.jsonl
RAY_ADDRESS=auto python -m experimental.refl.run --config-name=wan22_i2v_face_refl num_devices=8
```

## Layout

| Path | What |
|---|---|
| `trainer.py` | `REFLTrainer(BaseTrainer)` — driver: wiring + the 3-RPC train step |
| `roles.py` | `ReflActorRole(Remote)` — family-agnostic actor (`pipeline_target` + `model_config`) |
| `models/` | Per-model BPTT adaptations subclassing the core pipelines (`types.py` defines the contract): `wan21.py`, `wan22.py`, `sd3.py` — mirrors `unirl/models/` (graduates into the matching model packages) |
| `reward/` | Package-local differentiable rewards (VideoAlign, Face), each with an additive-only `requirements.txt` — mirrors `unirl/reward/` (graduates into it) |
| `examples/` | Flat Hydra configs (repo-wide schema) — mirrors the top-level `examples/` (graduates into it) |

## Environment

Targets the **locked core stack only** (`transformers>=5.6,<5.7`,
`peft>=0.20` — see `pyproject.toml`). There are deliberately no
version-compat branches: a wrong environment fails loudly; align the
environment, not the code. Reward and actor share one Python process, so
recipe `requirements.txt` files may only ADD packages, never re-pin the
core stack.

## Verification

| Config | Hardware | Head | Status |
| --- | --- | --- | --- |
| `wan21_t2v_videoalign_refl` (835 rollouts) | 8xH20 | pre-adjustment (`e3c6b940` lineage) | contributor long run — reward curve in PR #210 |
| `wan21_t2v_videoalign_refl` (2-rollout smoke, full 81f/480x832 geometry) | 8xH20, fleet image | `40b3f4c9` | PASS — grads flow reward → VAE → DiT LoRA |
| VideoAlign load + differentiable fwd/bwd on transformers 5.6.2 + peft 0.20 | 8xH20 (isolated venv) | `9087a671` | PASS — `grad_abs_mean=3.5e-3` |
| `wan22_i2v_face_refl` | 8xH20 | current head | pending (needs face assets + I2V dataset) |
| `sd3_pickscore_refl` (200 rollouts, ported from the legacy core path; hyperparameters preserved) | 8xH20 (torch 2.11 + transformers 5.6.2 + peft 0.20, fp32 LoRA master) | `9edcab00` | PASS — reward first-10 0.743 → last-10 0.903, matching the legacy-path curve (#120: 0.757 → 0.899) |
