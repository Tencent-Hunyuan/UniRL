# Sol-RL on H20

`sd3_solrl_fp8_h20.yaml` is the 32-H20 SD3.5-Large recipe.  It keeps one
configuration surface for all comparison arms so model, optimizer, reward,
dataset, evaluation, and placement cannot silently drift.

## Arms

Requested H20 adaptation (128 scouts, train 16):

```bash
python -m unirl.train_diffusion \
  --config-name diffusion/sd3/sd3_solrl_fp8_h20
```

BF16 six-step scout control (isolates the FP8 contribution):

```bash
python -m unirl.train_diffusion \
  --config-name diffusion/sd3/sd3_solrl_fp8_h20 \
  rollout.config.fp8_enabled=false \
  scout_sampling.rollout_precision=bf16
```

Paper-shape H20 FP8 bridge (96 scouts, train 24):

```bash
python -m unirl.train_diffusion \
  --config-name diffusion/sd3/sd3_solrl_fp8_h20 \
  contrastive_rollout.top_k=12 contrastive_rollout.bottom_k=12 \
  sampling.samples_per_prompt=24 \
  scout_sampling.samples_per_prompt=96
```

Paper comparator: full 10-step BF16 pool, select 24, no regeneration:

```bash
python -m unirl.train_diffusion \
  --config-name diffusion/sd3/sd3_solrl_fp8_h20 \
  contrastive_rollout.mode=naive \
  contrastive_rollout.top_k=12 contrastive_rollout.bottom_k=12 \
  sampling.samples_per_prompt=24 \
  scout_sampling.samples_per_prompt=96 \
  scout_sampling.num_inference_steps=10 \
  scout_sampling.rollout_precision=bf16 scout_sampling.reward_image_size=null \
  rollout.config.fp8_enabled=false
```

## Interpretation

The paper's exact held-out PickScore prompt split was not released.  The primary
reproduction criterion is therefore the relative HPSv2 gap between the
paper-shape FP8 bridge and BF16 comparator on the same local prompts, seeds, and
training-step budget (target: no worse than 1%).  The paper's absolute SD3.5-L
score, 0.3762, is reported as context rather than a pass/fail threshold.

Before a full run, profile the production-shape DiT and verify that scout calls
execute native E4M3 Tensor Core kernels while regeneration/evaluation do not.
Also compare FP8@6 and BF16@6 rankings against BF16@10 on fixed seeds.
