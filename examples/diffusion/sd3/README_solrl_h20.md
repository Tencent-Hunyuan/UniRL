# Sol-RL on H20

`sd3_solrl_fp8_h20.yaml` is the 32-H20 SD3.5-Large recipe.  It keeps one
configuration surface for all comparison arms so model, optimizer, reward,
dataset, evaluation, and placement cannot silently drift.

The task image must provide a CUDA/PyTorch-compatible Transformer Engine 2.2+
(`pip install -e ".[train,fp8]"`) plus an importable `hpsv2` package and the two
checkpoints configured by `HPSV2_OPEN_CLIP` and `HPSV2_CHECKPOINT`. Model-specific
reward packages are intentionally supplied by task images rather than pinned in
UniRL's shared training extra.

Trace-producing runs also require an immutable checkpoint revision or checksum:

```bash
export SOLRL_POLICY_SNAPSHOT_ID=<immutable-checkpoint-revision-or-sha256>
```

Reuse this ID only when both arms load exactly the same policy snapshot.

## Arms

Requested H20 adaptation (128 scouts, train 16):

```bash
SOLRL_TRACE_DIR=traces/fp8-128x16 WANDB_RUN_NAME=sd35l-solrl-fp8-128x16 \
python -m unirl.train_diffusion \
  --config-name diffusion/sd3/sd3_solrl_fp8_h20
```

BF16 six-step scout control (isolates the FP8 contribution):

```bash
SOLRL_TRACE_DIR=traces/bf16-128x16 WANDB_RUN_NAME=sd35l-solrl-bf16-128x16 \
python -m unirl.train_diffusion \
  --config-name diffusion/sd3/sd3_solrl_fp8_h20 \
  rollout.config.fp8_enabled=false \
  scout_sampling.rollout_precision=bf16 \
  'logging.tags=[sd3.5-large,sol-rl,bf16-scout,h20,diffusionnft]'
```

Paper-shape H20 FP8 bridge (96 scouts, train 24):

```bash
SOLRL_TRACE_DIR=traces/fp8-96x24 WANDB_RUN_NAME=sd35l-solrl-fp8-96x24 \
python -m unirl.train_diffusion \
  --config-name diffusion/sd3/sd3_solrl_fp8_h20 \
  contrastive_rollout.top_k=12 contrastive_rollout.bottom_k=12 \
  sampling.samples_per_prompt=24 \
  scout_sampling.samples_per_prompt=96
```

Paper comparator: full 10-step BF16 pool, select 24, no regeneration:

```bash
SOLRL_TRACE_DIR=traces/bf16-naive-96x24 WANDB_RUN_NAME=sd35l-solrl-bf16-naive-96x24 \
python -m unirl.train_diffusion \
  --config-name diffusion/sd3/sd3_solrl_fp8_h20 \
  contrastive_rollout.mode=naive \
  contrastive_rollout.top_k=12 contrastive_rollout.bottom_k=12 \
  sampling.samples_per_prompt=24 \
  scout_sampling.samples_per_prompt=96 \
  rollout.config.fp8_enabled=false \
  'logging.tags=[sd3.5-large,sol-rl,bf16-naive,h20,diffusionnft]'
```

Naive mode derives its generation policy from `sampling`; only the scout fanout
and optional reward-image resize remain scout-specific. Cross-run ranking is
valid only at rollout 0, before the two training arms update into different
policies. Compare that shared initial snapshot with:

```bash
python -m unirl.tools.solrl_rank_metrics \
  --proxy-dir traces/fp8-96x24 \
  --oracle-dir traces/bf16-naive-96x24 \
  --rollout-id 0
```

## Interpretation

The paper's exact held-out PickScore prompt split was not released.  The primary
reproduction criterion is therefore the relative HPSv2 gap between the
paper-shape FP8 bridge and BF16 comparator on the same local prompts, seeds, and
training-step budget (target: no worse than 1%).  The paper's absolute SD3.5-L
score, 0.3762, is reported as context rather than a pass/fail threshold.

Before a full run, profile the production-shape DiT and verify that scout calls
execute native E4M3 Tensor Core kernels while regeneration/evaluation do not.
Also compare FP8@6 and BF16@6 rankings against BF16@10 on fixed seeds at
rollout 0. Later-policy comparison requires a paired-oracle mode that generates
both arms from one immutable policy snapshot; independent training runs are not
valid oracle/proxy pairs after their first update.
