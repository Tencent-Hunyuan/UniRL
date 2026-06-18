# Trainer

> **Where it fits:** the orchestration hub — the conductor that builds the workers
> and drives the whole loop: **rollout → reward → advantage → train → sync**. In: a
> composed Hydra recipe (from an entrypoint). Out: a trained model. It is the one
> place that wires every other module together. Full map: [`../README.md`](../README.md)
> (the loop it drives is the [runtime data-flow diagram](../README.md#runtime-data-flow)).

## What it is

`unirl/trainer/` holds the per-domain `<Domain>Trainer` — `DiffusionTrainer`,
`ARTrainer`, `PETrainer`, `UnifiedModelTrainer` — all subclassing `BaseTrainer`.
A trainer is the **driver-side conductor**: it places the rollout and train workers
on GPUs, builds the rollout engine / reward service / train stack(s) / weight-sync
handler, and runs the optimizer loop over them. It owns **placement and sequencing**
— and nothing else: the loss math is `../algorithms`, the optimizer is `../train`,
sampling is `../rollout`, scoring is `../reward`.

## Why it exists

Every other module is deliberately blind to the rest — `rollout` doesn't know
`reward`, an algorithm doesn't know which engine sampled. Something has to wire them
into one loop and decide *where each runs*. That is the trainer. Keeping the wiring
in one per-domain class is what lets the loop body stay ~10 lines and every module
stay swappable by `_target_`.

## How it works

- **`BaseTrainer`** (`base.py`) owns the `DevicePool` (built from the top-level cfg:
  `num_devices` / `transport_kind` + the optional TransferQueue bootstrap) and the
  optional rank-0 wandb logger. Subclasses get the configured pool for free.
- **Build phase** (`__init__`). The trainer builds the remote graph in a
  `placement(...)` scope, threading **one shared bundle** into both consumers —
  `bundle → pipeline(bundle) → backend(bundle) → reward → algorithm → stack` — then
  the rollout engine. This shared-bundle injection is the [models contract](../models/README.md):
  replay reads the exact weights training updates. Layout decides the topology:
  `colocate` builds train + rollout as siblings on one slab; `separate` opens two
  disjoint `placement` slabs and runs a one-time cross-slab handshake for weight sync.
- **The loop** (`train_step`) is the conductor sequence, one rollout per call:
  `wake_up` → (sync the fresh adapter, if due) → `rollout.generate(req)` →
  `reward.score_and_attach(track)` → `track.compute_advantages(...)` → drop the
  reward-only decoded media → `stack.train_track(track)`. The driver first builds the
  typed `RolloutReq` (expanding each prompt into an N-sample GRPO group, authoring the
  `x_T` recipe, resolving the step scheduler's `sde_indices`).

The four trainers are three shapes:

| Trainer | Tracks | Train stack(s) | What's distinctive |
|---|---|---|---|
| `DiffusionTrainer`, `ARTrainer` | 1 | one `TrainStack` | the reference loop; diffusion adds the colocate FSDP-offload and DiffusionNFT EMA-adapter dance around `generate` |
| `PETrainer` | 2 (`ar` + `diffusion`) | two (one per model) | composed *trainside* rollout; the image reward is `propagate_rewards`-credited up to the `ar` track; each track trains its own model |
| `UnifiedModelTrainer` | 2 (`ar` + `image`) | one `UnifiedModelTrainStack` | one shared backbone (HunyuanImage3); both losses backward-accumulate into a single optimizer step |

**Extending it:** a new domain is a new `<Domain>Trainer(BaseTrainer)` that builds its
remotes inside a `placement(...)` scope and implements `train_step` + `train`; the
matching `../train_<domain>.py` entrypoint composes the recipe and calls it.

## Checkpointing

Available for the single-backend trainers (`DiffusionTrainer`, `ARTrainer`,
`UnifiedModelTrainer`); `PETrainer` is not wired. A checkpoint bundles the
model state (`save_mode=auto`: LoRA-only when LoRA is active, otherwise full;
`save_mode=full`: frozen base + LoRA adapters; `save_mode=adapter`: LoRA keys
only — MBs instead of GBs), the optimizer state (gathered full via DCP — not
per-rank shards; it only ever covers the trainable params, so it is
adapter-sized either way), the scheduler state, the step counters (`step`,
`optimizer_step_count`), and the LoRA config (rank / alpha / target_modules —
export tooling reads its scaling from it) — enough to resume training. Each one is written to
`<save_dir>/checkpoint-<step>/checkpoint.pt`. Save and load are collectives
(every rank participates in the gather/broadcast); only dist rank 0 writes the
file, and on load every rank reads it.

**Multi-node**: `save_dir` / `load_dir` must live on storage mounted on every
node — the same contract the recipes already place on `PRETRAINED_MODEL` and
data paths. A rank that cannot see the checkpoint fails fast on every rank at
load (instead of stranding the others in the broadcast until the NCCL timeout).

**Meta-init caveat**: full-state-dict checkpointing rejects bundles with
never-materialized params — the hi3 80B recipe keeps frozen vae/vit on meta, so
`UnifiedModelTrainer` checkpointing currently works only for fully-materialized
bundles. Sharded DCP checkpointing is the planned follow-up.

Driven by top-level config keys, read by the entrypoints and forwarded to
`train(...)`:

| Key | Default | Meaning |
| --- | --- | --- |
| `save_interval` | `0` | Save every N rollouts (and on the last); `0` disables saving. |
| `save_dir` | `./checkpoints` | Output folder for `checkpoint-<step>/`, resolved on the driver (with Hydra's legacy chdir the default lands in the run output dir). |
| `save_mode` | `auto` | `auto` = LoRA-only when LoRA is active, otherwise full; `full` = whole model state; `adapter` = LoRA keys only (the frozen base reloads from the pretrained snapshot on resume). |
| `load_dir` | unset | A checkpoint dir to restore and resume from; unset trains fresh. |

These keys are not in the recipe YAMLs, so append them with Hydra's `+` syntax.
The whole lifecycle — train with saves, resume, fold the LoRA into the base and
export to Hugging Face, share:

```bash
# 1. Train, saving LoRA-only checkpoints every 200 rollouts
bash examples/run_experiment_single_node.sh diffusion/sd3/sd3_trainside \
    num_rollouts=500 \
    +save_interval=200 +save_dir=/ckpts/sd3_run +save_mode=adapter

# 2. Resume (after a preemption, or to extend the budget). num_rollouts is the
#    TOTAL budget (here: rollouts 400..999); the same save_dir is fine —
#    checkpoint numbering continues, and the wandb run reattaches.
bash examples/run_experiment_single_node.sh diffusion/sd3/sd3_trainside \
    num_rollouts=1000 \
    +load_dir=/ckpts/sd3_run/checkpoint-400 \
    +save_interval=200 +save_dir=/ckpts/sd3_run +save_mode=adapter

# 3a. Export a merged model: fold the LoRA into the base weights (scaling from
#     the checkpoint's recorded lora_config) and write a standard save_pretrained folder
python -m unirl.tools.export_full \
    --checkpoint /ckpts/sd3_run/checkpoint-1000 \
    --base stabilityai/stable-diffusion-3.5-medium --subfolder transformer \
    --output /ckpts/sd3_run/hf-1000

# 3b. Or export a PEFT adapter artifact (adapter_model.safetensors +
#     adapter_config.json) for LoRA-aware loaders
python -m unirl.tools.export_adapter \
    --checkpoint /ckpts/sd3_run/checkpoint-1000 \
    --base stabilityai/stable-diffusion-3.5-medium \
    --output /ckpts/sd3_run/adapter-1000

# 4. Share / use the merged model or adapter folder
hf upload <user>/<repo> /ckpts/sd3_run/hf-1000
#   transformer = AutoModel.from_pretrained("<user>/<repo>", torch_dtype=torch.bfloat16)
#   pipe = StableDiffusion3Pipeline.from_pretrained(base, transformer=transformer)
```

`load_dir` restores model/optimizer/scheduler (plus the optimizer-step counter,
so EMA decay schedules continue) and resumes the loop from the saved step:
`training_progress` and the driver-authored x_T noise schedule continue, the
data stream fast-forwards to the resume point (exact when `run.seed` is set —
the shuffle is generator-seeded), and the first rollout force-syncs the
restored adapter into the rollout engine (which booted with fresh weights).
The wandb run also continues: `trainer_state.json` (driver-written, beside
`checkpoint.pt`) carries the run id and the `train/` step axis, and
`_init_wandb` reattaches to that run instead of starting a fresh one.

### Export to Hugging Face format

`checkpoint.pt` is a raw training checkpoint (PEFT-injected names, optimizer
state), not a release artifact. The offline checkpoint toolset lives in
`unirl/tools/` (the runtime counterpart for engine weight sync is
`unirl/utils/peft_merge.py`): `export_full` folds the LoRA delta into the base
weights and writes a standard `save_pretrained` folder; `export_adapter`
extracts a single adapter into a PEFT adapter folder.

Works with both checkpoint flavors: `save_mode=full` merges self-contained;
`save_mode=adapter` folds the LoRA keys onto the freshly loaded base weights.
The LoRA scaling comes from the `lora_config` recorded in the checkpoint;
`--lora-alpha` overrides it (needed only for checkpoints predating the record).
AR models: `--library transformers`, no `--subfolder`. For adapter artifacts,
use `python -m unirl.tools.export_adapter --checkpoint ... --base ... --output ...`.
NFT runs can export the EMA shadow adapter with `--adapter old`.

## Gotchas

- **The reference loop is intentionally minimal** — `num_updates_per_batch` multi-epoch
  replay and **eval cadence are deferred**. Checkpointing is wired (see
  [Checkpointing](#checkpointing)) for the single-backend trainers; `PETrainer` (two
  backends) is not covered.
- **`layout` only branches on `"separate"`** (`"colocate"` == `"colocated"`). The
  trainside direct-sampling engine cannot live on a `separate` slab — `_build_rollout`
  raises (it needs the pipeline as a local sibling).
- **`weight_sync` is built only when a `sync:` block is present** (dedicated engines);
  trainside sampling reads the live training weights and needs none (`self.weight_sync` stays `None`).
- **FSDP offload during `generate` is off by default** and force-gated off for trainside
  (it reuses the train model) and for DiffusionNFT (its EMA swap touches the backend around `generate`).
- **The bundle must be shared, not rebuilt** — the trainer injects one bundle into both
  pipeline and backend; a second `from_config` would silently desync replay. See [`../models/README.md`](../models/README.md).
