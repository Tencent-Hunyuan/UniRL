# Trainer

> **Where it fits:** the orchestration hub — the conductor that builds the workers
> and drives the whole loop: **rollout → reward → advantage → train → sync**. In: a
> composed Hydra recipe (from an entrypoint). Out: a trained model. It is the one
> place that wires every other module together. Full map: [`../README.md`](../README.md)
> (the loop it drives is the [runtime data-flow diagram](../README.md#runtime-data-flow)).

## What it is

`unirl/trainer/` holds the driver-side trainers for diffusion, autoregressive,
supervised fine-tuning, prompt-enhancement, unified-model, ReFL, asynchronous,
and agentic runs. They
subclass `BaseTrainer` directly or through a domain trainer. A trainer places the
rollout and train workers on GPUs, builds the rollout engine / reward service /
train stack(s) / weight-sync handler, and runs the optimizer loop over them. It
owns **placement and sequencing** — and nothing else: the loss math is
`../algorithms`, the optimizer is `../train`, sampling is `../rollout`, and
scoring is `../reward`.

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
- **The Sample-native loop** (`train_step`) is the conductor sequence, one
  rollout per call: `wake_up` → (sync weights, if due) →
  `rollout.generate(sample)` → `reward.score_and_attach(sample)` →
  `part.compute_advantages(...)` → drop reward-only decoded media →
  `stack.train_track(...)`. The driver builds a request `Sample` whose Parts
  preserve prompt lineage and carry sampling parameters. A single-stage stack
  receives the trainable frontier `Part`; `UnifiedModelTrainStack` receives the
  whole `Sample` so AR and image Parts are sharded by the same prompt trees.
  Agentic engines return a `List[Sample]` of variable-depth trajectories; their
  trainers assign each trajectory's advantage to all generated turns and
  concatenate those turn Parts for training. (ReFL — which differentiates
  directly through decoded media and uses no rollout Samples or advantages —
  lives outside core as `experimental/refl`.)

The current trainer surface is:

| Trainer | Training shape | What's distinctive |
|---|---|---|
| `DiffusionTrainer` | one diffusion `Part` → one `TrainStack` | Reference diffusion loop; supports trainside or dedicated rollout, optional separate reward GPUs, FSDP offload, and DiffusionNFT's EMA-adapter rollout. |
| `ARTrainer` | one AR `Part` → one `TrainStack` | Text or multimodal AR rollout with group/global advantage normalization and optional token-balanced DP shards. |
| `SFTTrainer` | dataset records → one standalone training `Part` | Reuses the RL TrainStack without rollout, reward, or advantages; owns exact epoch/cursor resume and full-set evaluation. |
| `AsyncARTrainer` | FIFO AR generation batch → one `TrainStack` | Separate train/rollout slabs with resident generation, batch-denominated `weight_sync_interval` publication (`max_staleness = interval - 1`) over an optimizer-update clock, and quiescence before weight sync, eval, or checkpoint. |
| `AsyncDiffusionTrainer` | FIFO diffusion generation batch → one `TrainStack` | The same update-versioned separate-slab loop for DiT. Requires `max_inflight=1` and reaps each generation before launching the next, so the cross-slab trajectory transfer never queues behind a fresh generation. |
| `PETrainer` | `ar` + `diffusion` Parts → two `TrainStack`s | Composed prompt-rewrite/image rollout; image rewards propagate to AR rewrites. `freeze_llm=true` trains and checkpoints diffusion only. |
| `UnifiedModelTrainer` | whole `Sample` → one `UnifiedModelTrainStack` | AR and image losses accumulate into shared-backbone optimizer steps while prompt-tree lineage remains intact during DP scatter. |
| `AgenticTrainer` / `AgenticEnvTrainer` | variable-depth `List[Sample]` → concatenated turn `Part` | Barrier multi-turn tool use. The base variant scores terminal answers; the env variant consumes per-trajectory environment returns. |
| `AgenticPartialTrainer` / `AgenticEnvPartialTrainer` | freshest complete trajectory groups → concatenated turn `Part` | Colocated over-sample/commit/abort loop. `carry` is for Sample-resumable stateless tools; `drop` purges tails from stateful environments that restart episodes. |
| `AsyncAgenticTrainer` / `AsyncAgenticEnvTrainer` | buffered complete trajectory groups → concatenated turn `Part` | Disaggregated train/rollout slabs, resident agentic drive, weight-version staleness control, and the same explicit `carry`/`drop` tail policy. |

The async variants program against the driver-side async engines in
`unirl/rollout/engine/asynchronous.py`: `AsyncBatchRolloutEngine` (AR/diffusion — non-blocking
batched generations, launch-time train-version stamps, completion-order FIFO batches) and `AsyncAgenticRolloutEngine`
(partial/async agentic — trajectory drives, group assembly, completion-time stamps).
The trainers keep the policy: optimizer-update launch slots, reap-vs-launch order, quiesce points,
and tail carry/drop.

**Extending it:** a new domain is a new `<Domain>Trainer(BaseTrainer)` that builds its
remotes inside a `placement(...)` scope and implements `train_step` + `train`; the
matching `../train_<domain>.py` entrypoint composes the recipe and calls it.

## Checkpointing

Available for the single-backend trainers (including diffusion, AR, unified-model,
ReFL, async, and agentic variants) and for every trained side of `PETrainer`. A
single-backend checkpoint bundles model state (`save_mode=auto`: LoRA-only when
LoRA is active, otherwise full; `save_mode=full`: the whole model state;
`save_mode=adapter`: LoRA keys only), optimizer and scheduler state, the step
counters (`step`, `optimizer_step_count`), and the LoRA config (rank / alpha /
target_modules / exclude_modules — export tooling reads its scaling and module
selection from it).

The default backend `checkpoint_format=torch` gathers full state to distributed
rank 0 and writes `<save_dir>/checkpoint-<step>/checkpoint.pt`. Setting the
relevant backend's `fsdp_cfg.checkpoint_format=dcp` writes reshardable DCP
shards plus `metadata.pt` under the checkpoint directory; each rank reads and
writes its own shard. For example, PE configures the AR and diffusion backends
independently, while ReFL uses `policy.fsdp_cfg`. Load auto-detects either
on-disk format.

With `checkpoint_format=dcp`, `checkpoint_async=true` stages the snapshot and
flushes shards in the background. The next save/load drains the prior future,
and every trainer drains the final save before worker teardown.

PE checkpoints use one checkpoint directory per trained side:
`<save_dir>/checkpoint-<step>/diffusion/` and, unless `freeze_llm=true`,
`<save_dir>/checkpoint-<step>/ar/`. Each side uses its backend's selected format.
The driver-owned `trainer_state.json` remains at the common
`checkpoint-<step>/` level. A normal PE save writes every trained side at the
same rollout step; resume expects those sibling directories together.

Save and load are collectives. In the default torch format every rank
participates in gather/broadcast and distributed rank 0 writes the file; in DCP
format every rank writes and reads its shard.

Async and partial agentic checkpoints restore the trainable model, optimizer,
scheduler, and counters, but not runtime-only rollout state: buffers, in-flight
generations, carried trajectories, and environment episodes restart empty.

**Multi-node**: `save_dir` / `load_dir` must live on storage mounted on every
node — the same contract the recipes already place on `PRETRAINED_MODEL` and
data paths. A rank that cannot see the checkpoint fails fast on every rank at
load (instead of stranding the others in the broadcast until the NCCL timeout).

**Meta-init caveat**: full checkpoints in the default torch format reject
never-materialized parameters. Use `checkpoint_format=dcp` for meta-init bundles
such as HI3 80B; DCP requires trainable parameters to be materialized but may
drop frozen auxiliary parameters that remain on meta.

Driven by top-level config keys, read by the entrypoints and forwarded to
`train(...)`:

| Key | Default | Meaning |
| --- | --- | --- |
| `save_interval` | `0` | Save every N rollouts (and on the last); `0` disables saving. |
| `save_dir` | `./checkpoints` | Output folder for `checkpoint-<step>/`, resolved on the driver (with Hydra's legacy chdir the default lands in the run output dir). |
| `save_mode` | `auto` (`adapter` for ReFL; `full` for Async AR) | `auto` = LoRA-only when LoRA is active, otherwise full; `full` = whole model state; `adapter` = LoRA keys only (the frozen base reloads from the pretrained snapshot on resume). |
| `load_dir` | unset | A checkpoint dir to restore and resume from; unset trains fresh. |

Recipes may expose these keys already. When a recipe does not, append them with
Hydra's `+` syntax.
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
so EMA decay schedules continue) and resumes the loop from the saved step.
Synchronous Sample-based trainers continue `training_progress` and
driver-authored x_T scheduling, fast-forward a deterministically seeded data
stream, and force the restored weights into a freshly started rollout engine
when needed. Async AR/diffusion resume reads the backend optimizer count as the
train version, fast-forwards the deterministic input stream to the saved
rollout step, and syncs those restored weights into the fresh engine.
Partial-agentic resume can consume a different
input sequence when an earlier over-sampled drive required refills, and ReFL
does not currently fast-forward its data source.

The W&B run also continues: driver-written `trainer_state.json` at the
checkpoint root carries the run id and `train/` step axis. For PE this file is
in the common parent above the `ar/` and `diffusion/` side checkpoints.

### Export to Hugging Face format

The checkpoint directory is a raw training artifact (PEFT-injected names and
optimizer state), not a release artifact. The offline checkpoint toolset lives in
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

For PE, point the exporter at one side's directory, for example
`checkpoint-1000/diffusion` or `checkpoint-1000/ar`, rather than at the common
parent directory. The exporters auto-detect both the legacy torch-format
`checkpoint.pt` and complete DCP checkpoints (`.metadata` plus `metadata.pt`);
an incomplete asynchronous DCP directory is rejected rather than partially
exported.

## Multiple optimizer updates per rollout

`TrainStack` and `UnifiedModelTrainStack` implement
`stack.num_updates_per_batch`. A value of `N` partitions each worker's rollout
shard into `N` **disjoint** mini-batches and runs one optimizer step per
mini-batch; it is not `N` epochs over the full rollout batch. The pre-update
policy anchor is prepared once and remains frozen across all `N` steps, so
later updates can use the algorithm's ratio/clip trust region against the same
rollout policy.

The stack rejects `N > 1` unless every participating algorithm declares
`supports_multi_update`. The per-worker batch must also divide evenly into `N`
updates. The unified stack slices corresponding AR and image mini-updates and
applies each pair in one shared optimizer step. Multi-update results retain
per-update metrics for logging as well as a rollout-level aggregate.

## Evaluation cadence

`eval_interval=0` disables evaluation. Trainers with evaluation support run a
baseline before training, then evaluate after every `eval_interval` completed
rollouts. Image, PE, unified, and ReFL trainers label a resumed baseline with
the restored step; AR and Async AR currently log that baseline at step 0. When
an evaluation and checkpoint fall on the same step, evaluation runs first.

- `ARTrainer` evaluates the requested prompt set in bounded batches and reports
  mean reward, also exposed as avg@k accuracy for binary evaluators.
  `AsyncARTrainer` quiesces its resident engine first.
- `DiffusionTrainer`, `PETrainer`, and `UnifiedModelTrainer` report image
  reward; optional `eval_rewards` suites can
  score the same generated samples or their own prompt sets. PE scores only the
  diffusion/image frontier. `AsyncDiffusionTrainer` reaches an empty hard
  boundary, syncs the current train version when needed, and then scores
  the resident rollout engine without offloading it afterwards.
- Agentic evaluation is not implemented. Barrier and partial variants raise if
  evaluation is enabled; async variants currently force it off.

### Decoupling diffusion eval from the RL rollout

`DiffusionTrainer` (sync and async) evaluates with the training `sampling:`
block plus, in override order: `eval_eta` (default `0.0` — deterministic ODE;
`<= 0` also clears the SDE gate, `> 0` keeps the training gate and rejects a
step-count override), `eval_samples_per_prompt`, then the optional
`eval_sampling:` overlay, which accepts any plain `DiffusionSamplingParams`
field (object-valued fields such as `scheduler` / `sde_strategy` are rejected)
and inherits `sampling:` for every field it does not mention:

```yaml
sampling:                 # the RL rollout: cheap, stochastic
  num_inference_steps: 10
  guidance_scale: 1.0
  height: 512
  width: 512
  eta: 0.7
  samples_per_prompt: 8

eval_interval: 20
eval_samples_per_prompt: 1
eval_sampling:            # eval only; unset fields inherit `sampling`
  num_inference_steps: 28
  height: 1024
  width: 1024
logging:
  log_media: true         # also uploads the eval panel (below)
```

CFG has no eval knob of its own: leave `guidance_scale` unmentioned and eval
runs at the training guidance (a CFG-off run cannot silently evaluate with CFG
on); name it in `eval_sampling:` to decouple the two. BAGEL-family params
consume `cfg_text_scale`, and passing the inert `guidance_scale` there raises.
Unknown overlay fields raise, and the retired per-field knobs
(`eval_cfg_text_scale`, `eval_num_inference_steps`, ...) fail fast with a
migration hint. Dynamic-shift models re-derive μ from the eval
steps/resolution, so a decoupled eval stays on the model's official schedule;
there is no per-eval time-shift override.

With `logging.log_media: true`, each eval also uploads up to `media_max_items`
generations to `eval/generated_media` (`eval/<suite>/generated_media` for
own-set suites) on the `eval/step` axis, captioned with prompt and reward. The
eval set is served in a fixed order and eval x_T is keyed on prompt content, so
at the default `eval_eta: 0` every eval renders the same prompts from the same
noise — a like-for-like filmstrip in which only the policy differs. An SDE eval
(`eval_eta > 0`) draws per-step noise seeded from the eval step, so its panels
vary by noise too; the trainer warns at startup then, as it does for pipelines
without driver-authored x_T. `media_log_interval` does not apply
(`eval_interval` already paces the panel).

## Gotchas

- **Multi-update means disjoint optimizer mini-batches, not repeated full-batch
  epochs.** See [Multiple optimizer updates per rollout](#multiple-optimizer-updates-per-rollout)
  for the algorithm and divisibility constraints.
- **Agentic evaluation remains deferred.** Barrier and partial recipes must use
  `eval_interval=0`; async trainers ignore the recipe value and force it off.
- **`layout` only branches on `"separate"`** (`"colocate"` == `"colocated"`). The
  trainside direct-sampling engine cannot live on a `separate` slab — `_build_rollout`
  raises (it needs the pipeline as a local sibling).
- **`weight_sync` is built only when a `sync:` block is present** (dedicated engines);
  trainside sampling reads the live training weights and needs none (`self.weight_sync` stays `None`).
- **FSDP offload during `generate` is off by default** and force-gated off for trainside
  (it reuses the train model) and for DiffusionNFT (its EMA swap touches the backend around `generate`).
- **The bundle must be shared, not rebuilt** — the trainer injects one bundle into both
  pipeline and backend; a second `from_config` would silently desync replay. See [`../models/README.md`](../models/README.md).
