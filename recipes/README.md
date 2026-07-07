# Recipes

A **recipe** is a self-contained, runnable training program: one package per task,
holding the trainer, the entrypoint, and its configs together.

```
recipes/<task>/
  trainer.py     # the <Task>Trainer class + the @hydra.main main() entrypoint
  __main__.py    # 2-line shim so `python -m recipes.<task>` runs main()
  configs/       # this recipe's experiment configs (one self-contained YAML per run)
```

The library it composes lives in [`../unirl/`](../unirl/README.md) — models,
algorithms, rollout engines, reward, the train stack/backends, and the distributed
runtime. A recipe wires those components into one RL loop; the library never imports a
recipe. The shared driver base is `unirl.train.base_trainer.BaseTrainer`.

## Recipes & how to launch

Each recipe is launched with `python -m recipes.<task>`. Its **default config** is that
entrypoint's built-in `config_name` — a safe place to start. Configs are addressed
**relative to the recipe's `configs/`** (no task prefix).

| Recipe | Launch | Default config (start here) | Trains |
|---|---|---|---|
| [`diffusion/`](diffusion/) | `python -m recipes.diffusion` | `sd3/sd3_trainside` | Image / video diffusion (`sd3`, `qwen_image`, `flux2_klein`, `wan21`, `wan22`, `hunyuan_video`, `hunyuan_video15`, `ltx2`, `z_image`, `bagel`) |
| [`ar/`](ar/) | `python -m recipes.ar` | `qwen_vl_grpo_geo3k_mc_4x8` | Autoregressive — vision-language (`qwen_vl`) + text-only (`qwen3`) |
| [`async_ar/`](async_ar/) | `python -m recipes.async_ar` | `qwen3_grpo_4b_base_dapo_sglang_async` | Disaggregated async AR (train + rollout on disjoint slabs) |
| [`pe/`](pe/) | `python -m recipes.pe` | `pe_trainside_pickscore` | Prompt-enhancer (Qwen3 rewriter + SD3, PickScore/WISE reward) |
| [`refl/`](refl/) | `python -m recipes.refl` | `refl_sd3` | ReFL — direct differentiable-reward backprop (DRaFT-K) for SD3 |
| [`unified_model/`](unified_model/) | `python -m recipes.unified_model` | `hi3_vllmomni` | Unified AR + diffusion (HunyuanImage3) |

## Running a recipe

```bash
# 0. Compose-check first — verifies the config composes and every ${oc.env:...} resolves
python -m recipes.diffusion --config-name=sd3/sd3_trainside --cfg job --resolve

# 1. Single node (launcher lives in ../scripts). ENTRY selects the recipe (default: diffusion).
bash scripts/run_experiment_single_node.sh sd3/sd3_trainside
ENTRY=ar bash scripts/run_experiment_single_node.sh qwen_vl_grpo_geo3k_mc_4x8
ENTRY=pe bash scripts/run_experiment_single_node.sh pe_trainside_pickscore

# 2. Multi-node (taiji)
bash scripts/run_experiment_multinode_taiji.sh sd3/sd3_sglang_rollout_colocate

# 3. Or invoke a recipe directly, without the launchers
python -m recipes.diffusion --config-name=sd3/sd3_trainside num_devices=8
```

Pass cluster-local paths and W&B identity through env vars (`PRETRAINED_MODEL`,
`DATA_PATH`, `EVAL_DATA_PATH`, `REPORT_TO_WANDB`, `WANDB_PROJECT`, `WANDB_ENTITY`).
The mooncake-backed configs (`*_tq_mooncake`) need their metadata server up first — start
it on the head node with `bash scripts/mooncake_master.sh start` before launching.

## Config-name schema

A config filename is a fixed-order, `_`-joined chain of segments. Every segment except
`model` is optional and is **omitted when it is the default or does not apply** — so a
name carries only what distinguishes it from its siblings, and related configs sort
together.

```
<model>[_<task>][_<size>][_<algorithm>][_<engine>][_<adapter>][_<topology>]
```

| Segment | Position | Values (examples) | Omit when |
|---|---|---|---|
| `model` | required, first | `sd3`, `qwen_image`, `flux2_klein`, `wan21`, `wan22`, `hunyuan_video`, `hunyuan_video15`, `qwen_vl`, `qwen3`, `hi3` | never |
| `task` | after model | `t2v`, `i2v` | text-to-image (the implicit default) |
| `size` | after task | `4b`, `14b` | only one size in the family |
| `algorithm` | middle | `dancegrpo`, `mixgrpo`, `nft`, `flowdppo`, `grpo`, `drpo` | plain FlowGRPO (diffusion default); GRPO (AR default) |
| `engine` | after algorithm | `trainside`, `sglang`, `vllmomni` | — |
| `adapter` | after engine | `full`, `lora` | unambiguous from the rest |
| `topology` | last | placement `colocate`/`separate`; sync `nccl`/`tensor`/`ipc`; engine mode `rollout`/`replay` | single-slab colocate default |

Worked examples (each lives under its recipe's `configs/`):

| Config | Reads as |
|---|---|
| `sd3/sd3_trainside` | SD3 · trainside engine · (default FlowGRPO) |
| `sd3/sd3_nft_sglang` | SD3 · DiffusionNFT · SGLang engine |
| `qwen_image/qwen_image_dancegrpo` | Qwen-Image · DanceGRPO |
| `wan22/wan22_t2v_14b_dancegrpo` | WAN 2.2 · text-to-video · 14B · DanceGRPO |
| `qwen_vl_grpo_geo3k_mc_4x8` | Qwen-VL · GRPO · geo3k multiple-choice · 4 nodes × 8 GPUs |

Domain-specific trailing qualifiers extend the chain: `pe/` appends the reward
(`pe_sglang_full_pickscore`); `ar/` (vision-language) appends dataset + task
(`qwen_vl_grpo_geo3k_mc_4x8`) and the cluster shape `<N>x<G>` (nodes × GPUs).

## Architecture — what a trainer does

The trainer is the **driver-side conductor**: it places the rollout and train workers on
GPUs, builds the rollout engine / reward service / train stack / weight-sync handler, and
runs the optimizer loop over them. It owns **placement and sequencing** — and nothing
else: the loss math is `../unirl/algorithms`, the optimizer is `../unirl/train`, sampling
is `../unirl/rollout`, scoring is `../unirl/reward`. Keeping the wiring in one per-task
class is what lets the loop body stay ~10 lines and every module stay swappable by
`_target_`.

- **`BaseTrainer`** (`unirl.train.base_trainer`) owns the `DevicePool` (from the top-level
  cfg: `num_devices` / `transport_kind` + the optional TransferQueue bootstrap) and the
  optional rank-0 wandb logger. Subclasses get the configured pool for free.
- **Build phase** (`__init__`) builds the remote graph in a `placement(...)` scope,
  threading **one shared bundle** into both consumers —
  `bundle → pipeline(bundle) → backend(bundle) → reward → algorithm → stack` — then the
  rollout engine. Layout decides topology: `colocate` builds train + rollout as siblings
  on one slab; `separate` opens two disjoint slabs with a cross-slab weight-sync handshake.
- **The loop** (`train_step`) is the conductor sequence, one rollout per call:
  `wake_up` → (sync the fresh adapter, if due) → `rollout.generate(req)` →
  `reward.score_and_attach(track)` → `track.compute_advantages(...)` → drop the reward-only
  decoded media → `stack.train_track(track)`.

| Recipe(s) | Tracks | Train stack(s) | What's distinctive |
|---|---|---|---|
| `diffusion`, `ar`, `async_ar` | 1 | one `TrainStack` | the reference loop; diffusion adds colocate FSDP-offload + the DiffusionNFT EMA-adapter dance around `generate` |
| `pe` | 2 (`ar` + `diffusion`) | two (one per model) | composed *trainside* rollout; the image reward is `propagate_rewards`-credited up to the `ar` track |
| `unified_model` | 2 (`ar` + `image`) | one `UnifiedModelTrainStack` | one shared backbone (HunyuanImage3); both losses backward-accumulate into a single optimizer step |
| `refl` | 1 | direct policy backprop | differentiable reward (DRaFT-K), no rollout/advantage loop |

**Adding a new task:** a new `recipes/<task>/` package whose `trainer.py` defines a
`<Task>Trainer(BaseTrainer)` that builds its remotes inside a `placement(...)` scope and
implements `train_step` + `train`, plus the `@hydra.main` `main()`; a 2-line `__main__.py`;
and a `configs/` tree.

## Checkpointing

Available for the single-backend recipes (`diffusion`, `ar`, `unified_model`); `pe` (two
backends) is not wired. A checkpoint bundles the model state (`save_mode=auto`: LoRA-only
when LoRA is active, otherwise full; `adapter`: LoRA keys only), the optimizer/scheduler
state, the step counters, and the LoRA config — enough to resume. Each is written to
`<save_dir>/checkpoint-<step>/checkpoint.pt`. Driven by top-level config keys forwarded to
`train(...)`: `save_interval` (0 disables), `save_dir` (default `./checkpoints`),
`save_mode` (`auto`/`full`/`adapter`), `load_dir` (restore + resume). These are not in the
config YAMLs, so append them with Hydra's `+` syntax:

```bash
# 1. Train, saving LoRA-only checkpoints every 200 rollouts
bash scripts/run_experiment_single_node.sh sd3/sd3_trainside \
    num_rollouts=500 +save_interval=200 +save_dir=/ckpts/sd3_run +save_mode=adapter

# 2. Resume (num_rollouts is the TOTAL budget; the same save_dir is fine; wandb reattaches)
bash scripts/run_experiment_single_node.sh sd3/sd3_trainside \
    num_rollouts=1000 +load_dir=/ckpts/sd3_run/checkpoint-400 \
    +save_interval=200 +save_dir=/ckpts/sd3_run +save_mode=adapter

# 3a. Export a merged model (fold LoRA into base weights, write a save_pretrained folder)
python -m unirl.tools.export_full \
    --checkpoint /ckpts/sd3_run/checkpoint-1000 \
    --base stabilityai/stable-diffusion-3.5-medium --subfolder transformer \
    --output /ckpts/sd3_run/hf-1000

# 3b. Or export a PEFT adapter artifact
python -m unirl.tools.export_adapter \
    --checkpoint /ckpts/sd3_run/checkpoint-1000 \
    --base stabilityai/stable-diffusion-3.5-medium --output /ckpts/sd3_run/adapter-1000
```

`load_dir` restores model/optimizer/scheduler (plus the optimizer-step counter, so EMA
decay schedules continue) and resumes the loop from the saved step; the first rollout
force-syncs the restored adapter into the rollout engine, and the wandb run continues
(`trainer_state.json` beside `checkpoint.pt` carries the run id + step axis).

**Multi-node:** `save_dir`/`load_dir` must live on storage mounted on every node. **Meta-init
caveat:** full-state-dict checkpointing rejects never-materialized params, so
`unified_model` checkpointing currently works only for fully-materialized bundles (the hi3
80B meta-init recipe is not yet supported).

## Adding or editing a config

Every config **must start with `# @package _global_`** on line 1 — its recipe's `configs/`
is a subdirectory, so without it Hydra would nest the whole config under a key and the
entrypoint's top-level fields would be missing. Cluster-local paths, model mounts, output
dirs, and W&B identity stay out of the YAML — pass them as env vars / CLI overrides;
configs read them with `${oc.env:...}`.

1. Copy the closest existing config in the right recipe's `configs/`.
2. Keep line 1 as `# @package _global_`; name the file per the schema above.
3. Keep every choice in YAML, instantiated by `_target_`; use `${oc.env:...}` only for
   deployment-specific paths and logging identity.
4. Before opening a PR, run the checks that match the files you touched:

```bash
# Compose the config and print the resolved result
python -m recipes.<recipe> --config-name=<config> --cfg job --resolve

# Python syntax check + recipe _target_ guard + lint
python -m compileall -q unirl recipes
python scripts/check_recipe_targets.py
pre-commit run --all-files
```
