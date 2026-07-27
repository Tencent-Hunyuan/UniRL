# Code Architecture

`unirl/` is the framework package — the layered, composable core that turns one
Hydra recipe into a multi-GPU RL training run. This README is the **map of the
source tree**: what the layers are, how they depend on each other, and which
module README to read next. (For *what* UniRL is and how to launch it, see the
[top-level README](../README.md); for *how tensors move across GPUs at runtime*,
see [`distributed/README.md`](distributed/README.md).)

<div align="center">
  <img src="../assets/code-architecture-new.png" alt="UniRL code architecture: a thin entrypoint hands a recipe to the trainer (the driver), which places a Worker pool and wires each loop component (rollout / reward / algorithm / train / weight-sync) as a Remote from the recipe's _target_; the Remotes run the model code (models, sde) and rest on types (the shared contracts) and distributed (the Remote/Worker/transport substrate)" width="100%">
</div>

UniRL is a thin entrypoint that hands a recipe to the trainer (the **driver**),
which places a Worker pool and wires each **loop component** as a `Remote` from the
recipe's `_target_` — not by import. (Deliberately *not* "stage": in this repo a stage
is a trainable unit of a model pipeline — `DiffusionStage` / `ARStage`, see
[`models/README.md`](models/README.md) — never a position in the training loop.)

As source, the package falls into four groups:

- **Entrypoints** (`train_diffusion.py`, `train_ar.py`, `train_pe.py`,
  `train_unified_model.py`, plus the `train_agentic*.py` multi-turn variants)
  — each composes and validates a Hydra recipe, then hands off to its trainer.
- **Orchestration** (`trainer/`) — the per-domain `<Domain>Trainer` owns GPU
  placement, builds the rollout and train workers, and runs the
  rollout→reward→advantage→train loop.
- **Training loop** (`rollout/`, `reward/`, `algorithms/`, `train/`) — the four
  pluggable components of one rollout, plus what they share: `models/`
  (per-model bundles), `sde/` (step kernels / σ schedule), and `data/` (sources).
- **Foundation** (`distributed/`, `config/`, `types/`, `utils/`) — the
  cross-cutting infrastructure every layer rests on: the Ray
  worker/dispatch/transport runtime, config build-and-validate, the shared typed
  contracts, and helpers.

## Module Map

| Path | Responsibility |
|---|---|
| `train_*.py` | Hydra entrypoints for diffusion, AR, prompt enhancement, unified models, and synchronous/partial/asynchronous agentic workflows |
| `trainer/` | Training lifecycle (`base.py` plus domain and agentic trainers): owns placement, builds workers, and runs the rollout→reward→advantage→train loop |
| `config/` | `require` + `validate_*` cross-component validators over the flat Hydra recipe (instantiation itself is `_target_`-driven, not in this module) |
| `distributed/` | Ray worker base (`Remote`) + placement/dispatch (`group/`), tensor transport (`tensor/`), and weight sync (`weight_sync/`) |
| `rollout/` | Rollout engine contracts and implementations (`engine/`: trainside, sglang, sglang_diffusion, vllm_omni, composed) |
| `train/` | Train stack: `TrainStack`, FSDP backend, LoRA/DiffusionNFT/mirror injection, EMA shadow, optimizer/lr |
| `algorithms/` | Per-track loss algorithms (GRPO, DiffusionNFT, FlowDPPO, DRPO) |
| `models/` | Per-model bundles, pipelines, stages, conditions; text/vision/vae helpers |
| `reward/` | `RewardService` holding one backend — local scorers or the remote HTTP client |
| `sde/` | SDE step kernels, σ schedule/shift, initial-noise generation (the `NoiseRecipe` contract lives in `types/`) |
| [`types/`](types/README.md) | Shared typed contracts: `Sample` / `Part`, primitives, conditions, segments, rewards, sampling; includes the request/response migration guide |
| `data/` | Data source and dataset readers |
| `utils/` | Logging, dtype, media, timing, checkpoint, and misc helpers |

## Deployment modes

The rollout engine and optional `sync:` section define how GPUs are used:

| Mode | Layout | Sync |
|---|---|---|
| Train-side sampling | Training workers generate samples directly | Not used |
| Separate rollout | Rollout and training use different GPU pools | Required |
| Colocated rollout | Rollout and training share GPU bundles with offload/onload | Required |

See [`distributed/README.md`](distributed/README.md) for how each mode places
workers and moves rollout data between them.

## Runtime Data Flow

The layers above turn one recipe into a repeating loop. A single training step
flows through them like this:

<div align="center">
  <img src="../assets/pipeline-dataflow-new.png" alt="UniRL data flow: the prompt becomes a Sample lineage whose generated Part is filled with a segment and modality-keyed primitives; reward scoring attaches rewards, advantage computation annotates the Part, and training consumes its segment plus advantages" width="100%">
</div>

1. An entrypoint composes the chosen `examples/<domain>/<recipe>.yaml` and runs validators.
2. The `<Domain>Trainer` (e.g. `trainer/diffusion.py`) acquires a Ray `DevicePool` and builds the rollout and train workers.
3. The trainer builds a typed `Sample` lineage and dispatches it to the rollout engine.
4. The engine returns the lineage with generated `Part`s filled with conditions, segments, primitive maps, and media previews.
5. `RewardService.score_and_attach` attaches rewards; `Part.compute_advantages` z-scores them into advantages.
6. `TrainStack.train_track(...)` shards the generated `Part` across train workers and runs the mini-batch optimizer loop.
7. Each train worker owns a model `Bundle`, an `FSDPBackend`, and one loss algorithm.
8. Dedicated-rollout modes (separate / colocate) sync trainer weights back to the rollout workers.

Agentic workflows extend step 3 into a trajectory: the coordinator repeatedly
invokes a single-turn engine, executes tool or environment actions between turns,
and returns `list[Sample]`. The agentic trainers then assemble the trainable turns
before applying the same reward, advantage, and train-stack contracts.

## Deeper Module Docs

- `trainer/README.md`: the orchestration hub — how a `<Domain>Trainer` places workers and drives the loop.
- `types/README.md`: the `Sample` / `Part` contract and migration from the retired request/response API.
- `config/README.md`: flat-recipe config — `require`/precision validators, `_target_` instantiation, cross-component contracts.
- `rollout/README.md`: rollout modes, engines, and the `Sample` / `Part` generation flow.
- `rollout/loop/README.md`: agent-loop environments, tools, trajectories, and partial-resume behavior.
- `train/readme.md`: train stack, FSDP backend, injection, EMA shadow.
- `algorithms/README.md`: per-track loss algorithms.
- `reward/README.md`: reward backends and custom scorers.
- `models/README.md`: model bundle and per-model package contracts.
- `sde/README.md`: SDE kernels, σ schedule, initial noise.
- `distributed/README.md`: distributed runtime — workers, dispatch, placement, and the rollout→train data plane.
- `distributed/weight_sync/README.md`: trainer→rollout weight-sync backends.
