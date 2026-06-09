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

## Gotchas

- **The reference loop is intentionally minimal** — `num_updates_per_batch` multi-epoch
  replay, **checkpoint cadence, and eval cadence are deferred**. `FSDPBackend.save()/load()`
  exist as primitives, but the loop never schedules them; there is no resume entrypoint yet.
- **`layout` only branches on `"separate"`** (`"colocate"` == `"colocated"`). The
  trainside direct-sampling engine cannot live on a `separate` slab — `_build_rollout`
  raises (it needs the pipeline as a local sibling).
- **`weight_sync` is built only when a `sync:` block is present** (dedicated engines);
  trainside sampling reads the live training weights and needs none (`self.weight_sync` stays `None`).
- **FSDP offload during `generate` is off by default** and force-gated off for trainside
  (it reuses the train model) and for DiffusionNFT (its EMA swap touches the backend around `generate`).
- **The bundle must be shared, not rebuilt** — the trainer injects one bundle into both
  pipeline and backend; a second `from_config` would silently desync replay. See [`../models/README.md`](../models/README.md).
