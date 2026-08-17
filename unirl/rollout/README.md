# Rollout

> **Where it fits:** the *rollout* step of the loop —
> **rollout** → reward → advantage → train → sync. In: a request `Sample` from
> the trainer. Out: a filled `Sample`. Full map: [`../README.md`](../README.md).

<div align="center">
  <img src="../../assets/rollout-engines-new.png" alt="UniRL rollout engines selected by _target_ across direct, separate, and colocated deployment modes" width="100%">
</div>

*Every rollout engine fills one `Sample`. Agentic scheduling and group assembly live
in the driver-side rollout manager.*

## What it is

`unirl.rollout` owns the rollout engines — the box that fills a typed `Sample` by
running a model pipeline and the SDE step kernels. The agentic engine runs the
model/environment turns for one trajectory. It does not compute reward or loss.

## Why it exists

The rollout can come from two unrelated codebases — the in-process training
`Pipeline`, or the SGLang fork sampling in its own subprocess. For on-policy RL they
must walk a *numerically identical* trajectory, because the trainer replays the
rollout to recompute log-probs and any drift silently pushes the GRPO ratio off 1.0.
So this module is a **verification boundary**, not just a backend-hiding shim: it
pins one σ schedule on the generated `Part`'s sampling params and verifies what the
backend used (`engine/sigma_verify.py`). Each engine adapts its backend wire format
into canonical `Part` fields, so a dedicated server swaps in for the training model
without the loop noticing, and a mismatch crashes loudly instead of training on a
wrong objective.

## How it works

- **One generation interface.** `BaseRolloutEngine` (`engine/base.py`)
  is a `Remote` whose concrete engines implement synchronous `generate(sample)`;
  each returns one `Sample`. Batch engines dispatch `generate` with `DP_SCATTER`.
  Agentic `generate` is undecorated because the driver manager addresses one
  engine slot per trajectory; Ray actor concurrency lets the inner backend batch
  concurrent calls.
- **The typed boundary** (`../types/`). A `Sample` is an ordered chain of `Part`s.
  Each Part carries lineage ids, a raw `primitive`, an encoded `segment`, replay
  conditions, sampling params (including the σ schedule), and optional decoded
  media. Single-stage flows fill one generated Part; composed PE fills its chained
  AR and diffusion Parts.
- **The engines.** `trainside` (in-process — the train actor's pipeline *is* the
  sampler), `sglang_diffusion` (dedicated diffusion), `sglang` (dedicated AR), `vllm_omni`
  (dedicated; HI3 / SD3 / HunyuanVideo), and `composed` (chains an AR child + a
  diffusion child for prompt enhancement) are the five single-turn engines.
  `agentic` wraps one of them with an environment to produce multi-turn
  trajectories. Each diffusion engine consumes the Part's pinned sigmas verbatim,
  and dedicated engines regenerate `x_T` from the recipe, so two engines start a
  rollout from the same noise. `forward_batch_size` bounds peak memory by slicing
  the Sample and concatenating the results.
- **Deployment modes:** *direct sampling* — the trainside engine, no `sync:`, the
  ratio is 1 on the first update; *separate* — a dedicated engine on its own GPUs
  plus a `sync:` block; *colocate* — a dedicated engine sharing GPUs with train,
  plus offload/onload and `sync:`.
- **Driver-side scheduling.** One `manager.RolloutManager` serves batch and
  agentic trainers. Its progress thread dispatches bounded work and observes
  readiness; trainer-thread collection resolves results, assembles agentic
  siblings, and applies the configured filter. Async batch trainers own training
  progress and publication cadence; the manager owns the published rollout version
  and preserves completed generations as FIFO batch chunks. `AgenticTrainer` collects
  one complete barrier batch per step. Async AR refills the same manager immediately
  after collection, so its existing progress thread overlaps the next generation
  with scoring and training. Publication and durable boundaries still quiesce that
  single manager before proceeding.

**Extending it:** a new single-turn engine adds `engine/<name>/config.py` (a
`BaseEngineConfig` whose `make_engine(**deps)` lazily imports and builds it) and
`engine/<name>/engine.py` (subclass `BaseRolloutEngine`, implement
synchronous generation over the whole-`Sample` contract — thread-safe for
concurrent callers if it should serve as an agentic inner, else serialized
internally — and dispatch `generate` with `DP_SCATTER`). A dedicated engine also
implements its weight-receive method and a matching `sync:` handler in
`../distributed/weight_sync`.

## Gotchas

- **Never recompute σ inside an engine** — the generated Part's pinned sigmas are
  the single source of truth; `engine/sigma_verify.py` checks the backend echo (it
  guards the GRPO log-prob ratio).
- **Batch `generate` must dispatch `DP_SCATTER`.** Agentic is the intentional
  exception: its undecorated method is reached through one `Handle.slot(...)`.
- **Direct sampling forbids a `sync:` block; dedicated requires one.** The trainside
  engine also can't live on a `layout: separate` slab — `_build_rollout` raises.
- **Quiesce before weight sync / eval / checkpoint on async paths** —
  `RolloutManager.quiesce()` pauses dispatch, drains batch work, and cooperatively
  suspends agentic trajectories at turn boundaries. `sync_weights()` rejects queued
  or in-flight work, pushes weights, and publishes the optimizer-update
  `output_version`; immutable buffered groups keep their original provenance and
  remain subject to the configured filter. Eval/checkpoint admission still requires
  an empty manager. Launch and scoring order remains trainer policy.
- **A resolve/route failure poisons the `RolloutManager`** — samples may already be
  lost, so every later call (including `empty` / `counts`) re-raises the original
  error rather than reporting clean state; only `close()` stays safe.
- **Reward/advantage methods are not engine code** — `Part.compute_advantages` and
  `Sample.propagate_rewards` are called by the trainer after scoring. An engine
  fills generation fields such as `segment`, `conditions`, `primitive`, and
  `media_preview`; rewards arrive later from `RewardService`.
