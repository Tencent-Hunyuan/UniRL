# Rollout

> **Where it fits:** the *rollout* step of the loop —
> **rollout** → reward → advantage → train → sync. In: a request `Sample` from
> the trainer. Out: a filled `Sample`, or a trajectory `list[Sample]` from the
> agentic coordinator. Full map: [`../README.md`](../README.md).

<div align="center">
  <img src="../../assets/rollout-engines-new.png" alt="UniRL rollout: five single-turn Sample engines plus the agentic trajectory coordinator, selected by _target_, across direct, separate, and colocated deployment modes" width="100%">
</div>

*Six engines share one broad ABC: five single-turn engines dispatch
`generate(sample) → Sample` with `DP_SCATTER`; the agentic coordinator dispatches
`generate(sample) → list[Sample]` with rank-zero broadcast.*

## What it is

`unirl.rollout` owns the rollout engines — the box that fills a typed `Sample` by
running a model pipeline and the SDE step kernels. Single-turn engines return that
filled `Sample`; the agentic engine coordinates repeated model/environment turns
and returns a trajectory list. It does not compute reward or loss.

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

- **One synchronous generation interface.** `BaseRolloutEngine` (`engine/base.py`)
  is a `Remote` whose concrete engines implement synchronous `generate(sample)`;
  each keeps its native batching/runtime path. Single-turn engines return one
  `Sample` and dispatch `generate` with `DP_SCATTER`; the agentic coordinator
  returns a trajectory list with rank-zero broadcast dispatch. Concurrency is
  threads, not asyncio: the agentic engine drives one trajectory per drain
  thread, so an engine meant to serve as its inner must make `generate` safe for
  concurrent callers (the SGLang backends keep concurrent in-flight requests
  batching together on the runtime; an event loop survives only inside the
  native backend, where the in-process SRT runtime requires one).
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

**Extending it:** a new single-turn engine adds `engine/<name>/config.py` (a
`BaseEngineConfig` whose `make_engine(**deps)` lazily imports and builds it) and
`engine/<name>/engine.py` (subclass `BaseSingleTurnRolloutEngine`, implement
synchronous generation over the whole-`Sample` contract — thread-safe for
concurrent callers if it should serve as an agentic inner, else serialized
internally — and dispatch `generate` with `DP_SCATTER`). A dedicated engine also
implements its weight-receive method and a matching `sync:` handler in
`../distributed/weight_sync`.

## Gotchas

- **Never recompute σ inside an engine** — the generated Part's pinned sigmas are
  the single source of truth; `engine/sigma_verify.py` checks the backend echo (it
  guards the GRPO log-prob ratio).
- **Single-turn `generate` must dispatch `DP_SCATTER`** — broadcast would return a
  list of per-worker Samples and break single-Sample consumption. Agentic is the
  intentional exception: `BROADCAST + RANK_ZERO` returns its trajectory list.
- **Direct sampling forbids a `sync:` block; dedicated requires one.** The trainside
  engine also can't live on a `layout: separate` slab — `_build_rollout` raises.
- **Reward/advantage methods are not engine code** — `Part.compute_advantages` and
  `Sample.propagate_rewards` are called by the trainer after scoring. An engine
  fills generation fields such as `segment`, `conditions`, `primitive`, and
  `media_preview`; rewards arrive later from `RewardService`.
