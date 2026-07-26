# Config

> **Where it fits:** cross-cutting — not a box in the loop. Every box (rollout,
> reward, train, sync) is built from a config dataclass whose field checks and
> precision aliases this module provides, and the recipe wiring them together
> has to pass this module's contracts first. Full map: [`../README.md`](../README.md).

## What it is

`unirl.config` is the small shared toolkit behind UniRL's flat-recipe config flow.
It owns **no** config dataclasses of its own — those live next to the components
that consume them — just the three things every recipe leans on:

| File | Role |
| --- | --- |
| `require.py` | `require(condition, message)` precondition helper. Stdlib-only. |
| `validation.py` | Shared **per-field** validators (precision aliases). |
| `contracts.py` | The **cross-component** contracts + `validate_recipe`, the driver-side gate. Stdlib-only. |

## Why it exists

A recipe is one flat YAML wired entirely by `_target_` dotpaths — there are **no**
Hydra config groups and no `defaults:` lists. That keeps every run reproducible
from a single file, but it also means Hydra type-checks nothing. This module is
where invariants get enforced instead, at two scopes:

- **Within a field** — each dataclass fails fast in `__post_init__` via
  `require(...)`, with a clear `ValueError`. Every precision field accepts the
  same aliases (`bf16`/`bfloat16`, `fp16`/…, `fp32`/…) through one shared
  `validate_precision_type`, so the rules and error message are identical
  everywhere.
- **Across sections** — the rules relating one section to another live in
  `contracts.py`, because no single dataclass can see both sides of them.

## How it works

A recipe is one flat YAML marked `# @package _global_`. Components are `_target_`
dotpaths, sub-configs are nested `_target_` blocks, shared values are `${...}`
interpolations. There is no ConfigStore and no registration step.

Instantiation is a **driver-routes / worker-materializes** split:

- `parse_hydra_cfg` (`../utils/hydra.py`) resolves only the *top-level* `_target_`
  on the driver and passes nested blocks through as plain dicts.
- `Worker._resolve_init_kwargs` (`../distributed/group/worker.py`) walks the tree on
  the worker and builds each nested `_target_` with `get_method(_target_)(**children)`
  — deliberately **not** `hydra.utils.instantiate`, so already-built objects pass
  through unchanged and each is constructed in the worker's own CUDA context.

### The cross-component gate

Every `unirl/train_*.py` opens with one line:

```python
validate_recipe(cfg, entrypoint="train_diffusion")
```

It runs on the driver before the trainer is constructed — before Ray, before the
engine's `_target_` is imported — so a contradictory recipe dies on the launching
process in about a second instead of somewhere inside a half-built cluster. Today
it enforces three contracts, all keyed off which rollout engine the recipe picked:

| Contract | Rejects |
| --- | --- |
| `validate_weight_sync_contract` | a `sync:` block on a direct-sampling engine; a dedicated engine with no `sync` handler; a handler whose transport the engine cannot receive (`IPCWeightSync` outside vllm-omni / composed); engine sections split across both sampling modes |
| `validate_rollout_layout` | `layout: separate` with a direct-sampling engine; a `layout` value that is neither `colocate` nor `separate` |
| `validate_offload_contract` | an explicit `enable_fsdp_offload: true` with a direct-sampling engine |

**Recipe shapes live in exactly one place.** Contracts never read a hard-coded
dotpath; they read `RecipeFacts.from_cfg(cfg)`, which absorbs the per-entrypoint
differences — `rollout` vs `ar_rollout` + `dit_rollout`, a single `sync` block vs
`train_pe`'s per-track map, and `train_sft`/`train_refl` having no rollout engine
at all (those simply have no engine contracts to check).

**Engines are identified by package, not class name.** `ENGINE_FAMILIES` maps
`unirl.rollout.engine.<family>` to whether the family samples in-process and
which weight-sync receive paths it implements, so a class rename cannot flip a
recipe into the wrong mode.

**Extending it:** a new component config is a plain `@dataclass` next to the
component (not here), with `require(...)` checks in `__post_init__`. A new
cross-component contract is a `validate_<thing>(cfg)` in `contracts.py` reading
`RecipeFacts`, added to `CONTRACTS` — which is what makes it run. If it needs a
recipe fact nobody has needed yet, add the field to `RecipeFacts` rather than
reaching into `cfg` from the contract.

## Verification

`scripts/check_recipe_contracts.py` (pre-commit hook `check-recipe-contracts`,
so it rides the lint-only CI alongside `check-recipe-targets`) asserts three
things on every run:

1. Every shipped recipe satisfies every contract.
2. Every combination the contracts claim to reject **is** rejected, and the valid
   shapes are not — so a contract that quietly became a no-op fails CI.
3. `ENGINE_FAMILIES` still matches the engine classes: `direct_sampling` is read
   back off each engine's `__init__` (does it take a `pipeline`? — the same
   duck-typed test the trainers use), and `weight_sync` off the receive methods
   the concrete class overrides. Adding an engine family without declaring it
   fails here rather than silently skipping its contracts.

All three run with `ast` + `yaml` only, no torch — which is why `contracts.py`
and `require.py` stay stdlib-only.

## Gotchas

- **The gate is per-entrypoint, one line.** A new `train_*.py` that forgets
  `validate_recipe(cfg, ...)` is simply ungated; nothing forces the call.
- **`# @package _global_` on line 1 is mandatory** — omit it and Hydra nests the
  whole recipe under a bucket key, so `cfg.batch_size` won't resolve.
- **Out-of-tree engines are not gated.** A `rollout._target_` outside
  `unirl.rollout.engine.*` has no known family, so the engine-dependent contracts
  log and skip rather than guess a mode for it.
- **Only an explicit `enable_fsdp_offload: true` is rejected.** When a recipe is
  silent the value comes from the entrypoint's default (`train_unified_model`
  defaults it to `True`, the rest to `False`), which is not a statement by the
  recipe author — and the trainers already force it off for direct sampling.
- **Contracts see the recipe, not the run.** Anything that depends on resolved
  runtime topology — `batch_size * samples_per_prompt` divisibility by the actual
  rollout/reward `dp_size`, for instance — cannot be checked here and stays in
  the trainer (`DiffusionTrainer.__init__`).
- **`validate_precision_type` validates but does not normalize** — it *returns* the
  canonical alias (`bf16`), but every call site invokes it as a bare statement and
  discards the result. So `model_precision: bfloat16` stays the raw string in `cfg`;
  downstream code must re-parse it with `parse_torch_dtype` itself.
