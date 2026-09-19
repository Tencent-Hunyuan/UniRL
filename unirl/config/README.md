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

A recipe is a flat YAML wired entirely by `_target_` dotpaths. Most recipes are
self-contained; a small number of derived recipes use local, string-only
`defaults:` entries plus `_self_` to override a sibling recipe. There are no Hydra
config groups, and Hydra type-checks neither form. This module is where invariants
get enforced instead, at two scopes:

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

- `parse_hydra_cfg` (`../trainer/hydra.py`) resolves only the *top-level* `_target_`
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
it enforces three contracts across the entrypoint, sampling, engine, sync, and
placement choices:

| Contract | Rejects |
| --- | --- |
| `validate_sampling_contract` | AR/diffusion sampling-parameter types routed to the wrong entrypoint |
| `validate_weight_sync_contract` | wrong engine family for an entrypoint; missing or extra sync/anchor fields; invalid anchor values; local/remote handler topology mismatches; unsupported receive/verification methods; incomplete or misrouted PE track maps; unsupported entrypoint-specific sync modes |
| `validate_rollout_layout` | invalid layout values; separate direct sampling; layout fields on entrypoints that do not consume them |

**Recipe shapes live in exactly one place.** Contracts never read a hard-coded
dotpath; they read `RecipeFacts.from_cfg(cfg)`, which absorbs the per-entrypoint
differences — `rollout` vs `ar_rollout` + `dit_rollout`, single vs modality-keyed
sampling, a single `sync` block vs `train_pe`'s per-track map, nested
composed/agentic child engines, and `train_sft` having no rollout engine at all.

**Engines are identified by package, not class name.** `ENGINE_FAMILIES` maps
`unirl.rollout.engine.<family>` to its valid entrypoints, whether it samples
in-process, and the complete weight-sync protocol methods it implements, so a
class rename cannot flip a recipe into the wrong mode.

**Extending it:** a new component config is a plain `@dataclass` next to the
component (not here), with `require(...)` checks in `__post_init__`. A new
cross-component contract is a `validate_<thing>(cfg)` in `contracts.py` reading
`RecipeFacts`, added to `CONTRACTS` — which is what makes it run. If it needs a
recipe fact nobody has needed yet, add the field to `RecipeFacts` rather than
reaching into `cfg` from the contract.

## Verification

`lint/check_recipe_contracts.py` (pre-commit hook `check-recipe-contracts`,
so it rides the lint-only CI alongside `check-recipe-targets`) asserts five
things on every run:

1. Every runnable recipe under `examples/` satisfies every contract. Two
   diffusion recipes currently stored under `examples/unified_model/` carry
   explicit `train_diffusion` ownership overrides.
2. Every combination the contracts claim to reject **is** rejected, and the valid
   shapes are not — so a contract that quietly became a no-op fails CI.
3. `ENGINE_FAMILIES` still matches the engine classes: `direct_sampling` is read
   back off each engine's `__init__` (does it take a `pipeline`? — the same
   duck-typed test the trainers use), and `capabilities` off the protocol methods
   the concrete class overrides. Adding an engine family without declaring it
   fails here rather than silently skipping its contracts.
4. Sync-handler local/remote metadata still matches whether its constructor owns
   a local `rollout` sibling.
5. Every `unirl/train_*.py` calls `validate_recipe` first, with its own entrypoint
   name, so adding an entrypoint cannot silently bypass the gate.

All five run with `ast` + `yaml` only, no torch — which is why `contracts.py`
and `require.py` stay stdlib-only.

## Gotchas

- **`# @package _global_` on line 1 is mandatory** — omit it and Hydra nests the
  whole recipe under a bucket key, so `cfg.batch_size` won't resolve.
- **Out-of-tree engines are not gated.** A `rollout._target_` outside
  `unirl.rollout.engine.*` has no known family, so the engine-dependent contracts
  log and skip rather than guess a mode for it.
- **Out-of-tree sampling classes are not guessed.** Recognized AR/diffusion
  sampling class names must match their entrypoint or track; an unrecognized
  target is left to runtime type validation.
- **Static lint defers unresolved `${...}` values.** The runtime gate sees the
  Hydra-resolved value and validates it normally; interpolations must therefore
  resolve to the same type the owning component expects.
- **Contracts see the recipe, not the run.** Anything that depends on resolved
  runtime topology — `batch_size * samples_per_prompt` divisibility by the actual
  rollout/reward `dp_size`, for instance — cannot be checked here and stays in
  the trainer (`DiffusionTrainer.__init__`).
- **`validate_precision_type` validates but does not normalize** — it *returns* the
  canonical alias (`bf16`), but every call site invokes it as a bare statement and
  discards the result. So `model_precision: bfloat16` stays the raw string in `cfg`;
  downstream code must re-parse it with `parse_torch_dtype` itself.
