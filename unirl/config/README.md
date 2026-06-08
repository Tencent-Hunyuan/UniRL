# Config

> **Where it fits:** cross-cutting — not a box in the loop. Every box (rollout,
> reward, train, sync) is built from a config dataclass whose field checks and
> precision aliases this module provides. Full map: [`../README.md`](../README.md).

## What it is

`unirl.config` is the small shared toolkit behind UniRL's flat-recipe config flow.
It owns **no** config dataclasses of its own — those live next to the components
that consume them — just the two things every dataclass leans on: a
`require(condition, message)` precondition helper (`require.py`), and a
`validation.py` of shared field validators plus cross-component contract checks.

## Why it exists

A recipe is one flat YAML wired entirely by `_target_` dotpaths — there are **no**
Hydra config groups and no `defaults:` lists. That keeps every run reproducible
from a single file, but it also means Hydra type-checks nothing. This module is
where invariants get enforced instead:

- Each dataclass fails fast in `__post_init__` via `require(...)`, with a clear
  `ValueError`.
- Every precision field accepts the same aliases (`bf16`/`bfloat16`, `fp16`/…,
  `fp32`/…) through one shared `validate_precision_type`, so the rules and error
  message are identical everywhere.
- The cross-component contracts that span multiple recipe sections (engine ↔ sync,
  offload, layout, batch geometry) are written down here too — though today they
  are documented intent, not an automatic gate (see Gotchas).

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

Validation runs in two layers:

- **Per-dataclass `__post_init__`** — local field invariants via `require(...)`
  and `validate_precision_type(...)`. **This is the only layer that runs today**
  (it fires at actor-build time).
- **Cross-component validators** (`validate_weight_sync_contract`,
  `validate_rollout_layout`, `validate_offload_contract`, …) take the whole `cfg`;
  most key off `is_direct_sampling(cfg)` (true when the engine `_target_` ends in
  `TrainsideRolloutEngine`). They encode the contracts but no live entrypoint calls
  them yet.

**Extending it:** a new component config is a plain `@dataclass` next to the
component (not here), with `require(...)` checks in `__post_init__`. A new
cross-component validator is a `validate_<thing>(cfg)` in `validation.py` that
branches on `is_direct_sampling(cfg)` — and that you must also wire in driver-side
for it to actually gate.

## Gotchas

- **The cross-component validators don't run today.** Not one `validate_*(cfg)` has a
  live call site (only `is_direct_sampling` is consumed); two aren't even re-exported
  from `config/__init__.py`. So e.g. `direct_sampling` + offload, or a `sync:` block on
  a trainside engine, is *not* rejected here — the only guard that fires is the trainer's
  own inline `layout=separate requires a dedicated engine` check. Don't assume a bad
  recipe is caught for you.
- **`# @package _global_` on line 1 is mandatory** — omit it and Hydra nests the
  whole recipe under a bucket key, so `cfg.batch_size` won't resolve.
- **`is_direct_sampling` is a `_target_` *suffix* match** — renaming or relocating
  the trainside engine class silently flips a run into dedicated mode.
- **`validate_precision_type` validates but does not normalize** — it *returns* the
  canonical alias (`bf16`), but every call site invokes it as a bare statement and
  discards the result. So `model_precision: bfloat16` stays the raw string in `cfg`;
  downstream code must re-parse it with `parse_torch_dtype` itself.
