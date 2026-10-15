# Config

> **Where it fits:** cross-cutting — not a box in the loop. Every box (rollout,
> reward, train, sync) is built from a config dataclass whose field checks and
> precision aliases this module provides. Full map: [`../README.md`](../README.md).

## What it is

`unirl.config` is the small shared toolkit behind UniRL's flat-recipe config flow.
It owns **no** config dataclasses of its own — those live next to the components
that consume them — just the two things every dataclass leans on: a
`require(condition, message)` precondition helper (`require.py`), and a
`validation.py` of shared field validators (`validate_precision_type`) plus the
colocated SGLang sleep check (`validate_memory_saver_contract`).

## Why it exists

A recipe is one flat YAML wired entirely by `_target_` dotpaths. Most recipes are
self-contained; a small number of derived recipes use local, string-only
`defaults:` entries plus `_self_` to override a sibling recipe. There are no Hydra
config groups, and Hydra type-checks neither form. Component configs and trainers
enforce invariants directly instead:

- Each dataclass fails fast in `__post_init__` via `require(...)`, with a clear
  `ValueError`.
- Every precision field accepts the same aliases (`bf16`/`bfloat16`, `fp16`/…,
  `fp32`/…) through one shared `validate_precision_type`, so the rules and error
  message are identical everywhere.
- Cross-component and topology preconditions are checked by the owning trainer or
  weight-sync handler before spawning actors.

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

**Extending it:** a new component config is a plain `@dataclass` next to the
component (not here), with `require(...)` checks in `__post_init__`.

## Gotchas

- **`# @package _global_` on line 1 is mandatory** — omit it and Hydra nests the
  whole recipe under a bucket key, so `cfg.batch_size` won't resolve.
- **`validate_precision_type` validates but does not normalize** — it *returns* the
  canonical alias (`bf16`), but every call site invokes it as a bare statement and
  discards the result. So `model_precision: bfloat16` stays the raw string in `cfg`;
  downstream code must re-parse it with `parse_torch_dtype` itself.
