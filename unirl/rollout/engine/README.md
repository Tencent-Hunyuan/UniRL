# Rollout engines

> **Where it fits:** the engine layer of the *rollout* step. In: a request
> `Sample`. Out: a filled `Sample`, or a trajectory `list[Sample]`.
> Full map: [`../README.md`](../README.md).

*Two halves of one design: `synchronous.py` holds the worker-side contracts every
per-backend subpackage implements; `asynchronous.py` holds the driver-side engines
the async trainers program against.*

## What it is

The engine ABCs, the driver-side async wrappers, and the two cross-engine helpers
(`ports.py`, `sigma_verify.py`). The per-backend engines live in the subpackages:
`trainside`, `sglang`, `sglang_diffusion`, `vllm_omni`, `fastvideo`, `composed`,
and the `agentic` coordinator.

## Why it exists

Every backend has to be swappable without the training loop noticing, and the
trainer replays the rollout — so any numerical drift silently pushes the GRPO ratio
off 1.0. These modules are where the two contracts that prevent that are written
down: what a worker-side engine must implement, and what the driver may assume.

## How it works

- **Worker side** (`synchronous.py`). `BaseRolloutEngine` is the broad ABC
  including coordinator engines; `SyncRolloutEngine` is the `Sample` → `Sample`
  refinement the per-backend subpackages implement. Engines complete construction
  in `__init__` — there is no separate initialize step.
- **Driver side** (`asynchronous.py`). Single-threaded, lock-free, ray-free;
  non-blocking dispatch is `Handle.launch_nowait`. Mechanisms are policy-free —
  `VersionedBuffer` (payload-agnostic freshness/staleness) and `InflightPool`
  (non-blocking pool of distributed `generate` calls); launch ceilings, reap/launch
  ordering and step loops live in the trainers. Both engines share one consumer
  surface: `poll` / `drain_freshest` / `pop_evicted` / `quiesce` plus an
  engine-owned `weight_version`.
- **σ round-trip** (`sigma_verify.py`). The adapter pins the gen Part's sigmas via
  `ensure_sample_sigmas`, forwards them, and asserts the worker echoed back the
  exact schedule it sent.
- **Ports** (`ports.py`). A `ReservedPorts` subclass declares one `int` field per
  port its subprocess needs — declaration order is reservation order. The engine
  reserves at the last responsible moment, in its ctor on its own node, right
  before the spawn. Bind-to-zero de-synchronizes colocated engines with no
  builder-side `base + rank * stride` math.

**Extending it:** subclass `SyncRolloutEngine`, implement synchronous generation
over the whole-`Sample` contract, and dispatch `generate` with `DP_SCATTER`. A
dedicated engine also implements its weight-receive method and a matching `sync:`
handler in `../../distributed/weight_sync`.

## Gotchas

- **`__init__.py` must import nothing.** The driver-side `asynchronous` module has
  to stay ray/torch-free to import, so consumers import the halves directly.
- **Port reservation is a hint, not a contract** — the sockets are closed
  immediately after binding so the subprocess can bind them itself, which leaves
  the usual bind-to-zero TOCTOU gap. Accepted deliberately.
- **Never hardcode the σ scale.** `sigma_verify` detects it dynamically; dividing
  by 1000 breaks on any model whose `num_train_timesteps` is not 1000.
- **An engine that will serve as an agentic inner must make `generate` safe for
  concurrent callers** — the agentic coordinator drives one trajectory per drain
  thread.
