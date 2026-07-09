# Utils

> **Where it fits:** shared helpers used across the loop. Full map:
> [`../README.md`](../README.md).

`utils/` is a grab-bag — each module is a small helper documented by its own
docstring (`timing.py`, `wandb_logger.py`, `hydra.py`, `media.py`, `misc.py`,
`profiling.py`, …). Read the module you need directly. The one exception
documented here is **GPU memory monitoring**, because it spans two files
(`memory_utils.py` + `memory_monitor.py`) and three subsystems (sampling on the
workers, orchestration here, hooks in the trainer).

---

# GPU Memory Monitoring

verl-parity GPU memory observability. Answers one question fast: **when a run
OOMs (or creeps toward it), which phase and which tensor is to blame?** It wraps
every trainer's hand-off boundaries (`wake_up → weight_sync → generate → sleep →
reward → train`, plus checkpoint save); sampling runs on the workers (which hold
the GPUs), orchestration on the CUDA-less Ray driver.

**Opt-in (off by default), observation-only** (no cache clears, no syncs) — each
wrapped phase spends two blocking BROADCAST probes, so you turn it on when you
want it: `+logging.memory.enabled=true` (or `UNIRL_MEM_MONITOR=1`, or just
`UNIRL_MEMSNAP=1` which auto-enables it). Once on, every step it logs four peak
metrics to wandb on the `rollout/step` axis, **max-across-ranks** (OOM dies on
the worst rank):

| Metric | Meaning |
|---|---|
| `perf/max_memory_allocated_gb` | tensors actually in use (peak) — rising = leak |
| `perf/max_memory_reserved_gb` | PyTorch's reserved pool (peak); minus allocated = fragmentation |
| `perf/device_memory_used_gb` | whole-device (`mem_get_info`) — **sees the colocated SGLang process** the allocator can't |
| `perf/cpu_memory_used_gb` | process RSS (host-side leaks) |

## Three levels (matches the OOM triage flow)

| Level | Turn on with | Answers |
|---|---|---|
| **0 — curves** | `+logging.memory.enabled=true` (or `UNIRL_MEM_MONITOR=1`) | *Is there a problem? Growing?* |
| **1 — boundary logs** | above + `+logging.memory.log_boundaries=true` | *Which phase spikes?* |
| **2 — snapshots** | `UNIRL_MEMSNAP=1` (env; auto-enables the monitor) | *Which line of code allocated it?* |

> These keys aren't in the base config, so as Hydra CLI overrides they need the
> `+` prefix (`+logging.memory.enabled=true`) — a bare `logging.memory.enabled=true`
> fails config composition in struct mode. No `+` needed if you add the
> [`logging.memory`](#configuration) block to your recipe yaml instead. The
> `UNIRL_*` env vars need neither.

**Level 1** emits a `[mem]` line at each phase's entry/exit. `peak_alloc` is that
phase's own peak (counter reset on entry); `(rankN, min …)` shows the worst rank
and the spread (large spread = one rank doing more):

```
[mem] stage=train peak_alloc=39.80 (rank7, min 27.09) reserved=45.10 device_used=84.0 (GB)
```

**Level 2** records every allocation's call stack, dumps a pickle, and **logs a
ranked report automatically** — the top call sites holding live memory land in
the training log inline, no separate file, no manual step:

```bash
UNIRL_MEMSNAP=1 UNIRL_MEMSNAP_STEPS="2:4" python -m unirl.train_... <recipe>
```
```
memory: snapshot step2 (rank 0)
live GPU allocations: 15.43 GB across 4180 blocks
top 15 call sites by live bytes:
  14.07 GB  x1847   transformers/modeling_utils.py:3722:to      ← model weights
   0.52 GB  x1274   unirl/train/backend/fsdp/wrap.py:123:fsdp_wrap
memory: snapshot dumped to outputs/memsnap/memsnap_step2_rank0.pickle
```

Each line is `<live GB> x<count> <file:line:func>`. **To find a leak**, dump two
steps (`"2:8"`) and compare — the call site whose GB grew is the leak. The pickle
is kept for a memory_viz deep-dive or a later re-run of `summarize_snapshot(path)`
(it reads the plain-dict pickle **without torch/CUDA**, anywhere). Recording hooks
every allocation (real overhead, big dumps), so it's off by default, rank-0 only,
step-ranged.

## Configuration

```yaml
logging:
  memory:
    enabled: false         # step curves + phase probes (default OFF — opt-in)
    log_boundaries: false  # the [mem] lines (default off)
    empty_cache_at: []     # phases to run aggressive_empty_cache at — the ONE
                           # behaviour-changing knob (default empty = never).
                           # e.g. ["sleep"] to test a fragmentation-OOM fix.
```

Env vars (same family as `UNIRL_PROFILE_*`): `UNIRL_MEM_MONITOR=0/1` overrides
`enabled` without editing yaml; `UNIRL_MEMSNAP=1` + `UNIRL_MEMSNAP_STEPS="2:4"` +
`UNIRL_MEMSNAP_DIR` / `UNIRL_MEMSNAP_RANKS` (default `0`) / `_MAX_ENTRIES` drive
snapshots.

Off entirely: `enabled=false` or `UNIRL_MEM_MONITOR=0` installs **nothing** —
byte-identical to an unpatched run.

## Reading the curves

- `max_allocated` flat = healthy; **rising every step = leak** → Level 1 to find
  the phase, Level 2 for the line.
- OOM but `allocated` flat/comfortable = not a true OOM → suspect **fragmentation**
  (`reserved − allocated` large) or the colocated engine (`device_used ≫ reserved`).
- `cpu_memory_used` climbing = host-side leak (data loading/caching, not the model).
- One rank ≫ the rest (Level-1 `min`/`max` spread) = imbalance; investigate it.
- **Check the y-axis range first** — wandb auto-zooms a flat 40 MB wobble into a mountain.
