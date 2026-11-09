# Utils

> **Where it fits:** shared helpers used across the loop. Full map:
> [`../README.md`](../README.md).

`utils/` holds narrowly named, domain-agnostic supporting mechanisms. It is not
the default home for every reusable function: model preprocessing stays with
its model, dataset conversion lives under `datasets/<dataset>/` (repo root),
scheduler/SDE policy under `sde/`, config resolution under `config/`, trainer
composition under `trainer/`, and backend-specific helpers stay with that
backend. A helper belongs here only when its semantics are stable and it has
several independent owners across the tree.

Each module (`dtypes.py`, `media.py`, `metrics.py`, `video.py`,
`wandb_logger.py`, `profiling.py`, …) is named for what it does and carries a
one-line docstring — read the one you need directly; it is short enough. The
cross-module features documented here are **stage metrics** and **GPU memory
monitoring**: workers sample them, the driver aggregates them, and the logger
publishes them.

The one non-obvious helper is `profiling.py`: `UNIRL_PROFILE` is a single switch
whose *value* names the region to record.

---

# Stage Metrics

`stage_metrics.py` records H3 denoising and decode, `RewardService` scoring,
FlashGRPO policy replay and backward, reference replay, and `TrainStack` optimizer
steps. `DiffusionTrainer` drains the colocated workers once after each training
track via `TrainStack.collect_stage_metrics`, then uses the existing `log_perf`
wandb path. The same collection carries row-weighted KL sums/counts and parameter
squared sums; the driver merges their global values into `TrainStepResult.metrics`
for the existing `train/` logging path. Evaluation discards its counters separately.

For each `stage` in `denoise`, `decode`, `score`, `policy_forward`,
`kl_ref_forward`, `backward`, `optimizer_step`:

| Key | Reduction |
|---|---|
| `perf/stage/<stage>_s` | sum calls per worker, then mean across workers |
| `perf/stage/<stage>_s_max` | sum calls per worker, then maximum across workers |
| `perf/mem/<stage>_peak_alloc_gb` | maximum allocated GiB across calls and workers |
| `perf/mem/<stage>_peak_reserved_gb` | maximum reserved GiB across calls and workers |
| `train/lora_norm` | square root of the global sum of squares of all live trainable parameters |
| `train/ema_live_distance` | square root of the global live-minus-shadow squared sum; absent without EMA |
| `train/kl_video`, `train/kl_audio` | per-stream KL, row-weighted across all micros and ranks |
| `train/kl_mean_shift_video`, `train/kl_mean_shift_audio` | row-weighted mean of each row's fp32 mean-shift L2 norm |
| `train/kl_ref_mean_global` | row-weighted global mean of the element-count-weighted joint KL |

## Gotchas

- **Stage timing synchronizes at both boundaries.** It adds no model calls, but
  reduces asynchronous overlap; it is not a zero-overhead profiler. With `G`
  pipeline calls, `S` score calls, and `M` training micros per rank, one optimizer
  update adds `4G + 2S + 4M + 2` synchronizations, plus `2M` when `beta>0`.
  `optimizer_step` includes clipping, scheduler, EMA updates and parameter metrics.
- **Parameter metrics sample after a successful optimizer step.**
  `BaseFSDP2Backend.parameter_squared_sums` reads `local_view` of live trainable
  parameters and `ema.shadow.iter_pairs()`, reducing each tensor in fp32 and
  accumulating in fp64. It queues one pinned CPU copy of one or two fp64 scalars;
  the enclosing optimizer stage's existing end synchronization must finish before
  `record_parameter_sums` reads them. No new collective or loss-buffer element is
  added. fp32 tensor reductions and EMA updates can differ from an fp64 oracle.
- **Parameter norms assume full FSDP, not HSDP.** Plain replicated Tensor parameters
  contribute only on process-group rank 0. H3's `transformer_blocks` LoRA targets
  are all inside fully-sharded blocks; `root_wrap=false` rejects stray trainable
  tensors on multiple ranks. DTensor replicate dimensions, including HSDP's
  replicate dimension, are not deduplicated and overcount the squared sums.
- **Stage peaks are device-wide, not allocation deltas.** A worker owns one CUDA
  device. Concurrent score/generate stages include each other's live allocations;
  their times are not additive wall time. The recorder folds every active interval
  before resetting the allocator, preserving peaks across overlapping stages.
  `memory_peaks` also preserves the outer `MemoryMonitor` high-water mark until
  that monitor explicitly resets it. CPU peaks are zero.
- **The collection covers colocated roles.** Separate rollout/reward worker pools
  are not collected by the training stack. Multi-update tracks report track-wide
  totals, not a separate point for each internal optimizer update. FlashGRPO batch
  normalization requires one update per track.

# H3 Heavy Diagnostics

`h3_diag.py` reads these environment variables once at import. With all unset,
it installs no runtime wrappers or FSDP patches; the stage recorder is unchanged.

- `H3_DIAG_AG=1` adds `perf/stage/<stage>_ag_s` and
  `perf/stage/<stage>_ag_count`, each the mean of per-rank step totals.
- `H3_DIAG_GRAD_DUMP=<directory>` writes `grad_rank<rank>.pt` once per worker
  process, at its first optimizer step before clipping. Values are detached fp32
  local gradient shards, keyed by parameter name; no DTensor gather occurs.
- `H3_DIAG_GRAD_AB=<directory>` recomputes the first update's synced, pre-clip
  gradient once per size in `H3_DIAG_GRAD_AB_MICROS` (default `1,8,8,4,2`) before
  the real update, without stepping the optimizer, and writes
  `<index>_micro<size>/grad_rank<rank>.pt` plus `meta_rank<rank>.json`. All
  variants share one process, one LoRA init, and one batch. An entry may carry a
  mask suffix that zeroes the advantages outside one half for that variant only:
  `lo`/`hi` keep the first/second half of each rank's rows, `ra`/`rb` keep the
  ranks below/at-or-above `world_size // 2` (one root prompt per rank in the H3
  8x8 layout). The policy loss is linear in advantages, so `lo + hi` and
  `ra + rb` each sum to the unmasked gradient up to rounding; the reference KL
  term is not masked.
- `H3_DIAG_GRAD_AB_ONLY=1` (requires `H3_DIAG_GRAD_AB`) skips that real update:
  the first `train_track` returns the last variant's result, so no optimizer,
  scheduler, or step-timed EMA update happens and its `grad_norm` is the
  variants' placeholder `0.0`. Later calls update normally.
- `H3_DIAG_FIX_MEANS=<directory>` (requires `H3_DIAG_GRAD_AB`) pins the start of
  every variant's backward. While the variants run, each policy log-prob is
  computed from `mu_ref + (mu - mu.detach())`, `mu_ref` being the stored video
  and audio transition mean of the same sample id: its value is that of `mu_ref`
  and `mu` receives `(x_next - mu_ref) / std**2` times the unchanged log-prob
  factor, so variants differ only in `d mu / d theta`. Without
  `means_rank<rank>.pt` the first variant captures it from its own forward (so
  run the rollout's per-rank batch as that variant's micro size) and saves it with
  the variant's metadata; a later process, e.g. fp32, loads it and casts to its
  own means. Each `meta_rank<rank>.json` adds `means_file`, `means_captured`,
  per-modality `substitutions`, `logp_rows`, `logp_equal_rows` and
  `logp_max_abs_diff` against the stored `old_logp`, and `policy_loss_micros`. A
  micro row id missing from the file raises. The reference KL, its no-grad
  replay, TGR weights, clipping and advantages keep the live means and values.
- `H3_DIAG_STATE_SAVE=<directory>` writes `state_rank<rank>.pt` at each rank's
  first `train_track` call: the incoming Parts' `sample_ids`, `segment`,
  `conditions`, and `advantages` (no decoded media or rewards) and every trainable
  parameter's local shard in its own dtype, keyed by name.
- `H3_DIAG_STATE_LOAD=<directory>` reads that file at the same point, replaces
  the incoming Parts, and copies the saved shards into the live trainable
  parameters; names and shard shapes must match. Each loaded floating tensor is
  cast to the dtype of this process's own tensor at the same path, and a path the
  own Part lacks raises. EMA shadows, optimizer state, and later calls are
  untouched. Both run before the `H3_DIAG_GRAD_AB` variants, so those and the
  real update use the loaded state. The file is a pickle (`weights_only=False`);
  load only files this code wrote, with the same rank count and per-rank batch.
- `H3_DIAG_REPLAY_ONLY=1` (requires `H3_DIAG_STATE_LOAD`) makes every
  `DiffusionTrainer.train_step` skip the data source, rollout, and scoring. The
  driver joins `parts[0]` of `state_rank0.pt` … `state_rank<N-1>.pt` in rank
  order, `N` being the train stack's `dp_size`, and passes that one Part to
  `train_track`; the directory must hold exactly `N` state files. The saved
  per-rank batches are equal, so `DP_SCATTER`'s contiguous split returns rank `r`
  the rows of its own file, and the rank-side load above then installs that
  file's Parts and LoRA shards as usual. No rollout Sample exists, so the step
  logs `rollout/num_samples: 0` and no reward metrics, and returns a NaN mean
  reward; `train/` and `perf/` are logged as usual.
- `H3_DIAG_REWARDS=<directory>` writes `rewards_rollout<id>.json` on the driver,
  in request order, with `sample_id`, zero-based `prompt_index` within that rollout,
  and `reward`. Use fresh directories for separate runs.

## Gotchas

- **Replay-only never wakes or sleeps the rollout engine.** The engine and the
  reward are still built, so their startup requirements stand, but the modules a
  normal run releases in `rollout.sleep()` before its first update are still
  loaded during the replayed one.
- **Replay-only applies to every `train_step` and skips nothing else.**
  Evaluation and checkpoints run as configured, and each later step replays the
  same rows into a real update. Run it with `num_rollouts=1`, `eval_interval=0`,
  and `save_interval=0`.
- **Replayed data keeps the saved dtypes.** The rank-side cast compares the file
  with the replayed Part, which is the same data, so it changes nothing; a
  process at another precision receives conditions in the saving run's dtype.
- **Joining ranks with different text lengths pads the shorter ones.** The
  driver right-pads their text embeds with masked zeros. The first call's
  rank-side load restores each file's own shape; later calls train on the padded
  rows, whose trimmed view differs from the unpadded tensor in the last bits
  once a micro holds more than one row.
- **All-gather timing serializes communication.** The patch measures synchronized
  `FSDPParamGroup.unshard` plus `wait_for_unshard` intervals, including launch and
  copy-out, not isolated NCCL latency. The count increments once per newly launched
  multi-rank collective, not again at wait or for no-ops. During score/generate
  overlap, H3 transformer collectives belong to denoise, not the non-FSDP scorer.
- **The patch depends on private FSDP2 APIs.** Torch 2.13 uses
  `torch.distributed.fsdp._fully_shard._fsdp_param_group.FSDPParamGroup`,
  `FSDPMeshInfo`, `_all_gather_result`, and `_all_gather_process_group`;
  `unshard(async_op=False)` sets the pending result and `wait_for_unshard()` clears
  it. Verify these contracts after a torch upgrade. Retire the patch when native
  profiler events provide the required stage-level communication totals.

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
