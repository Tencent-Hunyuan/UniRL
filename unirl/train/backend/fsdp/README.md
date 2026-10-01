# FSDP backend

> **Where it fits:** the training-side parameter backend — `fsdp_wrap` installs FSDP2
> sharding, `backend.py` owns construction/weight-load/optimizer order, `state.py` owns
> coarse role onload/offload, and `offload_plan.py` is the quantitative placement planner.

## offload_plan.py (section 6A)

Pure-Python planner: it consumes a *measured* `PhaseProfile` plus a `BlockInventory` and
emits an auditable resident/streamed split. It imports no torch, allocates nothing, reads
no checkpoint, and does not decide anything without a profile.

```python
plan = plan_placement(inventory, profile, capability, exposure_limit_ms=..., headroom_bytes_by_rank=...)
print(plan.to_json())
```

## Gotchas

- **The memory ledger is exclusive by contract.** `PhaseCost.resident_gpu_bytes` must
  measure the phase's per-rank high-water *excluding all block parameter storage*; the
  planner then adds `root_bytes`, resident block staging, streamed staging and headroom
  itself. A profile that already includes parameter bytes is double-counted and will
  produce an over-conservative plan — report the exclusion in the profile fingerprint.
- **`compute_ms` must exclude transfer waits.** It is kernel time only; `transfer_ms`
  carries the measured shard H2D + all-gather + cast. Adding a block wall time that
  already contains an H2D wait to the transfer model double-books the same stall.
- **Only `bandwidth_record_kind="shard"` may be summed with `shard_bytes`.** Full-parameter
  and cast-buffer transfers are separate storage roles with their own measured rates; the
  field exists so a profile cannot silently mix them.
- **An executor without a measured residency bound is charged conservatively.** With
  `live_streamed_bound=None` (the honest default for native FSDP2 `CPUOffloadPolicy`)
  every streamed block is counted as live at peak. The two-slot model is only applied
  when the executor *declares and has verified* that bound — a two-block memory model must
  never stand in for an unproven parameter lifecycle.
- **The plan is group-common.** Every rank in `per_rank` is evaluated for the same
  resident prefix, and the chosen candidate must satisfy the tightest rank budget and the
  node-summed host budget. There is no per-rank plan, because the ranks sharing an FSDP
  shard group must agree on ownership.
- **Alias groups and fixed blocks are placement units.** Tied/aliased blocks move together
  (a split is rejected), and root-owned leftover parameters are never streamed.
- **Full residency is not a fallback.** When no candidate satisfies the budgets the plan
  is `INFEASIBLE` with per-candidate reasons, and nothing is enabled.
- **Missing phases are missing evidence.** A profile that does not measure every phase in
  `PHASES` rejects every candidate instead of extrapolating from the phases it has.
