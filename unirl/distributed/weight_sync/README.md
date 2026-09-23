# Weight Sync

> **Where it fits:** the *sync* back-edge of the loop (dedicated modes only) —
> rollout → reward → advantage → train → **sync**. In: trained weights from the
> backend. Out: weights pushed into the rollout engine.
> Full map: [`../../README.md`](../../README.md).

<div align="center">
  <img src="../../../assets/weight-sync-new.png" alt="UniRL weight sync: every train rank all-gathers the FSDP shards into full tensors, then one of five handlers — a cell of the LoRA/full by colocate/separate grid — pushes them into the rollout engine" width="100%">
</div>

*Every `sync()` is the same two phases — **gather** the weights (an all-gather, so it runs on **every train rank**) then **push** into the engine — and the five handlers fill a **[ what × where ]** grid: LoRA vs full × colocate vs separate.*

## What it is

`unirl.distributed.weight_sync` delivers freshly-trained weights from the train
slab into the dedicated rollout engine(s). It is the dashed back-edge of the
training loop, and it exists **only** when rollout runs on a dedicated engine
(SGLang / vLLM-Omni, in `separate` or `colocate` layouts); direct sampling (the
trainside engine) needs no sync because it samples the live training weights
in-process.

## Why it exists

The non-obvious cost: pushing weights from an FSDP model is **not** a rank-0
operation, even though only rank 0 transmits. *Reading* the weights
(`extract_lora_tensors` for LoRA, `_to_full_tensor` for full) redistributes every
FSDP `DTensor` shard to `Replicate` — a collective the whole train mesh must enter
in lockstep. So every handler runs as a `BROADCAST`-dispatched `sync()`: ranks ≥1
drive their half of the all-gather and discard, rank 0 alone transmits. That is why
this is a sibling `Remote` family and not a rank-0 helper — and why gating the push
on `if rank == 0` deadlocks the gather. The transport fan-out (LoRA vs full;
NCCL / IPC / serialized; colocate vs cross-node) then rides on that one shared
materialization.

## How it works

- **Two families, picked by `_target_`.** *LoRA* (`lora/`) extracts the trained
  adapter and loads it via the engine's `set_lora_from_tensors`. *Full-weight*
  (`full/`) materializes the full base weights (optionally with LoRA folded in) and
  streams them in buckets.
- **`sync()` is a train-mesh collective.** Extracting weights redistributes each
  FSDP `DTensor` shard to a full tensor — a collective every train rank must run —
  so `sync()` is dispatched `BROADCAST`. Only the *push* is rank-0 (for NCCL and
  Remote-LoRA); ranks ≥1 run the all-gather and discard.
- **Transports.** LoRA: `LocalLoraWeightSync` (colocate, in-process sibling) and
  `RemoteLoraWeightSync` (cross-slab rank-0 Ray push). Full: `NCCLWeightSync`
  (separate slabs, cross-node capable), `TensorWeightSync` (colocate serialized
  handoff), `IPCWeightSync` (colocate CUDA-IPC over ZMQ). Colocate handlers take the
  engine as a same-Worker sibling; separate/NCCL need a one-time driver handshake.
- **Routing.** `param_prefix` prepends the model's canonical key prefix;
  `track_prefix` further prefixes so a `ComposedRolloutEngine` can demux the update
  to one child (PE registers one handler per track).

The one bit-equality safety net is the LoRA `verify` checksum read-back — but it is
vLLM-Omni-only, so SGLang LoRA runs have none.

**Extending it:** a new transport subclasses `LoraWeightSyncBase` or
`FullWeightSync` (the FSDP→full materialization, bucketing, and `name_remap` are
already done), implements `sync()` as `BROADCAST` shipping each bucket, and adds the
matching receiver on the engine side (`../../rollout/engine/`).

## Gotchas

- **A `pp_size>1` rollout cannot be weight-synced through `TensorWeightSync`** — SGLang
  deserializes `serialized_named_tensors[ps.tp_rank]`, a *stage-local* index, so every
  pipeline stage reads the same slots, and a CUDA-IPC payload is only openable on the GPU
  that produced it: stage `P-1` reading stage 0's handle dies on `cudaErrorInvalidValue`
  and takes the server with it. Sending CPU tensors instead is not an escape — SGLang's
  reduction patch rewrites the reduce args at index 6, which a CPU tensor's args do not
  have, so `ForkingPickler.dump` raises `IndexError`. `sync()` therefore fails closed on a
  PP layout; PP weight sync needs a rank-addressable mechanism (the checkpoint-engine IPC
  route, whose own guard says stage-local socket routing is required). Note the
  separate-slab `NCCLWeightSync` path has a *different* PP blocker (its group rank
  collides) and is guarded too.
- **A delegated rollout target occupies `tp_size * pp_size` group ranks** — one engine per
  DP replica owns that replica's whole TP x PP rank set, so `NCCLWeightSync.connect`
  requires `num_rollout_gpus == num_targets * tp_size * pp_size` and offsets target `i` by
  `i * tp_size * pp_size`. Counting `num_targets * tp_size` (TP only) under-sizes the
  group by `num_targets * tp_size * (pp_size - 1)`; the check exists because the
  single-engine fan-out made exactly that drift reachable.
- **`sync()` is a train-mesh collective** — the FSDP→full materialization runs on
  *every* train rank; never gate it behind `if rank == 0`. The rollout receiver owns
  routing after materialization: SGLang TP performs the push only on each group's
  `tp_rank == 0`, not global rank 0.
- **`transfer_queue` is not weight sync** — that's the rollout→trainer data plane
  for bulky rollout outputs (segments, conditions, decoded media); weight sync is
  trainer→rollout. Don't conflate them.
- **`param_prefix` mismatch silently corrupts the load** (wrong/zero layers) —
  `verify` is designed to catch it, but it's vLLM-Omni-only and off by default, so
  most runs have no net at all.
- **`RemoteLoraWeightSync` with `copy=False` breaks on TP>1 engines** — the zero-copy
  adapter handle carries a one-shot file descriptor consumed by the first worker, so
  the `collective_rpc` broadcast to ranks 2..N gets a dead handle (HI3). Set
  `copy=True` for any TP>1 stage; `copy=False` is only safe for a TP=1 separate slab (SD3).
- **`CheckpointWeightSync.version` is a filename sequence**, not a receiver
  idempotency key. Other transports carry no independent version ledger.
- **Direct vLLM IPC deliberately delegates transfer mechanics to vLLM 0.27.**
  UniRL supplies lazy canonical FSDP export, rank consensus, manifests, and
  fail-stop publication; vLLM owns CUDA IPC handle routing, TP slicing,
  model-specific fusion, and layerwise `load_weights`. Its packed producer reads
  one tensor beyond the configured byte boundary before flushing, so UniRL plans
  metadata chunks first to keep lazy materialization bounded.
