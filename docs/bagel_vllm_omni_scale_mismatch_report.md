# BAGEL vLLM-Omni T2TI Train-Time Incident

## Summary

A one-GPU BAGEL T2TI feasibility run (`P=1, N=4, M=1, U=1`) spent roughly
25 minutes in each train phase even though native generation took seconds. The
trainer was replaying every Stage-0 scheduler chunk as a separate full BAGEL
decoder call. A 64-token thinking trace therefore became 64 full-model context
updates, and the same context was rebuilt in the image anchor, ratio, and MSE
paths. Persistent FSDP CPU offload turned each call into repeated parameter
paging.

This exposed two separate problems:

1. **A replay-complexity bottleneck:** the trainer independently rebuilt the
   exact incremental Stage-0 context in three image-loss paths, turning native
   decode scheduling into repeated FSDP execution geometry.
2. **A scale-profile mismatch:** the feasibility profile replaced the intended
   `P=32, N=24, M=1, U=2` distributed run with a one-GPU, CPU-offloaded
   `P=1, N=4, M=1, U=1` run. Its timing is useful for diagnosing the bug, but it
   is not a production throughput measurement.

A later one-node run retained global `P=32, N=24, M=1, U=2` on eight H20s and
disabled both persistent and lifecycle FSDP offload. Native vLLM-Omni rollout
completed all 768 T2TI samples, but training deadlocked when ranks with different
exact replay depths entered different FSDP2 collectives. That is a separate
distributed-ordering defect from the original one-GPU paging bottleneck.

An initial repair equalized the number of decoder calls with a graph-connected
no-cache hidden-state chain. A second eight-H20 run (`uqem9ggy`) proved that
call-count equality was necessary but insufficient: all 96 old-policy contexts
per rank completed in lockstep, AR backward completed, and all 48 update-0 image
references completed, but the first image RatioNorm backward deadlocked. Native
stacks showed seven ranks recomputing BAGEL's cached attention branch while one
rank recomputed the no-cache padding branch. The replacement repair therefore
uses a bounded, cache-faithful continuation. The next eight-H20 run (`7d62ya97`)
crossed the first image backward and completed optimizer update 0 at 15:15:32
SGT on 2026-07-16 with no OOM, NCCL, or fatal error. This validates the repaired
collective ordering, but the first update still took roughly three hours, so it
does not validate acceptable throughput. The same run later OOMed in update 1's
first image backward: an FSDP2 pre-backward all-gather requested 130 MiB with
41.25 MiB free while PyTorch held 5.01 GiB reserved but unallocated. The
optimized relaunch therefore also defaults the trainer to expandable CUDA
allocator segments; it still uses no persistent or lifecycle FSDP offload.
That setting cannot be inherited by vLLM-Omni 0.20.0: sleep mode uses its
``CuMemAllocator`` pool, which explicitly rejects expandable segments. The
rollout boot now removes only ``expandable_segments:True`` while spawning the
Omni process tree, then restores the train actor's exact allocator environment.

The production recipe now selects a second exact execution order,
`layer_major`. It preserves every native chunk's attention inputs and cache
transition but traverses the equivalent `(layer, chunk)` dependency graph one
decoder layer at a time. Each FSDP-wrapped block is entered once per sample and
loops the native chunks while its parameters are resident. Unequal trace depths
therefore change local inner work rather than the number or order of distributed
wrapper collectives. This path is bit-exact against chunk-major on the real
BAGEL/FlashAttention H20 stack and has passed unequal-depth FSDP2 ordering on
both CPU/Gloo and bf16-compute/fp32-master CUDA/NCCL.

The first layer-major eight-H20 run with the exact requested geometry
(`P=32, N=24, M=1, U=2`) completed all 768 native vLLM-Omni samples and both
optimizer updates in one round. W&B measured 3,148.119 s of training and
3,984.172 s end to end. The next round did not start: after Adam state had been
created and completed gradients remained resident, vLLM-Omni Stage 1 could not
remap its sleeping `CuMemAllocator` pool. The run exited on the next wake with a
CUDA OOM. This is a post-round trainer/rollout lifecycle overlap, not a failure
of the completed replay/backward round.

A combined experimental one-rollout build completed on one H20 with the same
incident geometry. It included the one-call collapsed candidate, the
once-per-update reference swap, and the other lifecycle fixes, and measured
485.0 s of train time versus 1501-1528 s. That is a 3.09-3.15x matched-geometry
result, not an ablation of any one change. The exact-versus-collapsed parity gate
subsequently failed, so the production and smoke recipes still select `exact`.

## Observed Impact

The affected run and the combined experimental smoke used `P=1, N=4, M=1, U=1`,
AR max 64, four diffusion steps, one SDE step, and one-GPU FSDP CPU offload.
This makes the train-phase comparison like-for-like. The runs are
[incident `87xh7jqr`](https://wandb.ai/linyuwus/bagel-unigrpo/runs/87xh7jqr)
and [candidate `8rdz7bqo`](https://wandb.ai/linyuwus/bagel-unigrpo/runs/8rdz7bqo).

| Phase | Incident | Combined experimental smoke |
| --- | ---: | ---: |
| train | 1501-1528 s | 485.035 s |
| native generate | 6.6 s | 9.585 s |
| reward | 14.3 s | 17.286 s |
| total measured step | not recovered | 518.141 s |
| Omni sleep / wake | included in lifecycle estimate | 2.661 s / 1.755 s |

The train phase dominated the iteration. A 30-second `py-spy` sample remained
inside `prepare_segment -> replay -> _build_contexts_from_replay ->
rebuild_text_context_from_chunks -> forward_inference`, confirming that the
process was repeatedly rebuilding the thinking cache rather than spending the
time in the four-step image sampler. W&B also sampled the GPU idle about 26% of
the time during this host-paged workload.

The worker exceeded 300 GB RSS. About 63% of its pages were on NUMA node 1 while
GPU 0 was attached to NUMA node 0. That locality mismatch increased host-to-GPU
latency, but it was secondary: even ideal NUMA placement would still execute the
same excessive number of decoder traversals.

The candidate still reached roughly 299 GiB RSS and a sampled 90,199 MiB of
device memory during single-tensor Adam. `py-spy` showed the experimental train phase
progress through RatioNorm backward, FSDP `foreach_reduce`, and then
`_single_tensor_adam`; it no longer remained in the per-token context rebuild.
Those samples identify substantial remaining CPU-offload/optimizer cost, but do
not prove the residual 485 s is irreducible or free of other bottlenecks.

## R3 Exact Batch-32 Result

Run [`rqjoxria`](https://wandb.ai/linyuwus/bagel-unigrpo/runs/rqjoxria), at
code revision `6e39f70d`, used one node with eight H20s, exact layer-major
replay, the native Stage-0-to-Stage-1 gradient boundary, no persistent FSDP CPU
offload, and no whole-trainer lifecycle offload. Its resolved geometry was the
requested global `P=32, N=24, M=1, U=2`: 32 prompt groups, 24 thoughts and one
image per thought, 768 paired samples, and two disjoint optimizer updates.

The W&B SDK returned one complete round with these driver wall times:

| Phase | Seconds |
| --- | ---: |
| vLLM-Omni wake | 2.4502 |
| native generate | 824.3466 |
| vLLM-Omni sleep | 2.9709 |
| reward | 6.2722 |
| train, both updates | 3,148.1194 |
| total round | 3,984.1721 |

The round therefore took 66m24.2s, of which training took 52m28.1s and native
generation took 13m44.3s. The training-phase diagnostics were:

| Train phase | Update 0 | Update 1 |
| --- | ---: | ---: |
| AR backward | 30.8399 s | 31.0244 s |
| eager image anchor | 742.7420 s | -- |
| image context/reference preparation | 356.2396 s | 377.9501 s |
| image RatioNorm + MSE backward | 692.0468 s | 842.8917 s |
| pre-optimizer cache reclaim | 1.8565 s | 2.2655 s |
| optimizer | 1.1571 s | 0.1573 s |

These per-phase fields came from the r3 rank selected by the then-current DP
collector; the later DP-critical-path maximum reduction was not in revision
`6e39f70d`. The driver-level `train_time_s` and `step_time_s` above are the
authoritative end-to-end intervals.

The round was numerically finite. PickScore reward mean/std/min/max was
`0.776082/0.083199/0.537614/0.969348`, with zero zero-variance prompt groups.
Update-0 image ratios were exactly 1.0. Update-1 image ratio
mean/std/min/max was
`1.000000145/0.000001464/0.999998755/1.000001547`; shared gradient norms were
`0.186666` and `0.180893`. Replay-depth ownership reduced the measured
collective-work estimate from 37,683 to 21,695. The native trace depth had
mean 216.766, min/median/p90/p99/max `19/190/332/577/1024`.

External `nvidia-smi` sampling during image backward observed cyclic peaks on
all cards rather than monotonic growth. The worst sample was 97,281 MiB used of
97,871 MiB, only 590 MiB free. Both updates nevertheless completed. After the
19:38:22 completion log, the next rollout tried to wake Stage 1 at about
19:38:41 and failed while remapping the vLLM `CuMemAllocator` pool at
`cumem_allocator.cpp:139`; the run exited 1. The first rollout had woken before
Adam was initialized. At the second wake, FSDP parameters, newly created Adam
state, and completed gradients overlapped the Stage-1 remap. This chronology
and the later clean release to zero device memory identify a rollout-boundary
footprint problem, not a leak or an OOM inside the completed training round.

## Scale Mismatch

| Dimension | Intended production | Feasibility incident |
| --- | ---: | ---: |
| prompts `P` | 32 | 1 |
| thoughts per prompt `N` | 24 | 4 |
| images per thought `M` | 1 | 1 |
| optimizer updates `U` | 2 | 1 |
| training placement | 32-way FSDP, GPU-resident shards | one GPU, persistent CPU offload |
| incident sampling override | not applicable | AR max 64, diffusion `T=4`, one SDE step |

The production profile now identifies the 32-device geometry explicitly, while
the single-GPU profile is named as a smoke profile; see
[`bagel_vllmomni_t2ti.yaml`](../examples/unified_model/bagel_vllmomni_t2ti.yaml#L1-L26),
[`bagel_vllmomni_t2ti_smoke.yaml`](../examples/unified_model/bagel_vllmomni_t2ti_smoke.yaml#L1-L40),
and the explicit launcher selection in
[`launch_bagel_vllmomni_t2ti.sh`](../scripts/launch_bagel_vllmomni_t2ti.sh#L4-L35).
Production must retain `U=2`; smoke must override it to `U=1`.

The distinction matters in both directions. CPU paging made the feasibility run
much slower per model invocation than the intended GPU-resident FSDP run. On the
other hand, production has 24 local thought/image samples per DP rank and longer
thinking limits, so exact per-token replay would still scale poorly even after
removing CPU offload.

## Distributed Exact-Replay Deadlock

The one-node batch-32 run (`mclck3gt`) used eight data-parallel H20 workers, so
each rank received 96 of the 768 paired thought/image samples. Two disjoint
optimizer updates then contained 48 samples per rank. The launch resolved
`batch_size=32`, `samples_per_prompt=24`, `samples_per_prompt(image)=1`, and
`num_updates_per_batch=2`; the deadlock was not a silent batch-size override.

Captured thinking traces contained roughly 138-706 Stage-0 scheduler chunks per
sample. During the no-grad old-policy preparation, faster ranks completed their
96 exact reconstructions and entered update work while a slower rank was still
at sample 83. Every exact chunk invokes all 28 FSDP-wrapped decoder blocks. With
`reshard_after_forward=true`, the fast rank eventually issued backward-side
FSDP collectives while the slow rank was still issuing forward all-gathers. The
collective order no longer matched, and all ranks stopped making progress.

This diagnosis is based on rank progress, stable GPU allocations, low-power
100%-utilization stalls, and native stacks showing seven ranks synchronized from
forward-side CUDA operations while one rank was inside autograd. The run was
then stopped intentionally. It produced no optimizer metric and did not OOM.

Setting `reshard_after_forward=false` alone is not a sufficient repair. An
adversarial two-rank FSDP2 test still deadlocked when one rank's backward
reduce-scatter met the other rank's pre-backward all-gather. The exact path must
make repeated-module traversal counts equal across ranks, not merely retain
unsharded parameters after forward.

The first count-equalized implementation also failed. In run `uqem9ggy`, all
ranks crossed the previous rank-7 sample-83 boundary and completed the entire
old-policy anchor. They then completed AR backward and update-0 reference
preparation before allocations stopped changing during the first image
backward. Native traces localized the mismatch to activation-checkpoint
recomputation: seven ranks were in the cached branch around
`qwen2_navit.py:580`, while one was in the no-cache branch around line 601.
An adversarial FSDP2/NCCL reproducer confirmed the same all-gather versus
reduce-scatter mismatch. A barrier, disabling activation checkpointing, and
`reshard_after_forward=false` did not repair it. Forward call count and cached
autograd topology both have to match.

Run [`7d62ya97`](https://wandb.ai/linyuwus/bagel-unigrpo/runs/7d62ya97)
used the cache-faithful topology on one 8xH20 node with global
`P=32, N=24, M=1, U=2`, no persistent FSDP offload, and no lifecycle offload.
All 768 vLLM-Omni KV/image outputs completed. The replay-depth permutation
reduced the observed summed per-position target from 37,683 to 21,695. Every
rank completed update-0 RatioNorm and MSE backwards, gradient clipping, and
Adam, then entered update 1. The logger returns the two `U=2` updates together,
so crossing optimizer 0 is a correctness signal rather than a W&B train metric.
The run was still dominated by exact Stage-0 replay, demonstrating that bounded
padding fixes ordering but retains excessive wrapped-layer entry cost. At
15:47:44 SGT, update 1 image micro 0 then OOMed in FSDP2's pre-backward
all-gather. The 130 MiB request saw only 41.25 MiB device-free even though the
allocator reported 5.01 GiB reserved but unallocated; Ray subsequently exited
all eight train actors. This was allocator fragmentation at the second-update
peak, not an NCCL ordering failure.

## Traversal Accounting

vLLM-Omni records every scheduled input chunk and preserves its offsets in the
replay metadata
([capture](../unirl/rollout/engine/vllm_omni/patches/runtime.py#L117-L135),
[serialization](../unirl/rollout/engine/vllm_omni/patches/runtime.py#L160-L185)).
The replay spec retained those boundaries intentionally
([`BagelThinkKVReplaySpec`](../unirl/models/bagel/conditions.py#L50-L64)). In the
incident trace, each sample had `C_i=64` chunks. The trainer loop in
[`rebuild_text_context_from_chunks`](../unirl/models/bagel/rl_ops.py#L285-L346)
ran one full decoder context update per chunk.

Before the remedies, each image sample rebuilt that `C_i`-chunk context three
times:

1. no-grad old-policy anchor in
   [`prepare_segment`](../unirl/algorithms/bagel_flow_unigrpo.py#L284-L312);
2. grad-enabled ratio replay in
   [`_ratio_norm_surrogate`](../unirl/algorithms/bagel_flow_unigrpo.py#L519-L558);
3. detached MSE context through
   [`build_forward_kwargs`](../unirl/models/bagel/diffusion.py#L665-L683).

With one selected SDE step, each sample also ran four image velocity forwards:
anchor, ratio, reference MSE, and current-policy MSE. AR replay added one
teacher-forced forward per sample, not 64 autoregressive forwards
([`_replay_inference`](../unirl/models/bagel/ar.py#L375-L417)). Therefore:

```text
per sample:       3 * C_i + 4 image + 1 AR
                = 3 * 64 + 5
                = 197 initial decoder traversals

four-sample group:
  C_total         = 4 * 64 = 256
  initial total   = 3 * C_total + 4*N + N
                  = 768 + 16 + 4
                  = 788
```

The shorthand `3C + 20` is valid only when `C` means the **group-total** chunk
count (`C=256`), yielding `788`. Combining per-sample `C_i=64` with the
group-global `20` gives `212`, which is a mixed-unit subtotal and must not be
reported as a per-sample count.

Activation checkpointing wraps every decoder block
([FSDP setup](../unirl/train/backend/fsdp/wrap.py#L122-L133)). The grad-bearing
ratio cache, ratio velocity, current-policy MSE velocity, and AR passes added
`C_total + 12 = 268` recomputation traversals. The incident therefore executed
approximately:

```text
788 initial + 268 checkpoint recomputation = 1056 top-level forward-equivalents
1056 * 28 BAGEL blocks = 29,568 decoder-layer forward calls
```

This excludes backward kernels themselves. It is a model-invocation count, not a
claim that every invocation has identical FLOPs; sequence lengths differ.

## Root Cause And Amplifiers

**Root cause:** exact incremental Stage-0 reconstruction was repeated in the
image anchor, ratio, and MSE paths while every decoder block was FSDP-wrapped and
CPU-offloaded. For decode, each captured token therefore triggered another full
FSDP traversal in each path. The native boundaries cannot yet be labeled
removable: both tested coalescing geometries exceeded the numerical parity
budget.

**Amplifiers:**

- `CPUOffloadPolicy`, `reshard_after_forward=true`, and per-block FSDP wrapping
  paged blocks for every traversal
  ([configuration path](../unirl/train/backend/fsdp/wrap.py#L77-L87)).
- Activation checkpointing replayed all grad-bearing traversals during backward.
- Full fine-tuning made `_reference_weights` clone, replace, and restore every
  trainable local shard
  ([swap implementation](../unirl/algorithms/bagel_flow_unigrpo.py#L220-L282)).
  On one rank, "local shard" is the whole trainable decoder. The incident path
  performed this full stash/swap once per image micro-batch.
- The immutable bf16 reference, fp32 masters/gradients, Adam state, and temporary
  fp32 stash are consistent with the observed greater-than-300-GB RSS. CPU Adam
  also uses the single-tensor path
  ([optimizer setup](../unirl/train/optim.py#L95-L119)).
- Remote-NUMA pages and serial navit `bs=1` micro-batches further reduced device
  utilization. They did not create the `C` multiplier.

## Implemented Remedies

### 1. Experimental collapsed replay

T2TI replay now accepts `exact|collapsed`, validates the mode, and keeps `exact`
as the low-level and recipe default. The current `collapsed` candidate validates
every captured chunk, preserves the native initial prefill, concatenates the same
ordered decode-tail IDs, and performs one causal decode update while preserving
the KV-length and rope postconditions
([implementation](../unirl/models/bagel/rl_ops.py#L92-L107),
[rebuild](../unirl/models/bagel/rl_ops.py#L285-L346),
[stage wiring](../unirl/models/bagel/diffusion.py#L261-L281)). Unit tests prove
two fake-model calls instead of three in the fixture, with equal final cache and
embedding gradients
([tests](../tests/models/bagel/test_t2ti_replay.py#L143-L223)).

For the incident geometry, setting `C_i: 64 -> 2` changes expected invocation
counts from 788 initial / 1056 including recomputation to 44 initial / 64
including recomputation. That is a 16.5x count reduction, not a wall-time claim;
token and attention FLOPs remain sequence-length dependent. The measured 485 s
smoke used the earlier one-call candidate (`C_i: 64 -> 1`) and therefore does not
certify the current two-call implementation.

The real captured-trace gate rejected both candidates. For the one-call version,
cache relative L2 was 0.01290, velocity relative L2 was 0.00744, and selected
decoder-gradient relative L2/cosine were 0.05721/0.99836. Preserving the initial
prefill improved cache relative L2 to 0.00863, but velocity remained 0.00745 and
gradient relative L2/cosine became 0.06546/0.99796. The unchanged limits are
0.005 for cache/velocity relative L2 and 0.01/0.999 for gradient relative
L2/cosine. `collapsed` therefore remains explicit experimental behavior; neither
shipped recipe enables it.

### 2. Once-per-update reference swap

The unified stack now supplies all image micro-batches at each optimizer-update
boundary immediately before image backward
([stack hook](../unirl/train/unified_model_stack.py#L251-L252),
[cleanup](../unirl/train/unified_model_stack.py#L285-L306),
[call site](../unirl/train/unified_model_stack.py#L376-L380)). BAGEL builds their
detached MSE contexts first and computes all reference velocities inside one
`_reference_weights` scope
([`prepare_update_batch`](../unirl/algorithms/bagel_flow_unigrpo.py#L314-L391)).
The current-policy forwards and backwards still execute per sample, and direct
algorithm callers retain a fallback path.

This changes full local-shard stash/copy/restore complexity from `O(N)` swaps per
rollout shard to `O(U)` swaps. It is 4 to 1 for the feasibility profile and 24 to
2 per production DP rank. It does not remove the required `N` reference velocity
forwards.

### 3. Production/smoke profile separation

The production profile restores the intended distributed geometry and
GPU-resident FSDP placement. The named smoke profile owns one-GPU CPU offload and
reduced `P/N/U` settings. Composition and launcher-profile tests prevent the
smoke geometry from silently becoming the production recipe
([profile tests](../tests/rollout/vllm_omni/test_bagel_scale_profiles.py#L34-L88)).
Both profiles resolve to `t2ti_replay_chunk_mode: exact`; a missing schema key in the
first post-fix launch caused Hydra to reject the override before model startup,
and the profile test now covers the declared/inherited value. That launch error
was a validation wiring omission, not a cause of the original 25-minute phase.

### 4. Cache-faithful collective padding

Before the first exact decoder traversal for each sample, every pure-DP trainer
rank now all-reduces its local chunk count with `MAX`. A rank with `C` real
chunks and collective target `T` executes the real reconstruction followed by
`T-C` discarded decoder traversals. The returned KV cache, length, and rope
position remain those of the real trace.

The discarded traversals now use the same cached
`forward_cache_update_text(update_past_key_values=True)` path as real replay.
The dummy context forks the real terminal `NaiveCache`, keeps the newest K/V
slot from every layer, and executes a one-token cached update for each padding
slot. After every call, the returned cache is forked and trimmed back to its
newest slot. This preserves the per-layer cache recurrence and reverse FSDP hook
order while bounding retained cache length at one token per layer instead of
growing a full dummy prefix. The final exact zero touches terminal K/V from
every layer and is added to the replay log-prob; the semantic cache returned to
the image path is never mutated or advanced.

The previous no-cache hidden chain has been removed. A production-path two-rank
FSDP2 regression now uses fresh inputs per real chunk, three recurrent K/V
layers, activation checkpointing, `reshard_after_forward=true`, unequal replay
depths, and a semantic image traversal over the untouched real cache. The
bounded cached repair completes and matches the independently averaged unpadded
gradient exactly. Separate CUDA/NCCL adversarial tests passed depths 100 versus
3 on two ranks and 30/3/4/5 on four ranks.

The synchronization deliberately targets the current `fsdp_mode=full`, `SP=1`
recipe, where the default process group is the FSDP data-parallel world. It does
not claim HSDP, tensor/sequence-parallel, Ulysses, or expert-parallel support;
those layouts need their exact shard group and may impose activation-shape
collectives that a one-token dummy cannot satisfy.

### 5. Replay-depth-aware DP ownership

The driver now applies one deterministic permutation to both 1:1 AR and image
tracks before `DP_SCATTER`. For each optimizer update independently, it sorts
samples by exact replay depth, groups adjacent depths in buckets of DP size, and
assigns one sample from every bucket to each rank while balancing cumulative
load. The output still has equal contiguous shards, preserves each update's
original global membership, and keeps every AR/image row and advantage paired.
It changes ownership and reduction order only; `P=32`, `N=24`, `M=1`, `U=2`,
the 768 global samples, and exact replay math are unchanged.

An offline analysis of the captured 8x96 trace estimated that the summed
per-position collective target falls from 50,030 to 29,507 replay traversals,
1.752x to 1.034x the average real work. Run `7d62ya97` then logged its own
realized reduction from 37,683 to 21,695. Both are traversal accounting, not
wall-time results.

### 6. Layer-major exact replay

Exact reconstruction forms a two-dimensional dependency graph. Node
`(layer, chunk)` depends on `(layer - 1, chunk)` for its hidden state and on
`(layer, chunk - 1)` for its cached K/V. Both chunk-major and layer-major orders
are valid topological traversals. The new path constructs the same token IDs,
position IDs, rope embeddings, query/key indexes, causal flag, and mutable cache
updates for every original native chunk; it changes only the traversal order.

The dispatch is installed on each vendored decoder layer before Accelerate
device hooks, activation checkpointing, and FSDP2 wrapping. A normal layer call
falls through to the original vendor `forward`. A replay call enters the wrapped
layer once and invokes its `forward_inference` implementation directly for each
native chunk while that layer is unsharded. The direct inner call intentionally
does not recurse through `Module.__call__`, so it does not issue nested FSDP or
checkpoint hooks. The outer call still owns both mechanisms.

For the captured batch-32 trace, the depth-aware planner reduced one 96-sample
pass to 21,695 full-decoder traversals per rank. With 28 decoder blocks this is
607,460 wrapped block entries. Layer-major replay uses `96 * 28 = 2,688` wrapped
entries for the same pass, while retaining the same 21,695 native inner
block/chunk computations. This is a 226x reduction in wrapper entry and
unshard/reshard count, not a 226x reduction in attention FLOPs and not yet a
wall-time claim.

Layer-major replay does not need distributed dummy traversals: every rank enters
each wrapped block once per sample even when local chunk counts differ. The
depth-aware ownership permutation remains useful because it balances the local
inner compute duration between those synchronized outer entries. Cache-faithful
padding remains available for the chunk-major exact path as a validated
fallback.

### 7. Post-r3 memory and replay controls

R3 established the timing and memory baseline but ran revision `6e39f70d`; the
following controls were implemented afterward and have not yet been validated
by an r4 end-to-end run.

**Lazy exact update-0 anchor.** Because the two updates consume disjoint
mini-batches, update 0's exact current replay occurs at the same weights as its
old-policy anchor. The trainer now detaches that replay's log-probability and
transition mean for the update-0 anchor, while still eagerly computing all
later-update anchors before optimizer 0. At the r3 geometry this changes eager
anchor replays from 96 to 48 per rank and total Stage-0 context builds from 192
to 144 per rank without accepting cross-runtime rollout means or changing the
old-policy state.

**Prepared replay-data staging.** Detached Stage-0 caches and frozen reference
velocities queued for future image micros can now be staged on CPU and hydrated
one micro at a time. These are graph-free replay inputs and targets. This path
does not move FSDP parameters, gradients, optimizer state, or the rollout engine
and is therefore distinct from model-state CPU offload.

**Allocator reclamation and telemetry.** The production stack can return
inactive cached blocks after each image micro and optimizer step, reset peak
counters per update, and report peak allocated/reserved memory using a maximum
over DP workers. This does not reduce the live tensor requirement; it addresses
the observed sub-GiB fragmentation margin and makes the next H20 result
measurable.

**Optimizer-only rollout parking.** Before an external vLLM-Omni wake, the
trainer now clears completed gradients and moves only sharded Adam state to CPU.
Adam is restored only after every Omni stage acknowledges sleep. FSDP parameters
and shards remain GPU-resident, and configuration validation requires both
`enable_fsdp_offload=false` and `backend.fsdp_cfg.cpu_offload=false`. This narrow
boundary directly targets the completed-gradients-plus-Adam overlap that caused
r3's second Stage-1 wake OOM; it is not persistent or whole-trainer FSDP CPU
offload.

**Flow-many remains gated.** An exact CFG=1 implementation can traverse all
selected SDE velocity streams inside one layer-major decoder pass, reducing
wrapped-layer entries across anchor, reference, and policy velocity replay. It
may retain more simultaneous activations, so the production profile keeps
`t2ti_flow_many_enabled=false` until an instrumented r4 H20 memory and numerical
gate passes. No r4 throughput or memory success is claimed here.

## Verification Status

| Gate | Status | Evidence / remaining work |
| --- | --- | --- |
| Unit and config regression | passed | 110 passed, 1 skipped in the focused BAGEL/Omni suite; coverage includes cached padding, layer-major cache/loss/full-gradient parity, Accelerate-hook restoration, profile wiring, phased-hook compatibility, pairing, and update membership |
| Two-rank FSDP2 ordering | passed on CPU/Gloo | production replay helper, unequal 2-vs-5 real depths, three-layer K/V recurrence, activation checkpointing, `reshard_after_forward=true`, and a semantic image call completed and matched averaged unpadded gradients |
| Adversarial CUDA/NCCL ordering | passed in focused toys | bounded cache-DAG padding passed 2-rank depth 100-vs-3 and 4-rank 30/3/4/5 cases; the removed hidden-chain control reproduced the collective mismatch |
| Layer-major local exact parity | passed | production installer/helper are bit-exact against chunk-major for terminal K/V, downstream image output/loss, and every decoder gradient; wrapped calls fall from `chunks + image` to `replay + image` per layer |
| Layer-major FSDP2 ordering | passed on CPU/Gloo | unequal 2-vs-5 depths with composable activation checkpointing and `reshard_after_forward=true` complete without padding and match independently averaged gradients |
| Layer-major real BAGEL CUDA parity | passed bit-exact | captured 126-token/64-chunk trace on one H20; deterministic FlashAttention self-control and chunk-major-vs-layer-major K/V, velocity, transition mean, log-prob, and selected gradients all had zero difference |
| Layer-major CUDA/NCCL FSDP2 ordering | passed | two H20s on Torch 2.11/CUDA 12.9, unequal 2-vs-5 depths, bf16 compute/fp32 masters, activation checkpointing, and `reshard_after_forward=true`; exit 0 in 13.66 s |
| One-H20 end-to-end smoke | mechanical execution completed | exit 0, four images, finite losses, optimizer step, 485.035 s train; tested replay mode failed numerical parity |
| Captured exact/collapsed kernel parity | **failed** | both one-call and prefill-preserving two-call candidates exceeded cache, velocity, and gradient limits |
| Full RatioNorm/FSDP parity | not run | the standalone checker uses one captured sample, an unwrapped CUDA LoRA bundle, selected gradients, and a synthetic objective |
| Reference-hoist parity | unit-tested, not GPU-instrumented | lifecycle/error cleanup is covered; `N -> U` swap counts and full gradient parity still need instrumentation |
| Eight-H20 global batch 32 without padding | **failed** | all 768 rollouts completed; variable exact replay depths deadlocked FSDP training before the first optimizer metric |
| Eight-H20 count-equalized hidden padding | **failed** | all anchors, AR backward, and update-0 reference prep completed; cached-vs-no-cache topology deadlocked the first image backward (`uqem9ggy`) |
| Eight-H20 cache-faithful padding plus DP balancing | optimizer-0 gate passed; update 1 OOMed | `7d62ya97` completed optimizer 0 with no ordering failure, then fragmented at update-1 image micro 0; roughly three-hour first update remains unacceptable |
| Eight-H20 layer-major batch 32 | full training round passed; next wake OOMed | `rqjoxria` completed all 768 native samples and both updates in 3,984.172 s total; the following Stage-1 wake failed while Adam and completed gradients were resident |
| Post-r3 memory/lifecycle controls | unit/config covered; H20 pending | lazy update-0 anchor, prepared replay-data staging, allocator reclaim/telemetry, and optimizer-only rollout parking are implemented; r4 must pass the next wake and another round before they are accepted |
| Flow-many H20 gate | disabled, pending r4 | exact CFG=1 implementation and focused parity/collective tests exist, but production remains `t2ti_flow_many_enabled=false` until peak memory is measured |
| 32-device production | not run | encoded scale remains `P=32, N=24, M=1, U=2`; first prove repeatable r4 rounds on the one-node batch-32 geometry |
| Reward learning curve | not established | r3 produced one finite reward point (`0.776082` mean); one round cannot establish reward growth |

The standalone checker does compare per-layer K/V, Stage-1 velocity, transition
mean, log-prob, and representative decoder gradients with fixed stochastic
inputs. Its exact-order gate enables FlashAttention's deterministic backward and
requires an identical chunk-major self-control before accepting the candidate;
production keeps the native nondeterministic backward to avoid its extra memory
and runtime. The checker does not certify RatioNorm loss, the full gradient
vector, CPU-offload geometry, or distributed FSDP ordering; the separate
two-H20 FSDP2 gate covers collective topology with a production-shaped toy.

## Disposition

The incident bottleneck and the recipe scale mismatch are confirmed. The code
now separates production from smoke geometry, balances exact replay depth, and
uses exact layer-major replay without changing native chunks or logical
`P/N/M/U`. R3 proved that this path can complete one eight-H20 batch-32 training
round: both optimizer updates finished in 52m28.1s and the whole round finished
in 66m24.2s. That is materially below the prior roughly three-hour first-update
baseline, but it is not yet sustained throughput: the next Stage-1 wake OOMed.

The current r4 candidate retains `P=32, N=24, M=1, U=2`, exact replay, and
GPU-resident FSDP parameters/shards. It adds the exact lazy update-0 anchor,
one-micro prepared replay-data staging, allocator reclamation/telemetry, and
optimizer-only rollout parking. The last mechanism temporarily parks sharded
Adam state and clears completed gradients; it does not enable persistent or
lifecycle FSDP CPU offload. Flow-many is implemented but remains explicitly
disabled pending the H20 peak-memory gate.

The immediate acceptance test is an instrumented r4 run on the same one-node
geometry through the second rollout wake and both subsequent updates, with no
OOM, finite ratios/losses, and measured critical-path memory and wall time. No
r4 success has been observed yet. A repeatable one-node result is required
before scaling to 32 devices and collecting enough rounds to judge reward-curve
growth.
