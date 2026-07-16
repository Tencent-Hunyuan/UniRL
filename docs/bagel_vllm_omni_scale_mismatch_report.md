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
optimized relaunch therefore also defaults to expandable CUDA allocator
segments; it still uses no persistent or lifecycle FSDP offload.

The production recipe now selects a second exact execution order,
`layer_major`. It preserves every native chunk's attention inputs and cache
transition but traverses the equivalent `(layer, chunk)` dependency graph one
decoder layer at a time. Each FSDP-wrapped block is entered once per sample and
loops the native chunks while its parameters are resident. Unequal trace depths
therefore change local inner work rather than the number or order of distributed
wrapper collectives. This path is bit-exact against chunk-major on the real
BAGEL/FlashAttention H20 stack and has passed unequal-depth FSDP2 ordering on
both CPU/Gloo and bf16-compute/fp32-master CUDA/NCCL. End-to-end batch-32 timing
remains the deployment gate.

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
| Eight-H20 layer-major batch 32 | pending | both prerequisite CUDA gates passed; compare first-optimizer wall time and update-1 memory against `7d62ya97` |
| 32-device production | not run | encoded scale remains `P=32, N=24, M=1, U=2`; validate the one-node batch-32 run first |
| Reward learning curve | not run | a one-rollout performance smoke cannot establish an increasing reward curve |

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
now separates production from smoke geometry, hoists the full-FT reference swap,
captures replay metadata, collectively balances exact replay traversal depth,
uses cache-faithful bounded padding, redistributes replay depths without changing
batch or update membership, and can traverse exact replay layer-major. The
combined experimental build measured 3.09-3.15x faster in one matched-geometry
comparison, but both collapsed candidates changed replay outputs beyond the
preset parity budget and remain disabled. Cache-faithful chunk-major replay has
now passed a real eight-H20 optimizer gate, proving the distributed correctness
repair, while also proving that its remaining wrapper multiplier is too slow.

The immediate production candidate is layer-major exact replay. It changes
neither native chunk geometry nor logical `P/N/M/U` scale, removes collective
padding, and targets the FSDP/checkpoint wrapper overhead exposed by the
three-hour baseline. The first optimized launch deliberately keeps
`reuse_ratio_context_for_mse=false`; phasing RatioNorm before the reference swap
would retain image gradients beside the fp32 live-weight stash, and the baseline
already fragmented at the second-update peak. The launcher defaults
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for the optimized run, but
does not use offload or reduce batch size. Context reuse is implemented as a
separate follow-on optimization and remains disabled until an instrumented
peak-memory gate passes.

Real captured BAGEL/FlashAttention parity and unequal-depth CUDA/NCCL FSDP2
ordering now pass. The remaining gate is the same one-node batch-32 run through
both optimizer updates with materially lower wall time and no fragmentation.
Only then should the 32-device `P=32, N=24, M=1, U=2` run be used to evaluate
sustained throughput and reward growth.
