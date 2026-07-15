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

## Verification Status

| Gate | Status | Evidence / remaining work |
| --- | --- | --- |
| Unit and config regression | passed | 88 passed, 2 skipped; replay, lifecycle, scale-profile, and strict-contract coverage |
| One-H20 end-to-end smoke | mechanical execution completed | exit 0, four images, finite losses, optimizer step, 485.035 s train; tested replay mode failed numerical parity |
| Captured exact/collapsed kernel parity | **failed** | both one-call and prefill-preserving two-call candidates exceeded cache, velocity, and gradient limits |
| Full RatioNorm/FSDP parity | not run | the standalone checker uses one captured sample, an unwrapped CUDA LoRA bundle, selected gradients, and a synthetic objective |
| Reference-hoist parity | unit-tested, not GPU-instrumented | lifecycle/error cleanup is covered; `N -> U` swap counts and full gradient parity still need instrumentation |
| 32-device production | not run | scale is encoded, but exact replay's variable native chunk counts must be made collective-safe before launch |
| Reward learning curve | not run | a one-rollout performance smoke cannot establish an increasing reward curve |

The standalone checker does compare per-layer K/V, Stage-1 velocity, transition
mean, log-prob, and representative decoder gradients with fixed stochastic
inputs. It does not certify RatioNorm loss, the full gradient vector, CPU-offload
geometry, or distributed FSDP ordering.

## Disposition

The incident bottleneck and the recipe scale mismatch are confirmed. The code
now separates production from smoke geometry, hoists the full-FT reference swap,
captures replay metadata, and includes a measurable fast-path experiment. The
combined experimental build measured 3.09-3.15x faster in one matched-geometry
comparison, but both collapsed candidates changed replay outputs beyond the
preset parity budget. The fast path is not enabled by default, and the exact
production replay multiplier remains unresolved.

The next production-safe optimization should preserve exact chunk math while
removing duplicate reconstruction: phase image training per optimizer update so
RatioNorm builds each exact grad context once, retains detached K/V views after
backward, performs one batched reference swap, and reuses those views for MSE.
That removes one exact prefix reconstruction per image without changing cache
geometry. Separately, distributed exact replay needs collective-count balancing
or a validated FSDP policy that tolerates variable native completion lengths.
Only after those two items pass should the 32-device `P=32, N=24, M=1, U=2`
learning run be used to judge throughput and reward growth.
