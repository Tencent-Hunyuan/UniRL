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

1. **A real replay-complexity bug:** inference decode chunk boundaries were used
   as trainer forward boundaries.
2. **A scale-profile mismatch:** the feasibility profile replaced the intended
   `P=32, N=24, M=1, U=2` distributed run with a one-GPU, CPU-offloaded
   `P=1, N=4, M=1, U=1` run. Its timing is useful for diagnosing the bug, but it
   is not a production throughput measurement.

No post-fix GPU timing is available yet. All improvements below are expected
complexity reductions until the verification gates are completed.

## Observed Impact

The affected W&B run showed the following phase times. No run URL or credential
is included here.

| Phase | Observed wall time |
| --- | ---: |
| train | 1501-1528 s |
| native generate | 6.6 s |
| reward | 14.3 s |
| weight-sync/offload lifecycle | about 110-120 s |

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

**Root cause:** exact inference scheduler chunking was treated as required
trainer execution geometry. For decode, that made each new token trigger a new
full FSDP-wrapped decoder traversal, and T2TI rebuilt the result in three loss
paths. Exact token order and final KV geometry are required; 64 separate trainer
calls are not known to be required.

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

### 1. Parity-gated collapsed replay

T2TI replay now accepts `exact|collapsed`, validates the mode, and keeps `exact`
as the default. `collapsed` validates every captured chunk, concatenates the same
ordered `cache_input_ids`, and performs one causal prefill while preserving the
KV-length and rope postconditions
([implementation](../unirl/models/bagel/rl_ops.py#L92-L107),
[rebuild](../unirl/models/bagel/rl_ops.py#L285-L346),
[stage wiring](../unirl/models/bagel/diffusion.py#L261-L281)). Unit tests prove
one fake-model prefill instead of three in the fixture, with equal final cache and
embedding gradients
([tests](../tests/models/bagel/test_t2ti_replay.py#L143-L223)).

For the incident geometry, setting `C_i: 64 -> 1` changes expected invocation
counts from 788 initial / 1056 including recomputation to 32 initial / 48
including recomputation. That is a count reduction, not a measured 22x wall-time
speedup. Token and attention FLOPs remain sequence-length dependent. Collapsed
mode must remain opt-in until real BAGEL parity passes.

### 2. Once-per-update reference swap

The unified stack now supplies all image micro-batches at each optimizer-update
boundary
([stack hook](../unirl/train/unified_model_stack.py#L272-L289),
[call site](../unirl/train/unified_model_stack.py#L353-L365)). BAGEL builds their
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

## Verification Gates

The changes are not performance-complete until all of the following pass:

1. **Real-model replay parity:** on fixed Stage-0 traces, compare exact versus
   collapsed per-layer K/V tensors, final KV length/ropes, Stage-1 velocity,
   transition mean, log-prob, RatioNorm loss, and gradient norm/cosine. Run both
   CPU-offloaded smoke and GPU-resident FSDP geometry.
2. **Reference-hoist parity:** compare per-sample `v_ref`, MSE, total loss, and
   gradients against the former per-micro path. Instrument `_reference_weights`
   entries and require `N -> U`; verify update 1 observes weights after update 0.
3. **Invocation profiling:** count text-prefill, velocity, FSDP pre-forward, and
   checkpoint recomputation calls. For the incident fixture, collapsed replay
   should produce the 32/48 counts above.
4. **Memory and locality:** record RSS and PSS around reference snapshot, stash,
   restore, and optimizer step; record `numastat`, CPU affinity, GPU topology,
   host-to-device bandwidth, and GPU-idle samples. NUMA tuning follows work
   elimination, not vice versa.
5. **End-to-end timing:** rerun the one-GPU smoke and then the 32-device
   production profile. Report measured phase times and peak memory separately;
   do not infer production throughput from the feasibility run.

Until these gates pass, the incident is diagnosed and the asymptotic work is
reduced in code, but no GPU speedup or production stability result is claimed.
