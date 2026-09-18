# Trainside rollout engine

> **Where it fits:** the in-process `Pipeline` *is* the sampler, so this engine owns
> rollout execution order for direct-sampling recipes. In: a request `Sample` from the
> trainer. Out: the same `Sample` with the frontier gen Part filled.

## Scheduling modes

`scheduling_mode` selects how one frontier's rows are grouped into model forwards:

| Mode | Behaviour |
| --- | --- |
| `off` (default) | One `pipeline.generate` per frontier, optionally chunked by `forward_batch_size` rows. |
| `shape_bucket` | Fills one or more micros per geometry bucket, aligned to complete prompt groups. |
| `continuous` | Not implemented (step-level refill is a separate milestone); rejected at construction. |

`shape_bucket` is configured on the recipe's `rollout:` block — the constructor kwargs
`TrainsideEngineConfig` mirrors are not the runtime entry:

```yaml
rollout:
  _target_: unirl.rollout.engine.trainside.engine.TrainsideRolloutEngine
  scheduling_mode: shape_bucket
  bucket_batch_sizes: {default: 16}
```

`bucket_batch_sizes` is keyed by the bucket token logged at plan time
(`<pipeline>|<latent shape>|<conditioning layout>|cfg<rows>`), with `default` covering
buckets that are not named explicitly. An unlisted bucket with no `default` is a hard
error, not an unbounded micro.

## Gotchas

- **Capacity is counted in execution rows, not requests.** With CFG
  (`guidance_scale > 1`) the model runs `[uncond, cond]` rows, so a `default: 16`
  capacity holds 8 requests at `cfg_rows=2`. This is deliberately *not* interchangeable
  with `forward_batch_size`, which counts requests only — the two are mutually exclusive
  and setting both is rejected.
- **A prompt group is never split across micros.** `forward_batch_size` chunks a
  frontier by row count and can cut a GRPO sibling group in half; recipes worked around
  that by hand-setting `forward_batch_size == samples_per_prompt`. `shape_bucket` fills
  micros with whole groups and fails when a group is larger than its bucket capacity.
- **Geometry comes from the driver-pinned `init_noise_latent_shape`.** That field is
  produced by the pipeline's own `latent_shape(model_config, sampling_spec)` classmethod,
  so the planner never re-derives crop/padding/bucket rules. A pipeline without
  `latent_shape()`, or a run with `DISABLE_DRIVER_XT`, has no effective geometry and is
  rejected with `scheduling_mode='off'` as the alternative.
- **The micro schedule must be provable rank-uniform.** Ranks sharing an FSDP all-gather
  must run the same number of block forwards in the same order. `shape_bucket` therefore
  refuses any frontier that mixes execution buckets or unequal group sizes, because the
  per-rank bucket mix would then decide the micro count. A rank-uniform schedule exchange
  is the missing contract, not a barrier.
- **One geometry per shard is a representation limit, not a planner limit.** The plan
  type and the CPU harness support several buckets, but `LatentSegment.latents` is one
  dense `[N_segs, K, ...]` tensor and `Part.sampling_params` is a shared field, so
  heterogeneous geometry cannot be merged back into one frontier today. Per-row geometry
  needs a per-row carrier in `Part` (and a ragged trajectory), and is the documented
  follow-up rather than something the planner may hide behind padding or `dtype=object`.
- **`Part.concat` takes shared fields from the first operand.** Micros are checked to
  still carry the frontier's own `sampling_params`/`role`/`harness_status` before the
  merge, and the restored row order is asserted against the original `sample_ids`.
