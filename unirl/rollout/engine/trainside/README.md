# Trainside rollout engine

> **Where it fits:** the engine layer of the *rollout* step, direct-sampling mode.
> In: one DP shard of a request `Sample`. Out: a `Sample` whose frontier gen `Part`
> is filled by the train actor's own `Pipeline`.
> Full map: [`../README.md`](../README.md).

*`TrainsideRolloutEngine.generate` calls `pipeline.generate` in-process under
`no_grad`, optionally in row chunks of at most `forward_batch_size`, and
concatenates the chunks back in row order.*

## What it is

[`engine.py`](engine.py) holds `TrainsideRolloutEngine`, the in-process sampler:
the train actor's `Pipeline` is the rollout model, so there is no weight sync and
no `sync:` block. [`plan.py`](plan.py) holds `validate_group_chunks`, the optional
prompt-group boundary check. Reward scoring and group advantages happen downstream
of this package.

## Why it exists

Direct sampling runs the rollout forward on the same GPUs as the trainable model
and its resident encoders, so one `pipeline.generate` over a whole DP shard can
exceed device memory. `forward_batch_size` bounds that peak. Group reward and
advantage only need each prompt group's rows back in their original order, so a
chunk boundary may split a group.

## How it works

- **`forward_batch_size` caps the rows in each `pipeline.generate` call.** When it
  is set and the frontier `Part` has more rows, `generate` slices the frontier into
  consecutive `forward_batch_size`-row chunks, runs each against the shared input
  Parts, and `Part.concat`s the results in the original row order before group
  reward and advantage are computed.
- **`validate_group_boundaries` checks alignment and never changes chunk sizes.**
  The default is `false`. When `true` and chunking applies,
  `validate_group_chunks(gen.group_ids, forward_batch_size)` raises `ValueError`
  unless every prompt group either fits inside one chunk, or starts on a chunk
  boundary and spans a whole number of chunks.

| Group size | `forward_batch_size` | `validate_group_boundaries: true` |
| --- | --- | --- |
| 16 | 4, 8 or 1 | accepted |
| 24 | 1 | accepted |
| 16 | 6 | rejected: 16 is not a multiple of 6 |

```yaml
rollout:
  _target_: unirl.rollout.engine.trainside.engine.TrainsideRolloutEngine
  forward_batch_size: 4
  validate_group_boundaries: true
```

## Gotchas

- **The boundary check requires contiguous prompt groups.** A group id that
  reappears after another group raises `ValueError`, even when each run is aligned.
- **The boundary check is skipped when no chunking happens.** With
  `forward_batch_size` unset, or a frontier of at most `forward_batch_size` rows,
  `generate` makes one call and `validate_group_boundaries` has no effect.
- **`forward_batch_size` rejects a `Sample` with more than one gen `Part`.**
  Chunking would re-run the interior stages once per chunk; composed PE recipes
  chunk inside the SGLang diffusion sub-engine instead.
- **Changing `forward_batch_size` silently reorders global-RNG draws.** Per-step SDE
  noise drawn from the global torch RNG (SD3 passes no generator to
  `strategy.denoise`) is consumed in a different order, so the same seed yields
  different samples. Keep `forward_batch_size` fixed when reproducing a seeded run.
  Pipelines with per-sample step generators (`make_denoise_step_generators`, e.g.
  HunyuanVideo 1.5) are not affected.
- **Each forward's low-precision numerics depend on its batch size.** Outputs can
  differ in the last bits across `forward_batch_size` values even with identical
  noise.
- **With `disable_driver_xt: true` and `init_same_noise: false`, SD3's `x_T` also
  depends on chunking.** By default the trainer pins `init_noise_latent_shape` for
  pipelines that define `latent_shape` (SD3 does), and
  [`NoiseRecipe`](../../../types/noise_recipe.py) seeds each row's `x_T` from the
  seed and a per-row key (sample id, group id under `init_same_noise: true`, or the
  eval prompt key), independent of chunk boundaries. Without the recipe, SD3 falls
  back to `generate_latents`, whose `init_same_noise: false` path draws `x_T` from
  the global RNG.
