# Migrating rollout code to `Sample` and `Part`

`RolloutReq`, `RolloutResp`, and `RolloutTrack` have been removed. The rollout
boundary is now an endomorphism:

```text
Sample -> Sample
```

A request is a `Sample` containing input `Part`s and an empty generated `Part`.
The engine returns the same lineage with that generated part filled. This is a
breaking API change: there are no compatibility aliases for the retired types.

## Mental model

- `Sample.parts` is one ordered lineage chain. The previous part is the parent;
  there is no track-name dictionary or arbitrary track graph.
- `Part.input(...)` creates the root input. Additional input turns use
  `part.input_child(...)`.
- `sample.fork(branch, sampling_params=...)` appends an empty generation shell.
  `Part.sampling_params is not None` is what identifies a generated part.
- `part.fill(...)` records the generated segment, decoded primitives, and replay
  conditions. `sample.with_filled_frontier(...)` is the usual frontier shortcut.
- Lineage is encoded in `Part.sample_ids`: a fork appends a numeric path segment,
  such as `prompt-0/0`. `Part.group_ids` derives immediate-parent groups from
  those paths; `Sample.root_group_ids(i)` derives root-prompt groups.
- Raw conditioning remains on ancestor `Part.primitives`. `Sample.conditioning()`
  and the role-aware `turns()`, `text_conditioning()`, and
  `vision_conditioning()` views align those ancestors to the frontier. Encoded
  conditions used for replay live on the generated `Part.conditions`.

## Construction mapping

| Retired API | Current API |
| --- | --- |
| `RolloutReq(...)` | `Sample.request(Part.input(...), ...)` |
| `RolloutReq.make_root_track(..., branch=N)` | `sample.fork(N, sampling_params=params)` |
| `RolloutTrack.fork_track(..., branch=M)` | Another `sample.fork(M, sampling_params=params)`; use `Part.fork` only when constructing the child part directly |
| `RolloutResp(tracks=...)` | The `Sample` already contains every input and generated `Part`; `Sample(parts=[...])` is the lower-level constructor |

`fork` returns a new object, so retain its result. The branch count is explicit;
it is not inferred from `sampling_params.samples_per_prompt`.

## Field mapping

### `RolloutReq`

| Retired field | Current location or behavior |
| --- | --- |
| `sample_ids` | Root input `Part.sample_ids` |
| `group_ids` | Derived from lineage with `Part.group_ids` or `Sample.root_group_ids(part_index)`; it is no longer stored independently |
| `primitives` | Input `Part.primitives`, keyed by canonical modality (`text`, `image`, `video`, `audio`, or sparse URI container `media`) |
| `request_conditions` | No single request-level replacement. Keep raw inputs on ancestor `Part.primitives`; the generated part stores the encoded replay inputs in `Part.conditions`. Precomputed diffusion `initial_latents` belong on the generated `LatentSegment.initial_latents` |
| `sampling_params` | One typed `Part.sampling_params` per generated stage. Composed rollouts use one fork per stage |
| `stage_config` | Root input `Part.control` |
| `sigmas` | `DiffusionSamplingParams.sigmas` on the diffusion generation part |
| `metadata` | Root input `Part.metadata`; use `Sample.root_metadata(part_index)` to align it with descendants |
| `init_noise_group_ids` | Derived from generated `sample_ids` / `group_ids` by `NoiseRecipe.from_sample(...)`; deterministic eval may override them on the gen `Part` |
| `init_noise_latent_shape` | `DiffusionSamplingParams.init_noise_latent_shape` |

### `RolloutResp` and `RolloutTrack`

| Retired field | Current location or behavior |
| --- | --- |
| `RolloutResp.tracks` | Ordered `Sample.parts` |
| `RolloutResp.reward_compute_s` | `Sample.reward_compute_s` |
| `RolloutTrack.sample_ids` | `Part.sample_ids` |
| `parent_ids` | Derived with `Part.group_ids` (or `parent_id(sample_id)` for an individual ID) |
| `parent_track` | Structural: `Sample.parts[i - 1]` is the parent of `parts[i]` |
| `conditions` | `Part.conditions` |
| `segment` | `Part.segment` |
| `decoded` | `Part.primitives`; the modality map can hold multiple jointly generated outputs |
| `media_preview` | `Part.media_preview` |
| `rewards`, `component_rewards`, `advantages`, `status` | Same-named `Part` fields |

`Part.primitive_metadata` is the new shared metadata carrier for a decoded
modality, for example `{"audio": {"sample_rate": 48000}}`. Per-example dataset
and reward metadata remains in `Part.metadata`.

## Helper mapping

| Retired helper or pattern | Current API |
| --- | --- |
| `resp.tracks["ar"]` / `resp.tracks["image"]` | `sample.gen_part(ARSamplingParams)` / `sample.gen_part(DiffusionSamplingParams)` when that type occurs exactly once |
| Generation shell appended by the latest `fork` | `sample.frontier_gen_part(ParamsType)`; validates that the final Part is generated and has the requested type |
| Optional unique-stage lookup | `sample.gen_part_or_none(ParamsType)`; returns `None` for no match and raises for duplicates |
| Track index needed for write-back | `sample.gen_part_index(ParamsType)`, then `sample.with_parts(new_parts)` |
| `resp.root_track()` | The chain head is `sample.parts[0]`; the first generated stage is `sample.gen_parts()[0]` |
| `RolloutResp.concat(items)` | `Sample.concat(items)` |
| `RolloutTrack.concat(items)` | `Part.concat(items)` |
| `resp.split()` | `sample.split()`; each result contains one root prompt's complete lineage |
| `track.group_ids` | `part.group_ids` |
| Root-group lookup / `_root_group_per_sample(...)` | `sample.root_group_ids(part_index)` |
| `track.compute_advantages(...)` | `part.compute_advantages(...)`; use `group_layer=0` for root-prompt grouping |
| `resp.compute_track_advantages(...)` | Locate the part, call `Part.compute_advantages(...)`, and write the returned part back with `Sample.with_parts(...)` |
| `resp.propagate_rewards(op=...)` | `sample.propagate_rewards(op=...)` |
| `track.balance_shards(...)` | `part.balance_shards(...)` |
| `_track_with_field(...)` | `dataclasses.replace(part, field=value)`, followed by `replace_frontier(...)` or `with_parts(...)` |
| `metadata_only()` | Removed; there is no public compatibility helper |
| `tracks_with_segment_types(...)` | Iterate `sample.parts` and inspect `part.segment` explicitly |

Track names no longer select stages. Sampling-parameter type identifies a stage
only when that type occurs once: `gen_part(Type)`, `gen_part_index(Type)`, and
`gen_part_or_none(Type)` reject duplicate matches and report their Part indices.
Generation paths should use `frontier_gen_part(Type)`, because `fork` appends the
shell to fill. For a multi-stage generator, validate its complete trailing
structure (for example, `[..., AR, diffusion]`) instead of scanning historical
turns by type. If a trajectory intentionally contains repeated generated stages,
iterate `sample.gen_parts()` or address their known lineage positions explicitly.

## Conceptual before and after

Before:

```python
from unirl.types.rollout_req import RolloutReq
from unirl.types.primitives import Texts
from unirl.types.sampling import ARSamplingParams

ar = ARSamplingParams(samples_per_prompt=2)
request = RolloutReq(
    sample_ids=["prompt-0"],
    group_ids=["prompt-0"],
    primitives={"text": Texts(texts=["Write a caption"])},
    sampling_params={"ar": ar},
    stage_config={"ar": {"system_instruction": "Be concise"}},
)
response = engine.generate(request)
text_track = response.tracks["ar"]
```

After:

```python
from unirl.types import Part, Sample
from unirl.types.primitives import Texts
from unirl.types.sampling import ARSamplingParams

ar = ARSamplingParams(samples_per_prompt=2)
root = Part.input(
    ["prompt-0"],
    primitives={"text": Texts(texts=["Write a caption"])},
    control={"ar": {"system_instruction": "Be concise"}},
)
request = Sample.request(root).fork(ar.samples_per_prompt, sampling_params=ar)
response = engine.generate(request)
text_part = response.gen_part(ARSamplingParams)
```

For a composed AR-to-diffusion rollout, append one fork per stage:

```python
request = (
    Sample.request(root)
    .fork(ar.samples_per_prompt, sampling_params=ar)
    .fork(diffusion.samples_per_prompt, sampling_params=diffusion)
)
```

Do not manually repeat ancestor prompts or images during a fork. Their
frontier-aligned values are recovered from the sample-ID lineage by the
conditioning views.

## Prompt media (`MediaRefs`) — breaking note

URI-backed prompt inputs for Qwen3-Omni (and any consumer that opts into the
sparse media channel) now live under:

```python
Part.primitives["media"] = MediaRefs.from_rows([[MediaRef("video", "prompt", uri), ...], ...])
```

Use `Sample.prompt_media_refs()` to read that channel. Do **not** look for
prompt video/audio/image URIs under `primitives["video"|"image"|"audio"]`.

Those decoded modality keys remain the contract for **condition** media
(diffusion / V2V / image-edit): loaded `Images` / `Videos` tensors with
`role="condition"`.

`build_omni_messages` still accepts deprecated URI-backed `Videos.from_uris`
and normalizes them to video `MediaRef` rows; migrate callers to `MediaRefs`.
Waveform `Audios` and decoded frame `Videos` / `Images` stay rejected for
Omni prompts.
