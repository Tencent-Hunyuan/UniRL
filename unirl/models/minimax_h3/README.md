# MiniMax-H3 — joint video + stereo audio diffusion package

> **Where it fits:** one model package under [`unirl/models/`](../README.md). `MiniMaxH3DiffusionStage`
> packs `[text | keyframe conditions | audio | video]` rows per forward (`packing.py`).

## Gotchas

- **Keyframe condition rows travel from rollout to replay; they are never recomputed.**
  `MiniMaxH3KeyframeCondition.rows` holds the fp32 rows noised at rollout, and
  `MiniMaxH3DiffusionStep.predict_noise` prepends exactly those in `generate`, `replay` and
  `predict_noise_at_step`. diffusers 0.40.0 and vLLM-Omni 0.28 differ in condition noise and
  second-keyframe resize, so rows recomputed on the train side silently break the replay log-prob ratio.
- **Condition rows run at the target video timestep.** `row_timestep_plan` gives them `1 - sigma`; the
  released model runs them at `max(1 - sigma, 0.999)`. Until #391 step 2 lands, the transformer is told the
  keyframe rows are as noisy as the target, so an fl2va rollout is not the released model's FL2VA conditioning
  and no recipe may enable fl2va. `DELETE-WHEN:` #391 step 2 lands.
- **A batch with `keyframes` needs an entry for every sample.** A t2va sample's entry is anchors `()`,
  `torch.empty(0, 96)` fp32 rows and all-text tags, which `build_layout` turns into exactly the t2va layout.
  A producer emits an entry for every sample or for none; `MiniMaxH3Conditions` rejects a `keyframes` sample
  count that differs from `text`, and `Batch.concat` (#582) rejects a `Part.conditions` merge in which a
  non-empty sample lacks the `keyframes` key, which would otherwise shift every later sample onto the next
  sample's entry.
- **`text_token_tags` are stored unpadded.** `trim_text_padding` must run before `build_layout`, which
  checks the tag length against the trimmed text.
