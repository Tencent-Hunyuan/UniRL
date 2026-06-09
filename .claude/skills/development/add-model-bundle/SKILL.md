---
name: add-model-bundle
description: Add or update UniRL model package support. Use when adding diffusion or autoregressive model pipelines, model config dataclasses, Bundle/Pipeline/Stage/Conditions implementations, LoRA targets, FSDP wrapping hints, RolloutReq/RolloutResp plumbing, or multimodal text/image/video conditioning.
---

# Add Model Bundle

## Start Here

When adding a diffusion or autoregressive model, first inspect `unirl/models/README.md`, `unirl/models/types/`, and the closest package under `unirl/models/`:

- `unirl/models/sd3/`: image diffusion with text embeddings, CFG, VAE decode, and driver-provided initial latents.
- `unirl/models/wan21/`: text/image-to-video diffusion with image latent and CLIP-vision conditioning.
- `unirl/models/wan22/`: text-to-video diffusion.
- `unirl/models/flux2_klein/` and `unirl/models/qwen_image/`: image diffusion families with model-specific text/token conditioning.
- `unirl/models/hunyuan_video15/`: video diffusion with multiple text/vision encoders.
- `unirl/models/hunyuan_image3/`: mixed AR and diffusion topology for multimodal tasks.
- `unirl/models/qwen3/`: pure causal-LM AR package.
- `unirl/models/qwen_vl/` and `unirl/models/pe/`: additional package-specific patterns when relevant.

The current architecture is a typed pipeline:

`EncodeStage[P, C]` / `EmbedStage[P, C]` convert primitives to conditions, `DiffusionStage[C]` / `ARStage[C]` produce segments, `DecodeStage[S, P]` decodes segments, and `Pipeline.generate(req: RolloutReq) -> RolloutResp` composes the stages.

`Bundle` in `unirl/models/types/bundle.py` is an intentionally empty `Remote` subclass. Concrete bundles are plain weight holders; LoRA injection, FSDP wrapping, adapter switching, offload, and autocast lifecycle are owned outside the bundle.

## Implementation Checklist

1. Create `unirl/models/<model_name>/` rather than a single file. Typical files are `__init__.py`, `config.py`, `bundle.py`, `pipeline.py`, `conditions.py`, `diffusion.py` or `ar.py`, plus `text_embed.py`, `vae.py`, and vision helpers as needed.
2. In `config.py`, define `<Model>PipelineConfig` as a plain `@dataclass`. Recipes reference it by `_target_: unirl.models.<model_name>.<Model>PipelineConfig` (nested under the bundle/pipeline `config:` block) — no registration.
3. Include config fields that match the package's real needs: checkpoint paths, `model_precision`, auxiliary dtype fields, runtime `device`, `autocast_precision`, `trajectory_precision`, `logprob_precision`, schedule knobs such as `shift` for FlowMatch diffusion, `weight_sync_param_name_prefix`, `use_lora`, and `lora_target_modules`.
4. In `bundle.py`, implement `<Model>Bundle` as a plain class with `from_config(config)`. Load transformer, VAE, text encoders, vision encoders, tokenizers, processors, and schedulers as needed. Use `parse_torch_dtype(..., field_name=...)` for dtype fields, place the trainable module on the requested device and dtype, and freeze auxiliary modules with `requires_grad_(False)`.
5. In `conditions.py`, implement `<Model>Conditions(Batch)` with typed condition slots and `from_dict(d)` / `to_dict()`. Validate required slots, reject wrong types with actionable errors, and omit `None` optional slots from the outgoing dict.
6. Add embed/encode stages for inputs: `EmbedStage[Texts, TextEmbedCondition]`, `EncodeStage[Images, ImageLatentCondition]`, or model-specific variants. Keep tokenization, chat templates, text encoder fusion, image preprocessing, and upstream-compatible negative prompt defaults in these stages or in the pipeline that calls them.
7. For diffusion models, add `<Model>DiffusionStep(DiffusionStep[<Model>Bundle, <Model>Conditions])`. By local convention, it should expose `predict_noise(...)` for per-step transformer invocation, CFG batching, timestep scaling, condition concat, masks, and private third-party kwargs. Delegate SDE math to the supplied `StepStrategy`.
8. Add `<Model>DiffusionStage(DiffusionStage[<Model>Conditions])`. It owns latent initialization when supported by the package, the diffusion loop, trajectory storage, replay, precision policy, and `trainable_module()` when training-side injection needs the trainable root. Declare `_no_split_modules` on the stage when diffusers modules need FSDP wrapping hints.
9. For AR models, add `<Model>ARStep` and `<Model>ARStage(ARStage[<Model>Conditions])` instead of diffusion step/stage classes. Follow `unirl/models/qwen3/ar.py` for packed `TextSegment` generation and replay.
10. In `vae.py` or equivalent, implement `DecodeStage[LatentSegment, Images | Videos]` and any required `EncodeStage[Images | Videos, ImageLatentCondition]`. Apply the model's VAE scale, shift, dtype, layout, frame, and clamp conventions.
11. In `pipeline.py`, implement `<Model>Pipeline(Pipeline)` with `from_config(...)` and `generate(req)`. Validate required primitives and sampling params, require `req.sigmas` for diffusion pipelines, call stages in order, and return `RolloutResp(tracks={...})` with `RolloutTrack(sample_ids, parent_ids, conditions, segment, decoded)`.
12. Add `latent_shape(cls, *, model_config, sampling_spec)` when the driver should precompute `request_conditions["initial_latents"]` for deterministic group noise or resume behavior.
13. Update the package `__init__.py` to import and export public symbols from `config.py`, `bundle.py`, `pipeline.py`, and condition classes so importing `unirl.models.<model_name>` re-exports them.
14. Add at least one recipe YAML under `examples/<domain>/` (the v2 config dir, grouped by trainer domain) and document external checkpoint requirements there or in launcher environment docs.

## Wiring Touchpoints

Model packages are wired into recipes by `_target_` dotpath (no ConfigStore):

- Define `<Model>PipelineConfig` as a plain `@dataclass` in `config.py`.
- Recipes set `bundle._target_: ...<Model>Bundle.from_config` with a nested `config._target_: ...<Model>PipelineConfig`; the worker walker constructs them.
- Add new shared condition types under `unirl/types/conditions/` only when existing slots cannot express the semantics; export them from `unirl/types/conditions/__init__.py`.
- Add or update rollout-engine model-family enums only when the model is served through an engine that explicitly enumerates families, such as SGLang or vLLM-Omni configs.

Keep package-specific logic under `unirl/models/<model_name>/`. Put only cross-model protocols or reusable condition abstractions under `unirl/models/types/` or `unirl/types/conditions/`.

## Conditions And Field Kinds

`<Model>Conditions(Batch)` is the typed container passed to diffusion or AR stages and serialized through `RolloutResp.tracks[<slot>].conditions`. It owns conditioning slots only. Latents live in `LatentSegment`; sigma schedules live in `RolloutReq.sigmas` and segment metadata.

Use field kinds from `unirl/distributed/tensor/batch.py`:

- `field(kind=FieldKind.CONCAT, transport=True, default=None)`: per-sample, batch-aligned slots such as text, negative text, image latents, image embeddings, and masks.
- `field(kind=FieldKind.SHARED, transport=False, default=None)` or `shared_field(...)`: batch-shared metadata such as static position grids or spatial shape lists.
- `concat_field(...)` and `shared_field(...)` are available helper aliases, but the generic `field(...)` form is the most explicit when transport metadata matters.

Reuse existing condition slot types before adding new ones:

- `TextEmbedCondition(embeds, pooled, attn_mask)`: frozen text-encoder hidden states, optional pooled head, and optional attention mask.
- `TextTokenCondition(input_ids, attention_mask)`: token IDs and masks for models whose transformer owns token embeddings.
- `ImageEmbedCondition(embeds, attn_mask, spatial_shapes)`: CLIP/SigLIP/ViT-style image features.
- `ImageLatentCondition(latents)`: VAE-encoded image or video conditioning latents.
- `FusedMultimodalCondition(...)`: interleaved text/image token payloads for omni-style bundles.

Keep slot names semantic and flat: `text` / `negative_text`, `image_latent`, `image_embed`, `prompt`, etc. Do not reuse a slot name with different meaning, and do not hide CFG branches inside another condition object.

## Negative Prompt And CFG

CFG belongs in the diffusion step, with the pipeline and embed stages preparing positive and negative conditions:

- The pipeline validates prompt and negative prompt batch sizes.
- If upstream behavior requires CFG negatives and none were supplied, the pipeline should create the upstream-compatible empty negative primitive before embedding, such as `""` for SD3 or the model-specific canonical empty string for Qwen-style pipelines.
- `<Model>DiffusionStep.predict_noise(...)` should batch unconditional and conditional branches, run one transformer call, chunk outputs, and combine `uncond + guidance_scale * (cond - uncond)`.
- If `negative_text` is absent but `guidance_scale > 1.0`, either raise a clear error or use the package's established fallback, such as zero-init negative embeddings in SD3. Match the model's existing or upstream behavior explicitly.

## DiffusionStage Rules

`<Model>DiffusionStage.diffuse(...)` owns the rollout loop and `LatentSegment` assembly:

- Use `schedule=req.sigmas` passed by the pipeline; diffusion pipelines should raise if `req.sigmas is None`.
- Do not build sigma schedules inside the pipeline or stage. Hosting engines pin schedules with `unirl.sde.runtime.ensure_req_sigmas(req, policy)` before calling `generate(req)`.
- Validate schedule length against the requested step count.
- Initialize latents from request-provided `initial_latents` when the package supports deterministic driver-side noise; otherwise call the repository noise helper used by the closest template.
- Store trajectories at `compute_trajectory_positions(...)` plus the final clean latent position, with stored latents in `trajectory_precision` and log-probs in `logprob_precision`.
- Keep direct transformer calls inside `<Model>DiffusionStep.predict_noise(...)`. The stage should call `self.step.step(...)` or `self.step.step_with_logp(...)`.
- Implement `replay(...)` to recompute log-probs and previous-sample means from stored `LatentSegment` transitions for training.
- Implement `predict_noise_at_step(conditions, *, sample, sigma, params)` for forward-process algorithms such as DiffusionNFT; it should delegate to the same `predict_noise(...)` path so CFG and guidance behavior match `diffuse(...)` and `replay(...)`.
- Expose `trainable_module()` and `_no_split_modules` on the stage when training-side injection or wrapping needs the trainable root or FSDP hints.

## ARStage Rules

For causal-LM or multimodal AR paths:

- Use `ARStage[<Model>Conditions]` and `ARStep` from `unirl/models/types/ar.py`.
- `autoregress(...)` should produce a packed `TextSegment` with generated tokens, masks or lengths, and per-token log-probs aligned with replay.
- `replay(...)` should recompute log-probs for stored rollout tokens with the same tokenization and attention-mask semantics.
- Expose `trainable_module()` when training-side LoRA/FSDP injection needs the wrapped transformer root.
- Use `ARSamplingParams` for common generation controls and a package-specific params dataclass only for model-specific knobs.

## Tests To Add

Prefer small CPU tests with fakes or monkeypatches rather than loading real checkpoints:

- `tests/models/test_<model>_conditions.py`: `from_dict` / `to_dict` round trips, optional slots, wrong-typed slot errors, and missing required slot errors. Follow `tests/models/test_sd3_conditions.py` and `tests/models/test_hunyuan_image3_conditions.py`.
- `tests/models/test_<model>_diffusion_step_<topic>.py`: CFG batching, timestep scaling, masks, vision kwargs, and private transformer kwargs using fakes. Follow the WAN21 diffusion-step tests.
- `tests/models/test_<model>_pipeline.py` when pipeline wiring changed: construct fake stages, call `generate(req)`, and assert `RolloutResp.tracks[...]` keys, conditions, segment, decoded payloads, and `req.sigmas` validation.
- AR models: add or adapt `tests/test_qwen3_ar_stage.py`-style tests for generation and replay alignment.
- Shared condition or stage behavior: update `tests/types/test_conditions.py` or `tests/models/test_stages.py` only when shared contracts changed.
- Config registration and instantiation: use the patterns in `tests/config/test_config_registration.py` and `tests/config/test_config_instantiate.py` when adding Hydra config behavior.

Run targeted tests first, then broaden if shared condition, stage, or pipeline behavior changed:

```bash
pytest tests/models/test_<model>_conditions.py tests/models/test_<model>_diffusion_step_*.py tests/models/test_stages.py tests/types/test_conditions.py
```

Adjust the command to real files before running. If the model is AR-only or pipeline-only, replace diffusion-step tests with the relevant AR or pipeline tests.

## Review Before Finishing

- `<Model>PipelineConfig` is a plain `@dataclass`; recipes reference it (and `<Model>Pipeline.from_config`) by `_target_`.
- The package `__init__.py` re-exports the config / pipeline classes.
- `Pipeline.generate(req)` validates required primitives, stage params, negative prompt batch sizes, and `req.sigmas` for diffusion.
- `RolloutResp.tracks` use the intended output key, such as `"image"`, `"video"`, or `"text"`, and include `conditions`, `segment`, and decoded primitives when available.
- `<Model>Conditions.from_dict` and `to_dict` are symmetric and fail loudly for wrong or missing required slots.
- Per-sample tensors use `FieldKind.CONCAT`; shared metadata uses `FieldKind.SHARED`.
- The diffusion stage owns loop bookkeeping, trajectory storage, replay, and precision casts; the diffusion step owns transformer calls and CFG math.
- The sigma schedule is consumed from `req.sigmas`; it is not rebuilt in the model package.
- Bundle loading normalizes dtype/device, freezes auxiliary modules, and keeps trainable module naming compatible with `weight_sync_param_name_prefix`.
- LoRA target modules are explicit for production models; `None` is only used deliberately.
- Recipe YAML exists under `examples/<domain>/` (the v2 config dir, grouped by trainer domain) and documents required checkpoints or environment variables.
