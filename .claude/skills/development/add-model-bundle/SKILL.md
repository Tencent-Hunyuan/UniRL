---
name: add-model-bundle
description: Add or update UniRL model package support. Use when adding diffusion or autoregressive model pipelines, model config dataclasses, Bundle/Pipeline/Stage/Conditions implementations, LoRA targets, FSDP wrapping hints, Sample/Part plumbing, or multimodal text/image/video/audio conditioning.
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

`EncodeStage[P, C]` / `EmbedStage[P, C]` convert primitives to conditions, `DiffusionStage[C]` / `ARStage[C]` produce segments, `DecodeStage[S, P]` decodes segments, and `Pipeline.generate(sample: Sample) -> Sample` fills pre-forked generation Parts.

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
11. In `pipeline.py`, implement `<Model>Pipeline(Pipeline)` with `from_config(...)` and `generate(sample)`. Read raw inputs through `sample.conditioning()`, read sampling params from the typed generation Part, require `params.sigmas` for diffusion, call stages in order, and fill that Part with `segment`, modality-keyed `primitives`, and `conditions`.
12. Add `latent_shape(cls, *, model_config, sampling_spec)` when the driver should author a deterministic initial-noise recipe on the generation Part for group noise or resume behavior.
13. Update the package `__init__.py` to import and export public symbols from `config.py`, `bundle.py`, `pipeline.py`, and condition classes so importing `unirl.models.<model_name>` re-exports them.
14. Add at least one recipe YAML under `examples/<domain>/` (the v2 config dir, grouped by trainer domain) and document external checkpoint requirements there or in launcher environment docs.

## Wiring Touchpoints

Model packages are wired into recipes by `_target_` dotpath (no ConfigStore):

- Define `<Model>PipelineConfig` as a plain `@dataclass` in `config.py`.
- Recipes set `bundle._target_: ...<Model>Bundle.from_config` with a nested `config._target_: ...<Model>PipelineConfig`; the worker walker constructs them.
- Add new shared condition types under `unirl/types/conditions/` only when existing slots cannot express the semantics; export them from `unirl/types/conditions/__init__.py`.
- Add or update rollout-engine model-family enums only when the model is served through an engine that explicitly enumerates families, such as SGLang or vLLM-Omni configs.

Keep package-specific logic under `unirl/models/<model_name>/`. Put only cross-model protocols or reusable condition abstractions under `unirl/models/types/` or `unirl/types/conditions/`.

## Meta-Init (avoid the per-rank full-model load)

A bundle can build its trainable transformer on the **meta** device instead of eager `from_pretrained`, so the backend materializes + loads weights *after* sharding — avoiding the per-rank full-model GPU/host spike that OOMs large models. Both `FSDPBackend` and `VeOmniBackend` honor a single contract; opt in per recipe with `meta_init_transformer: true` on the bundle config (default `false` → unchanged eager path).

Single-transformer bundles (the common case) branch in `from_config`:

```python
if config.meta_init_transformer:
    transformer_config = <Class>.load_config(path, subfolder="transformer")   # diffusers
    transformer, meta_init_state = build_meta_init_transformer(               # unirl.models.types.meta_init
        lambda: <Class>.from_config(transformer_config),
        dtype=dtype,
    )
else:
    transformer = <Class>.from_pretrained(path, subfolder="transformer", torch_dtype=dtype).to(device, dtype=dtype)
...
bundle = cls(...)
if config.meta_init_transformer:
    bundle._transformer_weights_path = os.path.join(path, "transformer")      # diffusion layout
    bundle._meta_init_state = meta_init_state
return bundle
```

- `build_meta_init_transformer` builds under `accelerate.init_empty_weights(include_buffers=False)`, captures init-computed non-persistent buffers/plain tensor attributes, dtype-casts (metadata-only on meta), and stamps `init_weights` to a no-op. Stash its returned state on `bundle._meta_init_state`; the backend restores it after the sharded weight load, including across Ray actor serialization.
- Stash `_transformer_weights_path` = the safetensors dir the backend reads via `load_sharded` (`unirl/train/backend/sharded_load.py`): `<ckpt>/transformer` for diffusers-layout models; the **checkpoint root** (`path`) for AR/VL models loaded through `AutoModelForCausalLM` (no subfolder).
- AR/VL bundles use the same helper with an `AutoModelForCausalLM.from_config(...)` (qwen3) or `ModelClass(cfg)` (qwen_vl) factory. Structural setup that does not touch weights (`gradient_checkpointing_enable`, `requires_grad_(False)` for a frozen vision tower) runs on both builds and persists through `to_empty` + load.

Per-architecture init-computed state that `to_empty` destroys is handled by the shared capture/restore path. `build_meta_init_transformer` deliberately keeps buffers and `__dict__` tensors real on CPU while parameters are meta, and raises if a captured tensor is unexpectedly still meta. This covers plain-tensor rope tables (Qwen-Image `QwenEmbedRope.pos_freqs`) and non-persistent sincos buffers (SD3 `PatchEmbed.pos_embed`) without bespoke rebuild/deferred-stamp helpers.
- params the checkpoint omits (FLUX.2-klein guidance embedder): zero-init them post-load via a deferred op keyed on checkpoint-absent names — `to_empty` leaves them as garbage (not meta), so an `is_meta`-gated fix won't catch them.

Always confirm parity on a GPU pod: the meta build must load weights byte-identical to the eager path, on both backends.

Composite trainables with *embedded* frozen aux (only `hunyuan_image3` today — `transformer.vae` / `transformer.vision_model` live inside the meta-built wrapper) are the exception, and add two hooks on top of the contract:

- `trainable_module()` returns the bare decoder (`transformer.model`). The backend resolves the module to wrap/optimize/checkpoint via `resolve_trainable_module(bundle, trainable_attr)` (`unirl/train/backend/base.py`), which prefers this method and otherwise falls back to the named attr. Handing the backend the single decoder — not the heterogeneous composite — keeps the frozen aux *outside* the wrap (on meta until materialized, off the optimizer/checkpoint), which is what lets the composite run under VeOmni (its `parallelize` root-shards + whole-root-`to_empty`s its input) and makes the `weight_sync_param_name_prefix` (`"model."`) resolve correctly.
- a self-contained `materialize(device, with_aux=())` allocates + DCP-loads the decoder, the always-resident diffusion heads, and the opt-in vae/vit in one collective (mixed DTensor decoder + plain heads/aux); the backend calls it when no `_transformer_weights_path` is stashed.

Single-transformer bundles whose aux are separate eager modules need none of that.

## Conditions And Field Kinds

`<Model>Conditions(Batch)` is the typed container passed to diffusion or AR stages and serialized through `Part.conditions`. It owns conditioning slots only. Latents live in `LatentSegment`; sigma schedules live in the generation Part's `DiffusionSamplingParams.sigmas` and segment metadata.

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

- Use `schedule=params.sigmas` from the generation Part; diffusion pipelines should raise if it is `None`.
- Do not build sigma schedules inside the pipeline or stage. Hosting engines pin schedules onto the Sample before calling `generate(sample)`.
- Validate schedule length against the requested step count.
- Initialize latents from request-provided `initial_latents` when the package supports deterministic driver-side noise; otherwise call the repository noise helper used by the closest template.
- Store trajectories at `unirl.types.sampling.compute_trajectory_positions(...)` plus the final clean latent position, with stored latents in `trajectory_precision` and log-probs in `logprob_precision`.
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

## What To Verify

Follow the `CLAUDE.md` verification-harness rule: run small, uncommitted CPU
harnesses with fakes or monkeypatches and quote the commands and results in the
PR's Test Plan.

- **Conditions**: `from_dict` / `to_dict` round trips, optional slots, wrong-typed slot errors, and missing required slot errors.
- **Diffusion step**: CFG batching (uncond+cond in one transformer call), timestep scaling, masks, vision kwargs, and private transformer kwargs, using fake transformers.
- **Pipeline wiring**: construct fake stages, call `generate(sample)`, and assert lineage preservation, generation-Part conditions/segment/primitives, and pinned-sigma validation (`generate` must raise when `params.sigmas is None`).
- **AR models**: generation and replay token/log-prob alignment on a tiny fake model.
- **Recipe/config wiring**: instantiate added configs and recipes with lightweight fakes and verify their `_target_` paths resolve.

## Review Before Finishing

- `<Model>PipelineConfig` is a plain `@dataclass`; recipes reference it (and `<Model>Pipeline.from_config`) by `_target_`.
- The package `__init__.py` re-exports the config / pipeline classes.
- `Pipeline.generate(sample)` validates required conditioning primitives, typed generation params, negative prompt batch sizes, and pinned sigmas for diffusion.
- Filled generation Parts use canonical primitive keys such as `"image"`, `"video"`, `"audio"`, or `"text"`, and include conditions and a segment when available.
- `<Model>Conditions.from_dict` and `to_dict` are symmetric and fail loudly for wrong or missing required slots.
- Per-sample tensors use `FieldKind.CONCAT`; shared metadata uses `FieldKind.SHARED`.
- The diffusion stage owns loop bookkeeping, trajectory storage, replay, and precision casts; the diffusion step owns transformer calls and CFG math.
- The sigma schedule is consumed from the generation Part's params; it is not rebuilt in the model package.
- Bundle loading normalizes dtype/device, freezes auxiliary modules, and keeps trainable module naming compatible with `weight_sync_param_name_prefix`.
- LoRA target modules are explicit for production models; `None` is only used deliberately.
- Recipe YAML exists under `examples/<domain>/` (the v2 config dir, grouped by trainer domain) and documents required checkpoints or environment variables.
