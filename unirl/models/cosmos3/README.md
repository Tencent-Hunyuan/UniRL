# Cosmos3 (NVIDIA omnimodal world model) — SFT and text-to-image RL

> **Where it fits:** one model package under [`unirl/models/`](../README.md). It serves the
> two droid100 SFT recipes (`examples/sft/cosmos3_droid100_{videopred,action_bc}.yaml`), and
> `Cosmos3Pipeline.generate` plus `Cosmos3DiffusionStage.replay` serve the single-frame
> text-to-image FlowGRPO recipe `examples/diffusion/cosmos3/cosmos3_nano_trainside.yaml`.

## What it is

Cosmos3-Nano is a 16B Mixture-of-Transformers: a causal "understanding" (und) text stream
and a bidirectional "generation" (gen) stream share each `Cosmos3VLTextMoTDecoderLayer`
with disjoint parameter sets. SFT here trains the gen stream (velocity prediction) and
freezes the und stream by default (`freeze_understanding`). The und set is matched by the
full-name regexes in `bundle.py` (`embed_tokens`, `lm_head`, root `norm`, und
attention/MLP/layernorms); everything else — `add_{q,k,v}_proj` / `to_add_out`,
`mlp_moe_gen.*`, `*_moe_gen` norms, `proj_in`/`proj_out`, `time_embedder`, and the
action/audio modality heads — is the generation stream and stays trainable.

Training mirrors `Cosmos3OmniPipeline.__call__` steps 2–5 by calling the diffusers
pipeline's own helpers (`tokenize_prompt`, `_prepare_text_segment`,
`_prepare_vision_segment`, `_prepare_action_segment`, `_encode_video`), then replaces the
denoising loop with a single noised forward:

    sigma ~ p(t), flow-shift-warped exactly like set_timesteps
    x_t   = (1 - sigma) * x0 + sigma * eps     # noisy frames only; condition frames stay
                                               # clean x0 and get no timestep embedding
    v*    = eps - x0                           # flow_prediction convention:
                                               # x0_pred = sample - sigma * v
    loss  = MSE over noisy tokens

The transformer consumes ONE packed sequence per call (no batch dim); a training
micro-batch is one sample, with gradient accumulation across samples. The algorithm
(`unirl.algorithms.cosmos3_sft.Cosmos3JointFlowMatchSFT`) owns the sigma draw via the
shared `unirl.algorithms.sft.draw_shifted_sigma` (per-sample flow shift from the official
short-edge tier table 256/480/720 → 3/5/10; short edges above the 720p bin stay on the
720 tier rather than silently falling back to the checkpoint scheduler).

## Precision & loading

- **Uniform `master_precision` storage**: upcasting only trainable params would mix
  bf16/fp32 inside every MoT layer (the und stream is frozen), which FSDP2 rejects.
- **The SFT recipes run full fp32** (`model_precision`/`param_dtype: fp32`,
  `mixed_precision: false`): full-parameter SFT keeps the root FSDP group enabled
  (`root_wrap: true`) because `time_embedder`, `proj_in`, and `proj_out` are trainable,
  so setting `param_dtype: bf16` without a separate fp32 FSDP group for `time_embedder`
  all-gathers it in bf16 and doubles initial video-prediction velocity MSE (0.09986 →
  0.20077 on 8x H20).
- **`time_embedder` must stay fp32 in compute**, matching stock diffusers
  (`_keep_in_fp32_modules = ["time_embedder"]`). The full-fp32 SFT recipes satisfy
  this trivially. The bf16 RL recipe `cosmos3_nano_trainside` keeps it out of every FSDP
  group with `root_wrap: false` (see Gotchas) and is validated on H100, H200, and H20
  with `torch==2.13.0+cu130`. Any other bf16 mixed-precision recipe needs its own fp32
  exemption. Without one the outcome depends on the diffusers version: 0.39 casts only
  *after* the embedder, so the bf16 all-gather fails loudly on the fp32-sinusoid ×
  bf16-linear matmul; 0.40 casts the sinusoid to the embedder's parameter dtype first,
  so the embedder silently runs in bf16.
- **Timestep conditioning stays continuous fp32** (`sigma * num_train_timesteps`, no
  rounding), matching NVIDIA's official training (`cosmos-framework`
  `omni_mot_model.py`: `timesteps = sigmas * max_timestep`, fp32). diffusers-0.39
  *inference* truncates its scheduler grid to `np.int64` (`set_timesteps` casts every
  branch; the pipeline fills `t.item()`) — a ≤1-unit quirk the pretrained model
  tolerates. Do not "align" training to it: training σ are logit-normal draws that
  never land on the inference grid anyway, so flooring only quantizes the signal and
  biases the σ↔conditioning map by an average of −0.5 timestep units.
- **`meta_init_transformer: true`** (both SFT recipes) builds the 16B transformer on the meta
  device and lets the backend load sharded weights after wrapping — the eager path would
  put the full ~64 GB fp32 model on every rank before sharding. The sharded loader reads
  local `*.safetensors` only, so the shared meta-init resolver downloads and validates
  a Hub checkpoint's transformer shards before wrapping.
- WanVAE was trained with amp off; encode/decode run in the VAE's own dtype
  (`vae_precision`, fp32) and it stays a small eager module.

## Action BC (policy mode)

Embodiment domain for the DomainAwareLinear action heads: DROID/Franka single-arm =
`droid_lerobot` (domain id from diffusers' `_EMBODIMENT_TO_DOMAIN_ID`). The canonical
Cosmos3 width for that domain is 10 (9D EE pose + 1D gripper); debug datasets that carry
another layout (droid_100's 7-D flattened action) may override `raw_action_dim` — fine for
finetuning, but the head no longer matches the base checkpoint's pretrained action
semantics. Policy mode noises the full action chunk (no clean action conditioning);
channels ≥ `raw_action_dim` are pinned to zero in sample and target, matching inference.

## Data prep (`python datasets/droid100/prepare_droid100.py`)

Emits, under `--root` (default `datasets/droid100_debug`):

    frames/<sample_id>.pt    uint8 [T, 3, H, W]      (decoded once; training never
                                                      touches AV1/mp4)
    actions/<sample_id>.pt   float32 [T-1, D_raw]    (z-normalized action chunk)
    manifest.jsonl           one record per training window
    eval_manifest.jsonl      held-out episodes
    stats.json               action mean/std used for normalization

A window of `T` frames pairs with `T-1` action transitions (Cosmos3 policy convention:
`action_chunk_size + 1` frames). The default 192×320 canvas is exactly the Cosmos3 action
resolution-tier-256 bin (ratio 5:3), so training-time encoding and tier-based action
inference see the same canvas. Manifest media entries use `role="target"` with
`modality="video"` (frames) and `modality="action"` (chunk) — `"action"` is a supported
`MediaRef` modality whose URI only this package's track builder consumes.

## Generation

`Cosmos3Pipeline.generate` covers `num_frames=1` only; `DiffusionSamplingParams.num_frames`
defaults to 16, so a recipe must set it. `Cosmos3DiffusionStage.diffuse`
loops over samples, because the transformer takes one packed sequence per call, and
`Cosmos3DiffusionStep.predict_noise` reuses `pack_joint_sequence` with no condition frame.
Latents are `[B, 48, 1, H/16, W/16]` for `nvidia/Cosmos3-Nano`. The sigma schedule comes
from `Cosmos3Pipeline.build_schedule_policy` (static flow shift, the required `shift`
constructor argument); the solver is the injected `StepStrategy`. The running latent is
stored in `trajectory_precision` between steps.

`Cosmos3DiffusionStage.replay` recomputes the SDE log-probs with gradients, one sample and
one step per packed forward (two with CFG), feeding the stored latents back through the
same `step_with_logp` call as rollout. `predict_noise_at_step` is the same forward without
the solver step.

## Gotchas

- **CFG runs two transformer calls per step, not one batched call.** The cond and uncond
  prompts have different token lengths and the forward takes a single unbatched sequence,
  so `predict_noise` runs them one after the other and combines
  `uncond + guidance_scale * (cond - uncond)`, like `Cosmos3OmniPipeline`.
  `guidance_scale == 1.0` skips the uncond call.
- **`build_schedule_policy` ignores the checkpoint's `scheduler_config.json`.** The shipped
  config has `use_karras_sigmas: true`, under which diffusers drops `flow_shift`; official
  examples override it with `flow_shift=10, use_karras_sigmas=False`. Pin sigmas on the
  Sample to reproduce either one.
- **Generation feeds continuous fp32 `sigma * 1000` timesteps; diffusers inference
  truncates them to int64.** With matching sigmas, initial latents and `UniPCStrategy`,
  flooring the timestep reproduces `Cosmos3OmniPipeline` latents bit-for-bit with fp32
  weights (`nvidia/Cosmos3-Nano`, 832x480, 16 steps, guidance 4.0, H100,
  `trajectory_precision="fp32"`).
- **Each solver step runs in fp32 here and in the transformer dtype in diffusers 0.40.**
  `Cosmos3OmniPipeline` casts the sampling latents to `transformer.dtype`, so with bf16
  weights its UniPC arithmetic is bf16; `StepStrategy.denoise` promotes to fp32. With bf16
  weights the final latents therefore differ from stock diffusers (relative RMS 0.05 to
  0.17 over six prompt/seed pairs) and match bit-for-bit once the diffusers scheduler step
  is given fp32 inputs. Both bf16 variants sit equally far from the fp32-weight result.
- **The bundle loads `time_embedder` in fp32 for any `master_precision`.**
  `from_pretrained` keeps it fp32 (`_keep_in_fp32_modules`), so the eager load moves the
  transformer to the device without a dtype, and the meta-init load passes `keep_in_fp32`.
  `Cosmos3DiffusionStage.diffuse` raises if the stored dtype is not fp32. That check reads
  the stored parameter, so it does not catch an FSDP `mp_policy` that gathers the embedder
  in bf16 for the forward (see Precision & loading).
- **The RL recipe needs `root_wrap: false`.**
  `time_embedder`, `proj_in` and `proj_out` live outside the decoder layers; a root shard
  group would gather them in `param_dtype` (bf16) and the timestep MLP would run in bf16
  silently. Every LoRA target (`add_{q,k,v}_proj`, `to_add_out`) is inside a
  `Cosmos3VLTextMoTDecoderLayer`, so no trainable parameter is left outside the wrap.
- The und-param regexes are anchored with trailing `\.` on purpose: `to_out` must not
  swallow `to_add_out`, and `norm.` must not swallow `norm_moe_gen.`. Touch
  `_UND_PARAM_PATTERNS` only with the freeze-count log line as your regression check.
- `prepare_droid100` z-normalizes every action channel uniformly, including the
  near-binary gripper column. Fine for a debug BC run; faithful Policy-DROID reproduction
  should leave the gripper unnormalized (or min-max it) instead of z-scoring it.
- The per-model track builder exists because the record → (conditions, segment) mapping is
  inherently Cosmos3-specific (packed prompt tokens + action chunk through the joint
  stage's own helpers); it still reuses the generic `_media_uris` / `_require_local_uri`
  from `unirl/train/sft/track_builder.py`, whose builders stay model-agnostic.
- diffusers>=0.39 is required (first release with `Cosmos3OmniTransformer` /
  `Cosmos3OmniPipeline`). The transformer `forward` return differs by version: 0.39 has
  no `return_dict` and returns the bare `(vision, sound, action)` tuple; 0.40 added
  `return_dict=True` and returns a `Cosmos3OmniTransformerOutput` (`sample` / `sound` /
  `action`) by default. That output is a `BaseOutput`, so tuple-unpacking it yields its
  non-`None` field names rather than the three lists. `Cosmos3JointStage.predict_velocity`
  reads both forms; passing `return_dict=False` instead would break 0.39.
- Eval loss is noised velocity MSE at a per-sample deterministic (σ, ε) draw
  (`unirl.algorithms.sft.sample_eval_seed`) — comparable across runs, but not a
  sample-quality gate (use a later denoise path for PSNR/SSIM).
