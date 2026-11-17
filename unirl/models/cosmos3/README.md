# Cosmos3 (NVIDIA omnimodal world model) — SFT, text-to-image RL and image-to-video RL

> **Where it fits:** one model package under [`unirl/models/`](../README.md). It serves the
> two droid100 SFT recipes (`examples/sft/cosmos3_droid100_{videopred,action_bc}.yaml`), and
> `Cosmos3Pipeline.generate` plus `Cosmos3DiffusionStage.replay` serve the FlowGRPO recipes
> `examples/diffusion/cosmos3/cosmos3_nano_trainside.yaml` (single-frame text-to-image) and
> `cosmos3_{nano,edge}_i2v_trainside.yaml` (image-to-video).

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
  `mixed_precision: false`): torch 2.10/cu128 on H20 rejects every bf16 `cublasGemmEx`,
  and `bundle.py` scopes further cuBLAS workarounds (contiguous mRoPE, 2-D-GEMM
  DomainAwareLinear, non-MATH SDPA, frozen-DTensor `full_tensor()` linears) to that
  runtime via `_needs_h20_cu128_workaround()`. Newer CUDA stacks may switch back to
  fp32 storage + bf16 mixed compute (the official Cosmos recipe shape).
- **`time_embedder` must stay fp32 in compute**, matching stock diffusers
  (`_keep_in_fp32_modules = ["time_embedder"]`). The full-fp32 SFT recipes satisfy
  this trivially. The bf16 RL recipe `cosmos3_nano_trainside` keeps it out of every FSDP
  group with `root_wrap: false` (see Gotchas); it is validated on H100 with torch 2.13/cu130,
  not on H20. Any other bf16 mixed-precision recipe needs its own fp32 exemption.
  Without one the outcome depends on the diffusers version: 0.39 casts only *after* the
  embedder, so the bf16 all-gather fails loudly on the fp32-sinusoid × bf16-linear matmul;
  0.40 casts the sinusoid to the embedder's parameter dtype first, so the embedder
  silently runs in bf16.
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

`Cosmos3Pipeline.generate` covers two modes: `num_frames=1` text-to-image, and
`num_frames = 4k + 1 > 1` image-to-video from an `Images` turn (one source image per prompt,
a manifest `(image, condition)` ref through `MultimodalRLDataSource`). Text-to-video is
rejected: it has no parity or log-prob evidence. `DiffusionSamplingParams.num_frames`
defaults to 16, so a recipe must set it. `Cosmos3DiffusionStage.diffuse`
loops over samples, because the transformer takes one packed sequence per call, and
`Cosmos3DiffusionStep.predict_noise` reuses `pack_joint_sequence`.
Latents are `[B, 48, T_gen, H/16, W/16]` for both checkpoints, with `T_gen = 1` for an
image and `(num_frames - 1) / 4` for a video. The sigma schedule comes
from `Cosmos3Pipeline.build_schedule_policy` (static flow shift, the required `shift`
constructor argument); the solver is the injected `StepStrategy`. The running latent is
stored in `trajectory_precision` between steps.

For image-to-video, `build_conditions` runs the source image through diffusers'
`_preprocess_conditioning_image` (aspect-preserving resize, centre crop) and the VAE, and
stores latent frame 0 as `Cosmos3Conditions.image_latent`. The solver state, the stored
trajectory and the SDE log-prob cover the generated latent frames only.
`Cosmos3DiffusionStep._velocity` prepends the clean frame for the packed forward
(`condition_frame_indexes=[0]`, no timestep embedding on it) and drops its velocity;
`decode` prepends it again and returns `Videos`. Replay reads `image_latent` back from
`Part.conditions`, so the image is encoded once per sample.

`Cosmos3DiffusionStage.replay` recomputes the SDE log-probs with gradients, one sample and
one step per packed forward (two with CFG), feeding the stored latents back through the
same `step_with_logp` call as rollout. `predict_noise_at_step` is the same forward without
the solver step.

## Image-to-video recipe: reward and evaluation

`cosmos3_nano_i2v_trainside` scores each video with `VideoPickScoreScorer`, the reward of
the WAN image-to-video recipes: PickScore of one frame against the prompt. It sets
`frame_selection: middle`, pixel frame `num_frames // 2`. The reward measures prompt
alignment and appearance of that frame. It does not measure motion, temporal consistency
or fidelity to the source image, and no local reward reads the source image.

Evaluation runs every `eval_interval` rollouts on `EVAL_DATA_PATH` (an image manifest like
`DATA_PATH`; the PickScore prompt lists have no images) with deterministic sampling
(`eval_eta` defaults to 0) and the same reward.

Scope: the recipe is smoke-tested only (8 prompts x 8 samples, 3 rollouts, 4x H100). The
default scale, the periodic evaluation and a reward trend are not validated. Middle-frame
PickScore had a within-group standard deviation of 2e-3 to 4e-3 (two prompts, four samples
each, 9 frames), so it is a smoke baseline, not evidence of video RL gains.

The two SDE steps per rollout are drawn from steps 10-12 of 16 (sigma 0.750, 0.694, 0.625 at
shift 5). That is the sigma band of the text-to-image recipe (0.75 to 0.58 at shift 3)
re-derived for the shift-5 grid, not a setting tuned for video.

## Cosmos3-Edge

`nvidia/Cosmos3-Edge` (3.4B parameters, 28 layers, hidden size 2048, vocabulary 131072) loads
through the same `Cosmos3OmniTransformer` class. Its decoder layers are still
`Cosmos3VLTextMoTDecoderLayer` and keep the `add_{q,k,v}_proj` / `to_add_out` names, so the
backend block class and LoRA targets are those of Nano, and its VAE and scheduler configs
equal Nano's. `cosmos3_edge_i2v_trainside.yaml` inherits `cosmos3_nano_i2v_trainside` and
overrides the checkpoint, the prompting flags and the sampler.

- **Edge needs diffusers>=0.40.** Its architecture is selected by `hidden_act: relu2`,
  `qk_norm_for_text: false` and `use_und_k_norm_for_gen: true`, which
  `Cosmos3OmniTransformer` reads from 0.40.0 on; 0.39.0 has none of them. 0.40.0 warns
  that it ignores `backbone_type` and `temporal_compression_factor` from Edge's config,
  which is expected.
- **Prompting and sampler follow the model card's image-to-video example**: no system
  prompt (`default_use_system_prompt: false` in its `model_index.json`), no resolution or
  duration sentence, flow shift 12, 20 steps, guidance 6.0, 832x480 at 24 fps.
  `Cosmos3JointStage` builds `Cosmos3OmniPipeline` from the bundle's parts rather than
  `from_pretrained`, so the pipeline defaults in `model_index.json` are not read; the
  bundle config carries `use_system_prompt`, `add_resolution_template` and
  `add_duration_template`.
- **Where the recipe differs from the card.** It generates 9 frames, the smoke scope of the
  Nano recipe; the card's example generates 121 and lists 50 to 150 as supported. The
  negative prompt is empty and the prompts are plain text; the card ships a JSON negative
  prompt and recommends JSON-upsampled prompts. The sigma grid is the static shift-12 grid
  starting at 1.0; Edge's `use_native_flow_schedule: true` starts upstream at
  `1 - 1/1000` before the shift, a difference below 3e-4 in sigma on this grid.
- **SDE steps.** The two SDE steps per rollout are drawn from steps 16-18 of 20 (sigma
  0.750, 0.679, 0.571 at shift 12), the sigma band of the Nano recipes re-derived for this
  grid, not a setting tuned for Edge.
- **Scope.** The recipe is smoke-tested only (8 prompts x 8 samples, 3 rollouts, 4x A100
  80GB, torch 2.13 / cu130; peak 22 GiB allocated per GPU, 30 GiB in `nvidia-smi` with the
  reward model). The default scale, the periodic evaluation, a reward trend, the card's
  121-frame length and H20 are not validated. The reward limits of the Nano
  image-to-video recipe apply unchanged.
- **`freeze_understanding` leaves `k_norm_und_for_gen` trainable on Edge.** Edge has no und
  `norm_q` / `norm_k` and adds `self_attn.k_norm_und_for_gen` per layer, which
  `_UND_PARAM_PATTERNS` does not match. No Edge SFT recipe ships.

## Gotchas

- **A stored image-to-video trajectory does not contain the source frame.**
  `LatentSegment.latents[:, -1]` holds the generated frames only; decoding it needs
  `Cosmos3Conditions.image_latent` (`Cosmos3Pipeline.decode(segment, image_latent)`).
  `latent_shape` returns the same generated-frame shape for driver-side noise. Keeping the
  clean frame out of the state is what lets `FlowSDEStrategy`, `replay`, and algorithms that
  noise the final latent and call `predict_noise_at_step` run without a frame mask: the
  kernel adds noise to every element it is given and averages the log-prob over all of them.
- **`Cosmos3OmniPipeline` steps the clean frame, this package does not.** Upstream zeroes
  the velocity on frame 0 and runs the scheduler on the whole latent, so its frame 0 stays
  equal to the image latent only up to solver rounding and that rounding feeds the next
  forward. Here frame 0 is the stored latent at every step. Measured on
  `nvidia/Cosmos3-Nano`, 832x480, 9 frames, 16 steps, guidance 4.0, shift 5, H100, with
  matching sigmas, noise, `UniPCStrategy` and floored timesteps: with fp32 weights the
  generated frames match upstream bit-for-bit once a step callback restores its frame 0,
  and differ from stock upstream by relative RMS 2.5e-6 to 5e-5 (its frame 0 drifts by up
  to 1.2e-6). With bf16 weights upstream also runs the solver in bf16 (next bullets): its
  frame 0 drifts by up to 0.056 against a latent RMS of 0.7, stock upstream differs by
  relative RMS 0.03 to 0.04, and the match is bit-for-bit again once upstream gets the
  restored frame and fp32 solver inputs.
- **The source image is encoded as one frame, in fp32.** Upstream encodes the image
  repeated over all `num_frames` and keeps latent frame 0; the Wan VAE is causal, and the
  two encodes are bit-identical on the real checkpoint (9 and 17 frames). Upstream with
  bf16 weights first rounds the preprocessed pixels to bf16, which moves the latent by a
  relative RMS of 0.1% to 0.2%; this package keeps fp32 pixels.
- **Video decode costs about 9 GiB per sample at 832x480** on top of the 31 GiB bf16
  model (fp32 Wan VAE, same for 9 and 17 frames); four videos decoded together peak at
  66 GiB. `decode` takes the whole `generate` batch, so the recipe's
  `forward_batch_size: 1` is what keeps it at one video.
- **VideoPickScore's default `frame_selection: first` leaves no reward spread here.** Decoded
  frame 0 is the VAE round trip of the source image for all samples of a prompt: its reward
  std within a prompt group was 2e-5 to 3e-5, against 2e-3 to 4e-3 for the middle frame
  (two prompts, four samples each, 9 frames).
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
  `Cosmos3OmniPipeline`), and >=0.40 for Cosmos3-Edge. The transformer `forward` return differs by version: 0.39 has
  no `return_dict` and returns the bare `(vision, sound, action)` tuple; 0.40 added
  `return_dict=True` and returns a `Cosmos3OmniTransformerOutput` (`sample` / `sound` /
  `action`) by default. That output is a `BaseOutput`, so tuple-unpacking it yields its
  non-`None` field names rather than the three lists. `Cosmos3JointStage.predict_velocity`
  reads both forms; passing `return_dict=False` instead would break 0.39.
- The H20 workaround patches in `bundle.py` install frozen diffusers-0.39 copies of
  `DomainAwareLinear.forward`, `dispatch_attention_fn`, and the rotary `forward`. Because
  the pin is an open `diffusers>=0.39`, each patch first fingerprints the upstream
  parameter list via `_require_signature` and fails closed on drift — otherwise a newer
  diffusers would be silently overridden by the stale copy. Re-verify the frozen bodies
  against the installed diffusers before extending the fingerprints on a version bump.
- The H20 SDPA wrapper repeats kv heads before dispatch: Cosmos3 attention is GQA
  (32 q / 8 kv heads on Nano, 16 / 8 on Edge), the fused Flash/Efficient kernels require
  equal head counts, and
  the MATH fallback is exactly the strided-batched GEMM that is broken on H20+cu128.
- Eval loss is noised velocity MSE at a per-sample deterministic (σ, ε) draw
  (`unirl.algorithms.sft.sample_eval_seed`) — comparable across runs, but not a
  sample-quality gate (use a later denoise path for PSNR/SSIM).
