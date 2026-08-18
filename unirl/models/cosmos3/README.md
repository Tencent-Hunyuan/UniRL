# Cosmos3 (NVIDIA omnimodal world model) — SFT support

> **Where it fits:** one model package under [`unirl/models/`](../README.md), currently
> train-loss only: `Cosmos3Pipeline.generate` is unimplemented; the package exists for the
> two droid100 SFT recipes (`examples/sft/cosmos3_droid100_{videopred,action_bc}.yaml`).

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
- **The shipped recipes run full fp32** (`model_precision`/`param_dtype: fp32`,
  `mixed_precision: false`): torch 2.10/cu128 on H20 rejects every bf16 `cublasGemmEx`,
  and `bundle.py` scopes further cuBLAS workarounds (contiguous mRoPE, 2-D-GEMM
  DomainAwareLinear, non-MATH SDPA, frozen-DTensor `full_tensor()` linears) to that
  runtime via `_needs_h20_cu128_workaround()`. Newer CUDA stacks may switch back to
  fp32 storage + bf16 mixed compute (the official Cosmos recipe shape).
- **`time_embedder` must stay fp32 in compute**, matching stock diffusers
  (`_keep_in_fp32_modules = ["time_embedder"]`; the forward casts only *after* the
  embedder). The shipped full-fp32 recipes satisfy this trivially. A future bf16
  mixed-precision recipe must land an fp32 exemption for it (e.g. its own FSDP group with
  an fp32 gather/reduce policy) together with that recipe — without one, the bf16
  all-gather fails loudly on the fp32-sinusoid × bf16-linear matmul rather than silently
  coarsening the timestep signal.
- **Timestep conditioning stays continuous fp32** (`sigma * num_train_timesteps`, no
  rounding), matching NVIDIA's official training (`cosmos-framework`
  `omni_mot_model.py`: `timesteps = sigmas * max_timestep`, fp32). diffusers-0.39
  *inference* truncates its scheduler grid to `np.int64` (`set_timesteps` casts every
  branch; the pipeline fills `t.item()`) — a ≤1-unit quirk the pretrained model
  tolerates. Do not "align" training to it: training σ are logit-normal draws that
  never land on the inference grid anyway, so flooring only quantizes the signal and
  biases the σ↔conditioning map by an average of −0.5 timestep units.
- **`meta_init_transformer: true`** (both recipes) builds the 16B transformer on the meta
  device and lets the backend load sharded weights after wrapping — the eager path would
  put the full ~64 GB fp32 model on every rank before sharding. The sharded loader reads
  local `*.safetensors` only, so a Hub-ID checkpoint path is resolved to a local
  snapshot (`snapshot_download(allow_patterns=["transformer/*"])`) before wrapping.
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

## Data prep (`python -m unirl.utils.prepare_droid100`)

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

## Gotchas

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
  `Cosmos3OmniPipeline`); its `forward` returns the bare `(vision, sound, action)` tuple
  and has no `return_dict`.
- The H20 workaround patches in `bundle.py` install frozen diffusers-0.39 copies of
  `DomainAwareLinear.forward`, `dispatch_attention_fn`, and the rotary `forward`. Because
  the pin is an open `diffusers>=0.39`, each patch first fingerprints the upstream
  parameter list via `_require_signature` and fails closed on drift — otherwise a newer
  diffusers would be silently overridden by the stale copy. Re-verify the frozen bodies
  against the installed diffusers before extending the fingerprints on a version bump.
- The H20 SDPA wrapper repeats kv heads before dispatch: Cosmos3 attention is GQA
  (32 q / 8 kv heads), the fused Flash/Efficient kernels require equal head counts, and
  the MATH fallback is exactly the strided-batched GEMM that is broken on H20+cu128.
- Eval loss is noised velocity MSE at a per-sample deterministic (σ, ε) draw
  (`unirl.algorithms.sft.sample_eval_seed`) — comparable across runs, but not a
  sample-quality gate (use a later denoise path for PSNR/SSIM).
