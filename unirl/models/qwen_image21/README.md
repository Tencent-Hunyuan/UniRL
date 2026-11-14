# Qwen-Image-2.1 (text-to-image)

> **Where it fits:** one model package under [`unirl/models/`](../README.md); trainside
> rollout only, recipe [`qwen_image21_trainside`](../../../examples/diffusion/qwen_image21/qwen_image21_trainside.yaml).

## What it is

[Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) is a 7B, 32-layer
single-stream DiT. Qwen3-VL 8B encodes the prompt; its hidden states are projected and
concatenated with the target-image latent tokens into one sequence under a block-causal
mask (text causal, each image block bidirectional). Text tokens are modulated from
`t = 0` (`causal_condition`), the target tokens from the sampled `t`. The VAE is a
64-channel RGBA autoencoder with 16x spatial compression and the transformer consumes
latents unpatched (`patch_size=1`).

The transformer and VAE are vendored from diffusers (see
[`vendor/VENDOR_COMMIT.txt`](vendor/VENDOR_COMMIT.txt)) until a diffusers release ships
them. The package reuses `QwenImageConditions` and the Qwen-Image rollout / replay loop;
`QwenImage21DiffusionStep.predict_noise` and the geometry constants are what differ.

Scope: text-to-image, no CFG (the model is sampled with `guidance_scale=1.0`), no prefix
KV cache, no sequence parallelism, no external rollout engine.

## Gotchas

- **The target RoPE position depends on the padded text length.** `QwenImage21Rope`
  counts pad positions as text, so the target block's frame offset equals the padded
  prompt length. Batching a prompt with a longer one would change its velocity, and
  rollout (many prompts) and replay (micro-batch) would disagree. `predict_noise` runs
  one transformer call per distinct prompt length, slicing each group to its exact
  length, so no padding reaches the transformer and `encoder_hidden_states_mask` stays
  `None`.
- **The timestep is rounded like upstream.** diffusers passes `t = sigma * 1000` cast to
  the model dtype and then divides by 1000; `predict_noise` reproduces that rounding
  instead of casting `sigma` directly, so the transformer sees exactly upstream's inputs.
  Stock diffusers then computes the Euler `dt * v` product in bf16 (a 0-dim fp32 `dt`
  does not promote a bf16 tensor) while UniRL's SDE kernels use fp32, so a bf16 ODE
  rollout differs from `QwenImage21Pipeline(use_kv_cache=False)` by rounding only
  (~40 dB PSNR at 1024x1024 / 40 steps, the same size as toggling upstream's KV cache);
  with that product upcast in the upstream scheduler the two are bit-identical.
- **Text features are taken before Qwen3-VL's final RMSNorm.** The text-embed stage
  installs a forward hook that makes the norm return its input (as upstream does, since
  transformers >= 5 ties `hidden_states[-1]` to the normalized output) and reads
  `last_hidden_state` from the bare `Qwen3VLModel`, skipping the vocabulary-sized
  `lm_head` that the full `Qwen3VLForConditionalGeneration` forward would compute.
  The prompt is a raw template string fed to the processor (not `apply_chat_template`,
  which tokenizes differently), left-padded, and the system-prefix tokens are dropped.
- **The sigma policy is pinned in `build_schedule_policy`.**
  `FlowMatchSchedulePolicy.from_pretrained` derives `vae_scale_factor` from
  `block_out_channels`, which this VAE config lacks, so it would fall back to 8 and
  over-count the sequence length 4x (`mu` 1.31 instead of 0.69 at 1024x1024).
- **`height` / `width` must be multiples of 32; `generate` rejects anything else.** One
  encoder image slot covers a 2x2 group of latent tokens, so the latent grid must be
  even. Upstream silently floors the size first. `FlowMatchSchedulePolicy` counts
  `H/16 x W/16` tokens without that floor, so an unaligned size would get `mu` for a
  grid one row or column larger than the one being denoised.
- **The VAE decodes RGBA**; the decode stage composites alpha over white so rewards see
  RGB.
