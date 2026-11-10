# MiniMax-H3

> **Where it fits:** trainside text-to-video/audio rollout and replay over one shared bundle.
> In: prompts and sampling parameters. Out: video, audio, and diffusion trajectories.
> Full map: [`../README.md`](../README.md).

## Gotchas

- **H3 supports engine-grouped per-sample SDE requests.**
  `MiniMaxH3DiffusionStage.supports_order_independent_sde_noise` is true because
  `MiniMaxH3Pipeline.generate` passes sample ids as `denoise_seed_keys` to both
  stream generators. Regrouping therefore preserves each sample's step noise.
  The [FlashGRPO recipe](../../../examples/diffusion/minimax_h3/minimax_h3_t2va_flashgrpo.yaml)
  enables this path; its shared engine requirements live in
  [`trainer/README.md`](../../trainer/README.md#gotchas).
- **Sleep returns released auxiliary blocks to the CUDA driver.** With
  `rollout_resident=false`, `MiniMaxH3Pipeline.sleep` releases CUDA video/audio
  VAEs and any CUDA text encoder after decoding and inline `reward_stack` scoring,
  without retaining per-rank CPU weight copies. `vae_components_on_cpu` and
  `aux_components_on_cpu` independently control placement: CPU modules stay resident,
  so CPU VAEs do not prevent a CUDA text encoder from being released. Before the next
  rollout or eval, `wake_up` reloads released modules on their original devices;
  the shared trainable transformer stays untouched. Sleep then calls
  `torch.cuda.empty_cache()`, after the module references are dropped and whenever
  a module is recorded as released: freed blocks otherwise stay in PyTorch's
  caching allocator, where allocations made outside it (the cuBLAS handle, NCCL
  buffers) cannot use them.
- **H3 auxiliary checkpoints and construction config must stay unchanged between wakes.**
  `load_minimax_h3_vae` and `load_minimax_h3_text_encoder` in
  [`bundle.py`](bundle.py) serve both initial load and wake, preserving checkpoint
  paths, dtype, eval mode, and freezing; `from_bundle` must receive the bundle's
  construction config so reloading reproduces its weights.
- **H3's audio VAE has unregistered `weight_norm` tensors.** Moving only registered
  parameters/buffers to meta leaves those allocations alive; sleep drops every
  bundle, decode-stage, and text-embed-stage module reference, and wake rebinds
  all of them, including the text encoder's VAE lease.
- **Checkpoint overrides default only when `None`.** `vae_ckpt_path` and
  `text_encoder_ckpt_path` otherwise reach their loaders unchanged, including
  invalid empty paths.
- **A batched rollout shares one packed layout and one SDE step list.**
  `MiniMaxH3DiffusionStage.generate` accepts `[B, rows, channels]` video/audio
  latents with the same geometry, text length after trimming right-padding, and
  `sde_indices` for every row. Geometry and schedules are request-shared;
  mismatched latent shapes or text lengths are rejected before the forward.
  The existing transformer batch axis keeps attention documents independent;
  it does not concatenate samples into one sequence.
- **Flash-GRPO engine-mode training batches support multiple prompts per rank.**
  Set trainer `batch_size` to a positive multiple of `rollout dp_size`.
  With `rollout.forward_batch_size > 1`, trainside splits per-sample SDE requests
  at each prompt boundary, pins that prompt's `sde_indices`, then chunks its rows.
  Keep `stack.micro_batch_size=1` for training. Eval may have multiple
  prompts per rank; `samples_per_prompt % forward_batch_size == 0` keeps chunks
  within one prompt. See [`../../trainer/README.md`](../../trainer/README.md).
- **Batched forwards need not be bit-identical to serial forwards.**
  GEMM batch geometry can change rounding. B=1 keeps its original operations;
  each row's denoising noise remains keyed by sample ID, step and base seed,
  with a separate audio salt. Supply one seed key per row for order-independent
  noise. The engine retains each row's `[before, after, final]` video/audio
  slots and joint SDE log-probability.
- **Both VAEs decode one row at a time and return CPU media.**
  This keeps decode activations at B=1 even when denoising is batched. The audio
  VAE's batch axis is the two stereo channels of a single sample, not rollout
  rows. Decoded media retain input row order; CPU media storage grows with B.
