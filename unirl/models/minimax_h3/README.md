# MiniMax-H3

> **Where it fits:** text-to-video/audio rollout and replay over one shared bundle.
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
  the shared trainable transformer stays untouched. Sleep calls
  `torch.cuda.empty_cache()` so allocations outside the PyTorch allocator can use
  the freed memory for the cuBLAS handle and NCCL buffers. Removing this call
  caused `CUBLAS_STATUS_ALLOC_FAILED`; keep it after dropping the module references.
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
