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
  `vae_components_on_cpu=false`, `MiniMaxH3Pipeline.sleep` releases CUDA video/audio
  VAEs and any CUDA text encoder (`aux_components_on_cpu=false`), then calls
  `torch.cuda.empty_cache()` so allocations outside the PyTorch allocator can use
  the freed memory for the cuBLAS handle and NCCL buffers. Removing this call
  caused `CUBLAS_STATUS_ALLOC_FAILED`; keep it after dropping the module references.
- **Checkpoint overrides default only when `None`.** `vae_ckpt_path` and
  `text_encoder_ckpt_path` otherwise reach their loaders unchanged, including
  invalid empty paths. Initial construction and wake use the same loaders;
  see the [residency contract](../../trainer/README.md#gotchas).
