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
- **Sleep returns released VAE blocks to the CUDA driver.** `MiniMaxH3Pipeline.sleep`
  calls `torch.cuda.empty_cache()` to return approximately 10 GB of unloaded VAE
  blocks for CUDA allocations outside the PyTorch allocator, such as cuBLAS and
  NCCL workspaces. GPU A/B measurements of peak memory and sleep time with and
  without `empty_cache()` are pending.
- **Checkpoint overrides default only when `None`.** `vae_ckpt_path` and
  `text_encoder_ckpt_path` otherwise reach their loaders unchanged, including
  invalid empty paths. Initial construction and wake use the same loaders;
  see the [residency contract](../../trainer/README.md#gotchas).
