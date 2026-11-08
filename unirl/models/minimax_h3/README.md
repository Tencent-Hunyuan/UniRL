# MiniMax-H3

> **Where it fits:** text-to-video/audio rollout and replay over one shared bundle.
> Full map: [`../README.md`](../README.md).

## Gotchas

- **H3 supports engine-grouped per-sample SDE requests.**
  `MiniMaxH3DiffusionStage.supports_order_independent_sde_noise` is true because
  `MiniMaxH3Pipeline.generate` passes sample ids as `denoise_seed_keys` to both
  stream generators. Regrouping therefore preserves each sample's step noise.
  The [FlashGRPO recipe](../../../examples/diffusion/minimax_h3/minimax_h3_t2va_flashgrpo.yaml)
  enables this path; its shared engine requirements live in
  [`trainer/README.md`](../../trainer/README.md#gotchas).
