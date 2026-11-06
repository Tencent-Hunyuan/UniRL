# WAN 2.1

> **Where it fits:** text/image conditioning → video diffusion → VAE decode.
> Full map: [`../README.md`](../README.md).

## Gotchas

- **Driver-authored initial latents also pin SDE noise to sample identity.**
  `WAN21Pipeline.generate` passes the frontier's sample IDs and sampling seed
  to the existing per-sample, per-step noise helper, so restarting or changing
  rollout micro-batches preserves the noise. Direct `diffuse` calls without
  `denoise_seed_keys`, and requests with `disable_driver_xt=true`, retain global
  RNG behavior. This does not make GPU backward kernels deterministic.
