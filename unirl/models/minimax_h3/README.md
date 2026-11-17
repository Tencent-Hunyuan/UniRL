# MiniMax-H3 block training

> **Where it fits:** joint video/audio rollout and replay. In: text conditions and noisy latents.
> Out: video/audio transitions. Full map: [models](../README.md).

## How it works

`full_finetune_blocks: true` trains the original weights of `transformer_blocks`.
The text refiner, input/output projections and timestep modules remain frozen.
Text encoding and both VAEs stay frozen. The default `false` preserves existing LoRA construction.

The `minimax_h3_t2va_trainside_blocks` recipe inherits Flow-GRPO settings and removes LoRA.
It is a construction probe, not a validated learning recipe. Learning rate, Adam epsilon
and clipping need measurements before a learning comparison.

## Gotchas

- **Block training is not whole-transformer training.** Report the frozen peripheral weights
  and trainable parameter count when comparing against LoRA.
- **The original mixed precision layout is retained.** Keep `root_wrap: false` and
  `cast_forward_inputs: false`; frozen timestep and projection modules remain outside FSDP groups.
- **The measured 8-H100 configuration fails during backward.** With Torch 2.13,
  expandable allocator segments, full sharding, fp32 masters, resident VAEs and 10 replay
  steps at 256x384 with 124 frames,
  the original checkpoint loads and generates but does not reach the first Adam update.
  More shards or offloading need validation; learning remains unverified.
