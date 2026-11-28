# MiniMax-H3 block training

> **Where it fits:** joint video/audio rollout and replay. In: text conditions and noisy latents.
> Out: video/audio transitions. Full map: [models](../README.md).

## How it works

`full_finetune_blocks: true` trains the original weights of `transformer_blocks` (32.3B trainable parameters across 50 `MiniMaxH3TransformerBlock`s, 600 tensors).
The text refiner, input/output projections and timestep modules remain frozen (38 peripheral tensors).
Text encoding and both VAEs stay frozen. The default `false` preserves existing LoRA construction.

`minimax_h3_t2va_trainside_blocks` inherits Flow-GRPO settings from `minimax_h3_t2va_trainside`, removes LoRA, bounds replay graph residency with `algorithm.timestep_chunk_size: 1`, and sets `backend.fsdp_cfg.checkpoint_format: dcp` across `num_devices: 16` (`2×8` GPUs).

## Gotchas

- **Block training is not whole-transformer training.** Report the frozen peripheral weights
  and trainable parameter count when comparing against LoRA.
- **The original mixed precision layout is retained.** Keep `root_wrap: false` and
  `cast_forward_inputs: false`; frozen timestep and projection modules remain outside FSDP sharding while the parameterless root wrap shares a single FSDP communication context across all 50 blocks so `fp32` reduce-scatter buffers are recycled during backward.
- **Bound replay residency and use DCP checkpoints.** Set `algorithm.timestep_chunk_size: 1` so replay does not hold all 10 SDE steps in one autograd graph concurrently, and keep `backend.fsdp_cfg.checkpoint_format: dcp` (`torch` gathers 32.3B weights and `fp32` AdamW states onto rank 0).
- **Validated hardware envelope (256x384, 124 frames, 10 CPSSDE steps, Torch 2.13 + cu130).**
  - `2×8` (`16` GPUs): steady-state peaks at `55.74 GiB` allocated (`58.06 GiB` reserved) with `activation_offload: false` and `54.99 GiB` allocated (`57.71 GiB` reserved) with `activation_offload: true`, fitting both `2×8 H20 (95 GiB)` and `2×8 H100/A100 (80 GiB)`; 16-rank `dcp` save (`275 GB`) and resume are verified, and all 600 trainable tensors maintain finite non-zero gradients above `adam_epsilon: 1e-12` (median RMS `5.1e-9`).
  - `1×8` (`8` GPUs): single-node `1×8 H20 (95 GiB)` completes all 10 SDE steps at `92.56 GiB` allocated (`93.60 GiB` reserved) when `backend.fsdp_cfg.activation_offload: true` is enabled; `1×8 80 GiB` nodes do not fit sharded `bf16` weights + `fp32` master/grad/AdamW states plus activations without CPU optimizer offload.
