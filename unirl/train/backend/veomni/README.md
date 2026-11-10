# VeOmni Training Backend (`unirl.train.backend.veomni`)

> **Where it fits:** the distributed training backend under `unirl/train/` that wraps
> trainable model modules with FSDP2, Ulysses Sequence Parallelism (`sp/`), and Expert
> Parallelism (`ep/`). Full map: [`../../../README.md`](../../../README.md).

## What it is

`unirl/train/backend/veomni/` integrates ByteDance-Seed's VeOmni distributed runtime
into UniRL's `TrainStack`, providing composable FSDP2 (`wrap.py`), Ulysses SP (`sp/`),
and MoE Expert Parallelism (`ep/`) across autoregressive, diffusion, and unified
multimodal models (`Qwen3-MoE`, `Qwen3.5-MoE`, `HunyuanImage-3.0`).

## How it works

- **MoE Router Replay (`ep/router_replay.py`)** — `MoERouterReplayController` records
  top-k expert indices during reference/rollout forward and replays them during actor
  policy updates and activation-checkpointing recomputation while re-gathering live
  gate scores so router gradients remain differentiable.
- **Shared Expert + EP All-to-All overlap (`ep/models/hi3.py`)** — `FusedHunyuanMoE`
  dispatches `shared_mlp` on a side CUDA stream concurrently with routed-expert EP
  All-to-All dispatch/GEMM/combine collectives.

## Gotchas

- **Activation Checkpointing under Router Replay.** During `loss.backward()`, AC
  re-runs each MoE block's forward pass. `MoERouterReplayController.on_router_forward`
  detects `torch.utils.checkpoint` recomputation frames and pops the microbatch index
  from its per-router recompute stack so multi-microbatch gradient accumulation replays
  each microbatch's own routing indices during both forward and AC backward.
- **Shared Expert + EP All-to-All stream overlap.** In `FusedHunyuanMoE.forward`, when
  `overlap_shared_expert=True` under `ep_size > 1`, `shared_mlp` runs on a dedicated
  side CUDA stream synchronized via CUDA events so shared-expert GEMMs overlap with
  routed-expert EP dispatch/combine collectives.
