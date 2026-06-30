#!/usr/bin/env python
"""Multi-GPU FSDP2 replay microbench: serial vs batched-step replay.

Measures the *distributed* win that single-GPU microbenches miss: under FSDP2
with ``reshard_after_forward=True`` each replay forward re-all-gathers the full
sharded transformer, so the serial loop pays **S all-gathers** per ``replay()``
while batched-step replay pays **1**. This wraps the real transformer with the
training-path :func:`unirl.train.backend.fsdp.wrap.fsdp_wrap` (per-block
``fully_shard``) across ``nproc_per_node`` GPUs and times both paths.

  torchrun --nproc_per_node=2 scripts/profiling/fsdp_replay_microbench.py --model sd3
  torchrun --nproc_per_node=2 scripts/profiling/fsdp_replay_microbench.py --model qwen_image

Env: B, T, SDE_INDICES, ITERS, WARMUP, RESHARD(1/0), SD3_DIR, QWEN_IMAGE_DIR.
Correctness (ratio=1) is reported too; for clean timing pause the GPU occupier.
"""

from __future__ import annotations

import argparse
import os
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist


def _rank0(*a):
    if int(os.environ.get("RANK", "0")) == 0:
        print(*a, flush=True)


def _occupier(action: str) -> None:
    """rank-0 only: SIGSTOP/SIGCONT the /tmp/gpu_occupy.py busy-loop for a short
    clean-timing window (loading runs with it live; only the timed loop pauses,
    so we stay well under the external watchdog's restart threshold)."""
    if int(os.environ.get("RANK", "0")) != 0:
        return
    import signal as _sig

    sig = _sig.SIGSTOP if action == "stop" else _sig.SIGCONT
    try:
        out = os.popen("pgrep -f gpu_occupy.py").read().split()
        for pid in out:
            try:
                os.kill(int(pid), sig)
            except (ProcessLookupError, PermissionError):
                pass
    except Exception:
        pass


def _build(model_name: str, dev: torch.device):
    """Return (stage_serial, stage_batched, conds, params, x0, schedule, info)."""
    from unirl.sde.kernels import FlowSDEStrategy
    from unirl.sde.runtime import get_sigma_schedule
    from unirl.train.backend.fsdp.wrap import fsdp_wrap
    from unirl.types.conditions import TextEmbedCondition
    from unirl.types.sampling import DiffusionSamplingParams

    B = int(os.environ.get("B", "4"))
    T = int(os.environ.get("T", "6"))
    sde = [int(i) for i in os.environ.get("SDE_INDICES", "0,2,4").split(",")]
    reshard = os.environ.get("RESHARD", "1") == "1"
    dt = torch.bfloat16

    if model_name == "sd3":
        from diffusers import SD3Transformer2DModel

        from unirl.models.sd3.conditions import SD3Conditions
        from unirl.models.sd3.diffusion import SD3DiffusionStage, SD3DiffusionStep

        d = os.environ.get("SD3_DIR", "/data/models/stable-diffusion-3.5-medium")
        tf = SD3Transformer2DModel.from_pretrained(f"{d}/transformer", torch_dtype=dt).to(dev)
        blocks = ("JointTransformerBlock",)
        hw, ch = 64, 16
        g = torch.Generator(device=dev).manual_seed(0)
        conds = SD3Conditions(
            text=TextEmbedCondition(
                embeds=torch.randn(B, 333, 4096, device=dev, dtype=dt, generator=g),
                pooled=torch.randn(B, 2048, device=dev, dtype=dt, generator=g),
            ),
            negative_text=None,
        )
        StageC, StepC = SD3DiffusionStage, SD3DiffusionStep
        npar = sum(p.numel() for p in tf.parameters())
    elif model_name == "qwen_image":
        from diffusers import QwenImageTransformer2DModel

        from unirl.models.qwen_image.conditions import QwenImageConditions
        from unirl.models.qwen_image.diffusion import QwenImageDiffusionStage, QwenImageDiffusionStep

        d = os.environ.get("QWEN_IMAGE_DIR", "/apdcephfs/private_aimicahchen/models/Qwen/Qwen-Image-Edit")
        tf = QwenImageTransformer2DModel.from_pretrained(f"{d}/transformer", torch_dtype=dt).to(dev)
        blocks = ("QwenImageTransformerBlock",)
        hw, ch = 48, 16
        g = torch.Generator(device=dev).manual_seed(0)
        mask = torch.ones(B, 64, device=dev, dtype=torch.long)
        conds = QwenImageConditions(
            text=TextEmbedCondition(
                embeds=torch.randn(B, 64, 3584, device=dev, dtype=dt, generator=g), pooled=None, attn_mask=mask
            ),
            negative_text=None,
        )
        StageC, StepC = QwenImageDiffusionStage, QwenImageDiffusionStep
        npar = sum(p.numel() for p in tf.parameters())
    elif model_name == "z_image":
        from diffusers import ZImageTransformer2DModel

        from unirl.models.z_image.conditions import ZImageConditions
        from unirl.models.z_image.diffusion import ZImageDiffusionStage, ZImageDiffusionStep

        d = os.environ.get(
            "ZIMAGE_DIR", "/apdcephfs_fsgm3/share_305110755/hunyuan/public_models/Tongyi-MAI/Z-Image-Turbo"
        )
        tf = ZImageTransformer2DModel.from_pretrained(f"{d}/transformer", torch_dtype=dt).to(dev)
        blocks = ("ZImageTransformerBlock",)
        hw, ch = 64, 16
        g = torch.Generator(device=dev).manual_seed(0)
        mask = torch.ones(B, 64, device=dev, dtype=torch.long)
        conds = ZImageConditions(
            text=TextEmbedCondition(
                embeds=torch.randn(B, 64, 2560, device=dev, dtype=dt, generator=g), pooled=None, attn_mask=mask
            ),
            negative_text=None,
        )
        StageC, StepC = ZImageDiffusionStage, ZImageDiffusionStep
        npar = sum(p.numel() for p in tf.parameters())
    else:
        raise ValueError(model_name)

    _rank0(f"[fsdp-bench] {model_name}: params={npar / 1e9:.2f}B; wrapping FSDP2 (reshard={reshard}) ...")
    fsdp_wrap(tf, block_class_names=blocks, param_dtype="bf16", mixed_precision=True, reshard_after_forward=reshard)

    bundle = SimpleNamespace(transformer=tf, device=dev)
    common = dict(
        model=bundle,
        step=StepC(),
        strategy=FlowSDEStrategy(),
        autocast_precision="bf16",
        trajectory_precision="bf16",
        logprob_precision="fp32",
    )
    stage_s = StageC(**common, batch_replay_steps=False)
    stage_b = StageC(**dict(common, step=StepC()), batch_replay_steps=True)

    schedule = get_sigma_schedule(T, shift=3.0, device=dev)
    params = DiffusionSamplingParams(
        num_inference_steps=T, guidance_scale=1.0, height=hw * 8, width=hw * 8, eta=0.7, seed=0, sde_indices=sde
    )
    x0 = torch.randn(B, ch, hw, hw, device=dev, dtype=stage_s.trajectory_dtype)
    return stage_s, stage_b, conds, params, x0, schedule, dict(S=len(sde), B=B, params_b=npar / 1e9)


def _timed(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
    return (time.perf_counter() - t0) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["sd3", "qwen_image", "z_image"])
    args = ap.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dev = torch.device(f"cuda:{local_rank}")
    dist.init_process_group("nccl")
    torch.manual_seed(0)

    stage_s, stage_b, conds, params, x0, schedule, info = _build(args.model, dev)

    iters = int(os.environ.get("ITERS", "10"))
    warmup = int(os.environ.get("WARMUP", "3"))
    # Pause the occupier around ALL collective work (diffuse + replays + timing):
    # NCCL all-gather kernels can't co-schedule while the busy-loop occupier hogs
    # the SMs, so the first FSDP collective would hang. The (collective-free)
    # model load above runs with the occupier live, keeping this window short
    # (under the external watchdog's restart threshold).
    pause = os.environ.get("PAUSE_OCCUPIER", "0") == "1"
    if pause:
        dist.barrier()
        _occupier("stop")
        dist.barrier()
    try:
        with torch.no_grad():
            seg = stage_s.diffuse(conds, schedule=schedule, params=params, initial_latents=x0)
        # ratio=1: two batched replays must be bit-identical.
        with torch.no_grad():
            r1 = stage_b.replay(conds, segment=seg, params=params).log_probs.float()
            r2 = stage_b.replay(conds, segment=seg, params=params).log_probs.float()
            rs = stage_s.replay(conds, segment=seg, params=params).log_probs.float()
            rb = stage_b.replay(conds, segment=seg, params=params).log_probs.float()
        ratio_dev = (torch.exp(r1 - r2) - 1.0).abs().max().item()
        parity = ((rs - rb).abs() / (rs.abs() + 1e-6)).max().item()
        with torch.no_grad():
            ts = _timed(lambda: stage_s.replay(conds, segment=seg, params=params), iters, warmup)
            tb = _timed(lambda: stage_b.replay(conds, segment=seg, params=params), iters, warmup)
    finally:
        if pause:
            _occupier("cont")

    world = dist.get_world_size()
    _rank0(
        f"\n==== FSDP replay microbench: {args.model} ({info['params_b']:.2f}B, world={world}, "
        f"S={info['S']}, B={info['B']}) ===="
    )
    _rank0(f"  parity batched-vs-serial max rel = {parity:.3e}   ratio=1 determinism max|ratio-1| = {ratio_dev:.3e}")
    _rank0(
        f"  serial ({info['S']} fwds = {info['S']} all-gathers): {ts * 1e3:.2f} ms   "
        f"batched (1 fwd = 1 all-gather): {tb * 1e3:.2f} ms   speedup = {ts / tb:.2f}x"
    )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
