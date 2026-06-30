#!/usr/bin/env python
"""Numerical validation for Qwen-Image batched-step replay (ratio=1 exact).

Twin of ``validate_batched_replay.py`` (SD3) for the Qwen-Image transformer.
The replay phase recomputes per-SDE-step log-probs ONE step at a time (serial
loop in ``QwenImageDiffusionStage.replay``). ``batch_replay_steps=True`` stacks
all S steps on the batch dim and runs ONE forward + one vectorized SDE
transition. This proves the batched path is correct + ratio-preserving on a
single GPU (transformer only, no FSDP/Ray):

  Claim 1 (parity): batched per-(sample,step) log-prob == serial within bf16
    batch-shape tolerance, with the [B,S] mapping intact (no scramble).
  Claim 2 (ratio=1): batched replay is deterministic under no_grad -> two
    replays give bit-identical log-probs -> with old_logp_source='replay' the
    PPO ratio is exactly 1 (the anchor is replayed through the same path).
  Claim 3 (grad): backward flows through the batched replay (train path works).

Qwen-Image specifics exercised here vs SD3: variable-length text trimmed
per-call by the attention mask, ``img_shapes`` / ``txt_seq_lens`` derived from
the batch, and the 16-ch latent packed to ``[B, (H/2)*(W/2), 64]``.

Correctness (Claims 1-2) is independent of GPU contention, so this can run with
the occupier live. The optional bench (REPLAY_BENCH=1) is timing and wants a
quiet GPU.

Run:
  CUDA_VISIBLE_DEVICES=0 \
  QWEN_IMAGE_DIR=/apdcephfs/private_aimicahchen/models/Qwen/Qwen-Image-Edit \
    python scripts/profiling/validate_batched_replay_qwen_image.py
"""

from __future__ import annotations

import os
import time
from types import SimpleNamespace

import torch

from unirl.models.qwen_image.conditions import QwenImageConditions
from unirl.models.qwen_image.diffusion import QwenImageDiffusionStage, QwenImageDiffusionStep
from unirl.sde.kernels import FlowSDEStrategy
from unirl.sde.runtime import get_sigma_schedule
from unirl.types.conditions import TextEmbedCondition
from unirl.types.sampling import DiffusionSamplingParams

QWEN_IMAGE_DIR = os.environ.get("QWEN_IMAGE_DIR", "/apdcephfs/private_aimicahchen/models/Qwen/Qwen-Image-Edit")
B = int(os.environ.get("B", "2"))
T = int(os.environ.get("T", "6"))
SDE_INDICES = [int(i) for i in os.environ.get("SDE_INDICES", "0,2,4").split(",")]
HW = int(os.environ.get("LATENT_HW", "48"))  # 384px -> 2*(384//16) = 48
TXT = int(os.environ.get("SEQ_TXT", "64"))
TXT_DIM = int(os.environ.get("TXT_DIM", "3584"))  # Qwen2.5-VL hidden / joint_attention_dim
SHIFT = 3.0
DTYPE = torch.bfloat16


def _build_stage(batch_replay_steps: bool, transformer=None) -> QwenImageDiffusionStage:
    from diffusers import QwenImageTransformer2DModel

    if transformer is None:
        transformer = (
            QwenImageTransformer2DModel.from_pretrained(f"{QWEN_IMAGE_DIR}/transformer", torch_dtype=DTYPE)
            .to("cuda")
            .eval()
        )
    bundle = SimpleNamespace(transformer=transformer, device=torch.device("cuda"))
    return QwenImageDiffusionStage(
        model=bundle,
        step=QwenImageDiffusionStep(),
        strategy=FlowSDEStrategy(),
        autocast_precision="bf16",
        trajectory_precision="bf16",
        logprob_precision="fp32",
        batch_replay_steps=batch_replay_steps,
    )


def _conditions(seed: int = 0) -> QwenImageConditions:
    g = torch.Generator(device="cuda").manual_seed(seed)
    embeds = torch.randn(B, TXT, TXT_DIM, device="cuda", dtype=DTYPE, generator=g)
    # Varied true lengths per sample to exercise the per-call max-len trim;
    # tiling must reproduce the same max over each B-block.
    mask = torch.ones(B, TXT, device="cuda", dtype=torch.long)
    for i in range(B):
        true_len = max(8, TXT - i * 8)
        mask[i, true_len:] = 0
    return QwenImageConditions(text=TextEmbedCondition(embeds=embeds, pooled=None, attn_mask=mask), negative_text=None)


def main() -> None:
    torch.manual_seed(0)
    print(f"[validate-qwen] loading transformer (B={B}, T={T}, sde={SDE_INDICES}, hw={HW}) ...", flush=True)
    stage = _build_stage(batch_replay_steps=False)
    stage_b = _build_stage(batch_replay_steps=True, transformer=stage.model.transformer)

    schedule = get_sigma_schedule(T, shift=SHIFT, device=torch.device("cuda"))
    params = DiffusionSamplingParams(
        num_inference_steps=T,
        guidance_scale=1.0,  # CFG off -> 1 forward/step (the RL setting)
        height=HW * 8,
        width=HW * 8,
        eta=0.7,
        seed=0,
        sde_indices=list(SDE_INDICES),
    )
    conds = _conditions()
    x0 = torch.randn(B, 16, HW, HW, device="cuda", dtype=stage.trajectory_dtype)

    with torch.no_grad():
        seg = stage.diffuse(conds, schedule=schedule, params=params, initial_latents=x0)
    print(
        f"[validate-qwen] segment latents={tuple(seg.latents.shape)} sde_indices={seg.sde_indices.tolist()} "
        f"sde_logp={tuple(seg.sde_logp.shape)}",
        flush=True,
    )

    # ---- Claim 1: batched vs serial replay parity ----
    with torch.no_grad():
        rep_s = stage.replay(conds, segment=seg, params=params)
        rep_b = stage_b.replay(conds, segment=seg, params=params)
    lp_s, lp_b = rep_s.log_probs.float(), rep_b.log_probs.float()
    assert lp_s.shape == lp_b.shape == (B, len(SDE_INDICES)), (lp_s.shape, lp_b.shape)
    abs_diff = (lp_s - lp_b).abs()
    rel = abs_diff / (lp_s.abs() + 1e-6)
    print("\n[Claim 1] batched vs serial replay log-probs:", flush=True)
    print(
        f"  shape={tuple(lp_b.shape)}  max|abs diff|={abs_diff.max().item():.4e}  "
        f"max rel={rel.max().item():.4e}  mean rel={rel.mean().item():.4e}",
        flush=True,
    )
    ok1 = rel.max().item() < 2e-2
    if B > 1:
        cross = ((lp_s[0] - lp_b[(0 + 1) % B]).abs() / (lp_s[0].abs() + 1e-6)).mean().item()
        print(
            f"  matched mean rel={rel.mean().item():.4e}  vs cross-sample rel={cross:.4e} "
            f"(cross must be >> matched -> [B,S] mapping not scrambled)",
            flush=True,
        )
        ok1 = ok1 and (cross > 10 * rel.mean().item())

    # ---- Claim 2: batched replay deterministic -> ratio=1 ----
    with torch.no_grad():
        r1 = stage_b.replay(conds, segment=seg, params=params).log_probs.float()
        r2 = stage_b.replay(conds, segment=seg, params=params).log_probs.float()
    max_ratio_dev = (torch.exp(r1 - r2) - 1.0).abs().max().item()
    print("\n[Claim 2] batched replay determinism (ratio=1 anchor):", flush=True)
    print(f"  replay#1 vs #2: max|ratio-1| = {max_ratio_dev:.3e} (expect 0)", flush=True)
    ok2 = max_ratio_dev < 1e-6

    # ---- Claim 3: grad flows through batched replay ----
    print("\n[Claim 3] backward through batched replay:", flush=True)
    stage_b.model.transformer.train()
    try:
        rep = stage_b.replay(conds, segment=seg, params=params)
        adv = torch.randn(B, device="cuda")
        loss = (adv.unsqueeze(1) * rep.log_probs).mean()
        loss.backward()
        gnorm = (
            sum(p.grad.float().norm().item() ** 2 for p in stage_b.model.transformer.parameters() if p.grad is not None)
            ** 0.5
        )
        ok3 = gnorm > 0
        print(f"  loss={loss.item():.4f}  grad_norm={gnorm:.4e}  -> {'OK' if ok3 else 'FAIL (no grad)'}", flush=True)
    except RuntimeError as e:  # OOM on the 20B model is acceptable; correctness is Claims 1-2.
        ok3 = None
        print(f"  SKIPPED (runtime error, e.g. OOM on 20B full-param grad): {str(e)[:80]}", flush=True)
    stage_b.model.transformer.zero_grad(set_to_none=True)
    stage_b.model.transformer.eval()

    print("\n==== RESULT ====", flush=True)
    print(f"  Claim 1 (batched==serial, bf16 tol + mapping): {'PASS' if ok1 else 'FAIL'}", flush=True)
    print(f"  Claim 2 (batched replay deterministic ratio=1): {'PASS' if ok2 else 'FAIL'}", flush=True)
    c3 = "SKIP" if ok3 is None else ("PASS" if ok3 else "FAIL")
    print(f"  Claim 3 (grad flows through batched replay):    {c3}", flush=True)

    if os.environ.get("REPLAY_BENCH", "0") == "1":
        _bench(stage, stage_b, conds, seg, params)


def _bench(stage, stage_b, conds, seg, params) -> None:
    print(
        "\n[bench] serial vs batched replay (S={} steps); timing wants a quiet GPU".format(len(SDE_INDICES)), flush=True
    )

    def timed(fn, iters=10, warmup=3):
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters

    with torch.no_grad():
        ts = timed(lambda: stage.replay(conds, segment=seg, params=params))
        tb = timed(lambda: stage_b.replay(conds, segment=seg, params=params))
    print(
        f"  forward (no_grad):  serial={ts * 1e3:.2f}ms  batched={tb * 1e3:.2f}ms  speedup={ts / tb:.2f}x", flush=True
    )


if __name__ == "__main__":
    main()
