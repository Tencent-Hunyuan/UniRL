#!/usr/bin/env python
"""Phase 1 ReFL oracle — single-process gate for the shared grad sampler +
differentiable reward (Protocol form).

Runs the REAL components in one process with plain ``torch.autograd``:
  SD3 pipeline → inject LoRA → ``draft_generate`` (shared `draft_k_sample` +
  grad `decode`) → PickScore ``compute_rewards_differentiable`` →
  ``-reward.mean()`` → ``backward()``.

Gate checks (all must pass):
  0. protocol — the reward satisfies the `DifferentiableReward` Protocol.
  1. connectivity — every LoRA param gets a finite, non-None, non-zero grad.
  2. DRaFT-K isolation — grad norm differs between K=1 and K=T (the mask works).
  3. preprocessing fidelity — differentiable PickScore ≈ the PIL processor path.

Run on the pod (needs GPU + SD3.5 + PickScore):
  PRETRAINED_MODEL=/path/to/sd3.5-medium python scripts/phase1_refl_oracle.py
Exit code 0 = PASS, 1 = FAIL/error.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

import torch

from unirl.models.draft import draft_generate
from unirl.models.sd3.config import SD3PipelineConfig
from unirl.models.sd3.pipeline import SD3Pipeline
from unirl.reward.base import DifferentiableReward
from unirl.reward.local.pickscore import PickScoreRewardScorer, PickScoreSpec
from unirl.sde.kernels import FlowSDEStrategy
from unirl.train.inject import inject_lora
from unirl.types.primitives import Texts
from unirl.types.reward import RewardRequest
from unirl.types.sampling import DiffusionSamplingParams

LORA_TARGETS = [
    "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
    "attn.to_k", "attn.to_out.0", "attn.to_q", "attn.to_v",
]
PROMPTS = [
    "a photo of a cat sitting on a windowsill",
    "a red bicycle leaning on a brick wall",
    "an astronaut riding a horse on the moon",
    "a steaming bowl of ramen with an egg",
]


def build(model_path: str, device: torch.device):
    cfg = SD3PipelineConfig(
        pretrained_model_ckpt_path=model_path,
        model_precision="bf16", autocast_precision="bf16",
        trajectory_precision="bf16", logprob_precision="fp32",
        shift=3.0, device=device,
    )
    pipeline = SD3Pipeline.from_config(cfg, strategy=FlowSDEStrategy())
    inject_lora(
        pipeline.bundle.transformer,
        rank=32, alpha=64, target_modules=LORA_TARGETS,
        dropout=0.0, bias="none", task_type="FEATURE_EXTRACTION",
    )
    for _, p in pipeline.bundle.transformer.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()  # fp32 LoRA master
    pipeline.bundle.transformer.train()
    return pipeline, cfg


def sample_reward(pipeline, cfg, reward, prompts, *, steps, guidance, hw, draft_k, seed):
    params = DiffusionSamplingParams(
        num_inference_steps=steps, guidance_scale=guidance, height=hw, width=hw,
        eta=0.0, samples_per_prompt=1, seed=seed, init_same_noise=False,
    )
    images = draft_generate(
        pipeline, model_config=cfg, texts=Texts(texts=list(prompts)),
        params=params, draft_num_steps=draft_k, activation_checkpoint=False,
    )
    rewards = reward.compute_rewards_differentiable(images.pixels, list(prompts))
    return images, rewards


def _grad_norm(named):
    grads = [p.grad.norm() for _, p in named if p.grad is not None]
    return torch.norm(torch.stack(grads)).item() if grads else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=os.environ.get("PRETRAINED_MODEL", "stabilityai/stable-diffusion-3.5-medium"))
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--hw", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tol", type=float, default=2e-2)
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    print(f"model={args.model_path} steps={args.steps} guidance={args.guidance} hw={args.hw}", flush=True)

    pipeline, cfg = build(args.model_path, device)
    reward = PickScoreRewardScorer(config=PickScoreSpec(batch_size=8, device="auto"), base_device="cuda")
    model = pipeline.bundle.transformer
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    print(f"trainable LoRA params: {len(named)}", flush=True)

    ok = True

    # ---- Check 0: protocol ----
    c0 = isinstance(reward, DifferentiableReward)
    print(f"[c0] isinstance(PickScore, DifferentiableReward)={c0}", flush=True)
    print(f"[c0] {'PASS' if c0 else 'FAIL'} protocol", flush=True)
    ok = ok and c0

    # ---- Check 1: connectivity (grad through ALL steps, K=T) ----
    model.zero_grad(set_to_none=True)
    images, rewards = sample_reward(
        pipeline, cfg, reward, PROMPTS, steps=args.steps, guidance=args.guidance,
        hw=args.hw, draft_k=args.steps, seed=args.seed,
    )
    print(
        f"[c1] images {tuple(images.pixels.shape)} finite={bool(torch.isfinite(images.pixels).all())} "
        f"reward_mean={rewards.mean().item():.4f} reward.grad_fn={rewards.grad_fn is not None}",
        flush=True,
    )
    (-rewards.mean()).backward()
    n_none = sum(1 for _, p in named if p.grad is None)
    n_nonfinite = sum(1 for _, p in named if p.grad is not None and not torch.isfinite(p.grad).all())
    norm_kt = _grad_norm(named)
    print(f"[c1] grad_none={n_none} grad_nonfinite={n_nonfinite} total_grad_norm={norm_kt:.4e}", flush=True)
    c1 = len(named) > 0 and n_none == 0 and n_nonfinite == 0 and norm_kt > 0
    print(f"[c1] {'PASS' if c1 else 'FAIL'} connectivity", flush=True)
    ok = ok and c1

    # ---- Check 2: DRaFT-K isolation (K=1 vs K=T) ----
    model.zero_grad(set_to_none=True)
    _, rewards1 = sample_reward(
        pipeline, cfg, reward, PROMPTS, steps=args.steps, guidance=args.guidance,
        hw=args.hw, draft_k=1, seed=args.seed,
    )
    (-rewards1.mean()).backward()
    norm_k1 = _grad_norm(named)
    rel = abs(norm_k1 - norm_kt) / max(norm_kt, 1e-12)
    print(f"[c2] grad_norm K=1={norm_k1:.4e}  K=T={norm_kt:.4e}  rel_diff={rel:.3f}", flush=True)
    c2 = norm_k1 > 0 and rel > 0.01
    print(f"[c2] {'PASS' if c2 else 'FAIL'} DRaFT-K isolation", flush=True)
    ok = ok and c2

    # ---- Check 3: preprocessing fidelity (differentiable vs PIL path) ----
    with torch.no_grad():
        images_d, rewards_diff = sample_reward(
            pipeline, cfg, reward, PROMPTS, steps=args.steps, guidance=args.guidance,
            hw=args.hw, draft_k=0, seed=args.seed,
        )
        req = RewardRequest(primitives={"text": Texts(texts=list(PROMPTS))}, generated={"image": images_d})
        rewards_pil = torch.tensor(reward._compute_model_rewards(req), device=device, dtype=torch.float32)
    max_abs = (rewards_diff.detach().float() - rewards_pil).abs().max().item()
    print(
        f"[c3] diff_mean={rewards_diff.mean().item():.4f} pil_mean={rewards_pil.mean().item():.4f} "
        f"max_abs_diff={max_abs:.4e}",
        flush=True,
    )
    c3 = max_abs < args.tol
    print(f"[c3] {'PASS' if c3 else 'FAIL'} preprocessing fidelity (tol={args.tol})", flush=True)
    ok = ok and c3

    print(f"\nORACLE {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        print("ERROR: oracle raised:\n" + traceback.format_exc(), flush=True)
        sys.exit(1)
