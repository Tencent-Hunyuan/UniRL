#!/usr/bin/env python
"""ReFL parity check — the shared `draft_k_sample` reproduces a family's `diffuse`.

For any diffusion family, asserts that the shared, family-agnostic
`draft_k_sample(eta=0, K=T)` produces the same clean latent as that family's own
`diffuse(eta=0, sde_indices=[])` given identical conditions + schedule + initial
noise. This is the cross-family fidelity gate behind replacing per-family
`diffuse_draft_k` with one shared loop.

Run on the pod (needs the family's checkpoint):
  python scripts/refl_parity_check.py \
    --pipeline-target unirl.models.sd3.pipeline.SD3Pipeline \
    --config-target  unirl.models.sd3.config.SD3PipelineConfig \
    --model-path /path/to/checkpoint
Exit code 0 = PASS, 1 = FAIL/error.
"""

from __future__ import annotations

import argparse
import sys
import traceback

import torch
from hydra.utils import get_class

from unirl.models.draft import draft_k_sample
from unirl.sde.kernels import FlowSDEStrategy
from unirl.sde.noise import generate_latents
from unirl.sde.runtime import get_sigma_schedule
from unirl.types.primitives import Texts
from unirl.types.sampling import DiffusionSamplingParams

PROMPTS = ["a photo of a cat", "a red bicycle on a beach"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline-target", default="unirl.models.sd3.pipeline.SD3Pipeline")
    ap.add_argument("--config-target", default="unirl.models.sd3.config.SD3PipelineConfig")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--guidance", type=float, default=1.0)
    ap.add_argument("--hw", type=int, default=512)
    ap.add_argument("--frames", type=int, default=16, help="video families only; WAN needs (frames-1)%4==0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-3, help="max rel-diff on the final latent")
    args = ap.parse_args()

    device = torch.device("cuda")
    pipeline_cls = get_class(args.pipeline_target)
    config_cls = get_class(args.config_target)
    cfg = config_cls(pretrained_model_ckpt_path=args.model_path, device=device)
    # Compare both paths in fp32 so the per-step dtype policy is identical (diffuse
    # casts latents to trajectory_dtype; draft_k_sample keeps the kernel's fp32).
    # This isolates the LOOP logic — the thing the refactor must preserve.
    if hasattr(cfg, "trajectory_precision"):
        cfg.trajectory_precision = "fp32"
    pipeline = pipeline_cls.from_config(cfg, strategy=FlowSDEStrategy())
    print(f"family={args.pipeline_target.split('.')[-1]} steps={args.steps} guidance={args.guidance}", flush=True)

    params = DiffusionSamplingParams(
        num_inference_steps=args.steps, guidance_scale=args.guidance,
        height=args.hw, width=args.hw, num_frames=args.frames, eta=0.0, sde_indices=[],
        samples_per_prompt=1, seed=args.seed, init_same_noise=False,
    )
    with torch.no_grad():
        conds = pipeline.build_conditions(Texts(texts=PROMPTS), guidance_scale=args.guidance)
        shift = float(getattr(cfg, "shift", 3.0))
        schedule = get_sigma_schedule(args.steps, shift=shift, device=device)
        shape = pipeline_cls.latent_shape(model_config=cfg, sampling_spec=params)
        init = generate_latents(
            batch_size=len(PROMPTS), latent_shape=tuple(shape), device=device,
            dtype=getattr(pipeline.diffusion, "trajectory_dtype", torch.bfloat16),
            init_same_noise=False, samples_per_prompt=1, noise_group_ids=None, base_seed=args.seed,
        )

        # Family's own diffuse (deterministic) vs the shared draft loop (K=T).
        seg_diffuse = pipeline.diffusion.diffuse(conds, schedule=schedule, params=params, initial_latents=init)
        clean_diffuse = seg_diffuse.latents[:, -1].float()
        seg_draft = draft_k_sample(
            pipeline.diffusion, conds, schedule=schedule, params=params,
            draft_num_steps=args.steps, initial_latents=init,
        )
        clean_draft = seg_draft.latents[:, -1].float()

    max_abs = (clean_draft - clean_diffuse).abs().max().item()
    denom = clean_diffuse.abs().max().item() or 1.0
    rel = max_abs / denom
    print(f"final-latent max_abs_diff={max_abs:.3e}  max_rel_diff={rel:.3e}  (tol={args.tol})", flush=True)
    ok = rel < args.tol
    print(f"PARITY {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        print("ERROR: parity check raised:\n" + traceback.format_exc(), flush=True)
        sys.exit(1)
