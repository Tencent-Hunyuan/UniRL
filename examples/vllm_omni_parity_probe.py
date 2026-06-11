#!/usr/bin/env python
"""ODE parity probe: engine vs trainside, same prompt, same x_T, same sigma.

eta=0 makes both paths fully deterministic, so per-step latent deltas are
pure implementation divergence. Sequence:

  1. Boot the vllm_omni engine (qwen_image_t2i); generate ONE sample with
     a NoiseRecipe-authored x_T (byte-identical regenerable), pinned sigma,
     eta=0. Harvest: dense trajectory latents + the engine's captured text
     conditioning.
  2. Shut the engine down; load the TRAINSIDE diffusers transformer alone.
  3. Re-run the SAME denoise loop trainside via QwenImageDiffusionStep
     .predict_noise (the exact replay code path), from the SAME x_T, with
     the ENGINE-CAPTURED conditioning (isolates DiT-impl divergence from
     conditioning divergence), in two integration variants:
       mirror     x carried in bf16 between steps (mirrors the engine loop)
       faithful   x carried in fp32 (trainside replay convention)
     plus a timestep-rounding variant of step-0 mu:
       tsround    timestep = bf16(sigma*1000)/1000 (the engine's path)
                  vs bf16(sigma) (trainside's path)
  4. Print per-step max/mean abs deltas engine-vs-trainside + step-0 mu
     deltas, so the divergence is localized and quantified.

Usage (pod, venv active, ONE free GPU):
  DIFFRL_OMNI_BOOT_SERIALIZE=0 \\
  PRETRAINED_MODEL=/root/diffusionrl/models/local/Qwen-Image \\
  python examples/vllm_omni_parity_probe.py
"""

from __future__ import annotations

import gc
import os
import time
from types import SimpleNamespace

import torch

T0 = time.time()


def log(msg: str) -> None:
    print(f"[parity +{time.time() - T0:7.1f}s] {msg}", flush=True)


def stats(name: str, a: torch.Tensor, b: torch.Tensor) -> str:
    d = (a.float() - b.float()).abs()
    denom = b.float().abs().mean().clamp_min(1e-12)
    return f"{name}: max={d.max().item():.3e} mean={d.mean().item():.3e} rel={d.mean().item() / denom.item():.3e}"


def main() -> int:
    model_path = os.environ.get("PRETRAINED_MODEL", "/root/diffusionrl/models/local/Qwen-Image")
    prompt = os.environ.get("PROMPT", "a jung male cyborg with white hair sitting down on a throne in a dystopian city")
    steps = int(os.environ.get("STEPS", "12"))
    hw = int(os.environ.get("HW", "384"))
    seed = int(os.environ.get("SEED", "42"))
    lat_h = lat_w = hw // 8

    from unirl.models.qwen_image.config import _qwen_image_dynamic_overrides
    from unirl.rollout.engine.vllm_omni.config import VLLMOmniPorts, VLLMOmniEngineConfig
    from unirl.sde.runtime import FlowMatchSchedulePolicy
    from unirl.types.noise_recipe import NoiseRecipe
    from unirl.types.primitives import Texts
    from unirl.types.rollout_req import RolloutReq
    from unirl.types.sampling import DiffusionSamplingParams

    overrides = _qwen_image_dynamic_overrides()
    policy = FlowMatchSchedulePolicy.from_pretrained(
        model_path, shift=3.0, require_dynamic=True, dynamic_overrides=overrides
    )
    sigmas = policy.compute_sigma(num_inference_steps=steps, height=hw, width=hw, device=torch.device("cpu"))
    log(f"pinned sigmas: {[round(float(s), 5) for s in sigmas]}")

    x_t = NoiseRecipe(noise_group_ids=["s0"], base_seed=seed, latent_shape=(16, lat_h, lat_w)).resolve()
    log(f"x_T: shape={tuple(x_t.shape)} fp={x_t.dtype} checksum={x_t.sum().item():.6f}")

    # ---------------- 1. engine rollout (ODE) ----------------------------
    cfg = VLLMOmniEngineConfig(model_path=model_path, modality="qwen_image_t2i", enable_sleep_mode=False)
    model_config = SimpleNamespace(
        shift=3.0, use_lora=False, use_dynamic_shifting=True,
        dynamic_shift_overrides=overrides, max_sequence_length=512,
    )
    log("booting engine ...")
    engine = cfg.make_engine(model_config=model_config, ports=VLLMOmniPorts.reserve())
    req = RolloutReq(
        sample_ids=["p0/r0"],
        group_ids=["g0"],
        primitives={"text": Texts(texts=[prompt])},
        sampling_params=DiffusionSamplingParams(
            num_inference_steps=steps, height=hw, width=hw,
            guidance_scale=1.0, eta=0.0, seed=seed,
        ),
        init_noise_group_ids=["s0"],
        init_noise_latent_shape=[16, lat_h, lat_w],
        sigmas=sigmas.clone(),
    )
    resp = engine.generate(req)
    track = resp.tracks["image"]
    seg = track.segment
    eng_traj = seg.latents[0].float()  # [K, C, H, W]
    eng_idx = [int(i) for i in seg.indices.tolist()]
    cond = track.conditions["text"]
    embeds = cond.embeds.clone()
    mask = cond.attn_mask.clone()
    log(f"engine traj: K={len(eng_idx)} indices={eng_idx} latent={tuple(eng_traj.shape)} embeds={tuple(embeds.shape)}")
    engine.shutdown()
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    log("engine down; loading trainside transformer ...")

    # ---------------- 2. trainside transformer ---------------------------
    from diffusers import QwenImageTransformer2DModel

    from unirl.models.qwen_image.conditions import QwenImageConditions
    from unirl.models.qwen_image.diffusion import QwenImageDiffusionStep
    from unirl.types.conditions.text import TextEmbedCondition

    transformer = QwenImageTransformer2DModel.from_pretrained(
        model_path, subfolder="transformer", torch_dtype=torch.bfloat16
    ).to("cuda").eval()

    # Optional fp32-island alignment: vllm-omni's reimplementation runs
    # time-embed / AdaLN modulation / norm_out / proj_out in FULL fp32;
    # diffusers under autocast runs the modulation Linears in bf16. Wrap the
    # matching trainside modules to compute in fp32 (autocast-exempt) to test
    # whether the island mismatch is the dominant per-forward divergence.
    if os.environ.get("PROBE_FP32_ISLANDS", "0") == "1":
        import torch.nn as nn

        targets = []
        for name, mod in transformer.named_modules():
            short = name.split(".")[-1]
            if short in ("time_text_embed", "norm_out", "proj_out", "img_mod", "txt_mod") and name.count(short) >= 1:
                targets.append((name, mod))
        log(f"fp32-island wrap: {len(targets)} modules ({sorted(set(n.split('.')[-1] for n, _ in targets))})")

        def wrap(mod: nn.Module) -> None:
            mod.float()
            orig_forward = mod.forward

            def fp32_forward(*args, _orig=orig_forward, **kw):
                with torch.autocast("cuda", enabled=False):
                    args = tuple(a.float() if torch.is_tensor(a) and a.is_floating_point() else a for a in args)
                    kw = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in kw.items()}
                    out = _orig(*args, **kw)
                if torch.is_tensor(out):
                    return out.to(torch.bfloat16)
                if isinstance(out, tuple):
                    return tuple(o.to(torch.bfloat16) if torch.is_tensor(o) and o.is_floating_point() else o for o in out)
                return out

            mod.forward = fp32_forward

        for _, mod in targets:
            wrap(mod)

    shim = SimpleNamespace(transformer=transformer)
    step_impl = QwenImageDiffusionStep()
    conds = QwenImageConditions(
        text=TextEmbedCondition(
            embeds=embeds.to("cuda", torch.bfloat16), pooled=None, attn_mask=mask.to("cuda")
        )
    )
    log("trainside transformer up")

    def mu(x: torch.Tensor, sigma_val: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            return step_impl.predict_noise(
                shim,
                x.to("cuda"),
                sigma_val.to("cuda"),
                conds,
                guidance_scale=1.0,
                latent_h=lat_h,
                latent_w=lat_w,
            ).float()

    # step-0 mu under both timestep rounding paths (D2 quantification)
    x0 = x_t.float().cuda()
    s0 = sigmas[0].float()
    v_plain = mu(x0, s0)
    s0_eng = (s0 * 1000).to(torch.bfloat16).float() / 1000.0  # engine's rounding grid
    v_tsround = mu(x0, s0_eng)
    print(flush=True)
    log("=== step-0 model-output divergence (same x_T, engine-captured conds) ===")
    log(stats("D2 timestep-rounding alone (trainside vs engine ts-grid)", v_plain, v_tsround))

    # ---------------- 3. ODE integration, two carry variants -------------
    results = {}
    for variant, carry_bf16 in (("mirror_bf16_carry", True), ("faithful_fp32_carry", False)):
        x = x_t.float().cuda()
        traj = [x.clone()]
        for i in range(steps):
            v = mu(x, sigmas[i].float())
            dt = (sigmas[i + 1] - sigmas[i]).float().cuda()
            x = x + v * dt
            if carry_bf16:
                x = x.to(torch.bfloat16).float()
            traj.append(x.clone())
        results[variant] = torch.stack(traj, dim=0)  # [T+1, C, H, W]
        log(f"{variant}: done")

    # ---------------- 4. compare vs engine trajectory --------------------
    print(flush=True)
    log("=== per-step |engine - trainside| (engine indices) ===")
    for variant, tr in results.items():
        log(f"--- {variant} ---")
        for k, idx in enumerate(eng_idx):
            if idx < tr.shape[0]:
                log("  " + stats(f"x[{idx:2d}]", eng_traj[k], tr[idx]))
    log("PARITY PROBE COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
