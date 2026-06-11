#!/usr/bin/env python
"""Dump sample images from a vllm_omni engine (LIN-382 visual check).

Boots ONE engine for the given modality, generates one image per prompt at
recipe-faithful settings, saves PNGs (PickScore annotated when the scorer
loads), and writes a manifest.

Usage (inside the pod, venv active, GPUs free):
  DIFFRL_OMNI_BOOT_SERIALIZE=0 \\
  PRETRAINED_MODEL=/root/diffusionrl/models/local/Qwen-Image \\
  DUMP_DIR=/mnt/bj/dump/lin382-qwen-samples \\
  python examples/vllm_omni_dump_images.py
"""

from __future__ import annotations

import os
import time
from types import SimpleNamespace

import torch

T0 = time.time()


def log(msg: str) -> None:
    print(f"[dump +{time.time() - T0:7.1f}s] {msg}", flush=True)


def main() -> int:
    model_path = os.environ.get("PRETRAINED_MODEL", "/root/diffusionrl/models/local/Qwen-Image")
    modality = os.environ.get("DUMP_MODALITY", "qwen_image_t2i")
    dump_dir = os.environ.get("DUMP_DIR", "/mnt/bj/dump/lin382-qwen-samples")
    prompts_file = os.environ.get("PROMPTS_FILE", "datasets/pickscore/test.txt")
    n = int(os.environ.get("N_PROMPTS", "8"))
    steps = int(os.environ.get("STEPS", "12"))
    hw = int(os.environ.get("HW", "384"))
    seed = int(os.environ.get("SEED", "42"))

    os.makedirs(dump_dir, exist_ok=True)

    with open(prompts_file) as f:
        prompts = [line.strip() for line in f if line.strip()][:n]
    log(f"{len(prompts)} prompts from {prompts_file}")

    from unirl.models.qwen_image.config import _qwen_image_dynamic_overrides
    from unirl.rollout.engine.vllm_omni.config import VLLMOmniEngineConfig, VLLMOmniPorts
    from unirl.types.primitives import Texts
    from unirl.types.rollout_req import RolloutReq
    from unirl.types.sampling import DiffusionSamplingParams

    cfg = VLLMOmniEngineConfig(model_path=model_path, modality=modality, enable_sleep_mode=False)
    model_config = SimpleNamespace(
        shift=3.0,
        use_lora=False,
        use_dynamic_shifting=True,
        dynamic_shift_overrides=_qwen_image_dynamic_overrides(),
        max_sequence_length=512,
    )
    log("booting engine ...")
    engine = cfg.make_engine(model_config=model_config, ports=VLLMOmniPorts.reserve())
    log("boot OK")

    try:
        req = RolloutReq(
            sample_ids=[f"p{i}/r0" for i in range(len(prompts))],
            group_ids=[f"g{i}" for i in range(len(prompts))],
            primitives={"text": Texts(texts=prompts)},
            sampling_params=DiffusionSamplingParams(
                num_inference_steps=steps,
                height=hw,
                width=hw,
                guidance_scale=1.0,
                eta=0.0,
                seed=seed,
            ),
        )
        t = time.time()
        resp = engine.generate(req)
        log(f"generated {len(prompts)} images in {time.time() - t:.1f}s")

        pixels = resp.tracks["image"].decoded.pixels  # [B, 3, H, W] in [0, 1]

        # Best-effort PickScore annotation (same scorer as the e2e reward).
        scores = [None] * len(prompts)
        try:
            from unirl.reward.local.pickscore import PickScoreRewardScorer, PickScoreSpec
            from unirl.types.reward import RewardRequest

            scorer = PickScoreRewardScorer(config=PickScoreSpec(batch_size=8, device="auto"), base_device="cuda")
            request = RewardRequest(
                primitives={"text": Texts(texts=list(prompts))},
                generated={"image": resp.tracks["image"].decoded},
            )
            scores = list(scorer._compute_model_rewards(request))
            log(f"pickscores: {[round(float(s), 4) for s in scores]}")
        except Exception as e:  # noqa: BLE001
            log(f"pickscore annotation skipped: {e!r}")

        from PIL import Image as PILImage

        manifest = []
        for i, prompt in enumerate(prompts):
            arr = (pixels[i].clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            tag = f"{float(scores[i]):.4f}" if scores[i] is not None else "na"
            name = f"{i:02d}_ps{tag}.png"
            PILImage.fromarray(arr).save(os.path.join(dump_dir, name))
            manifest.append(f"{name}\t{prompt}")
            log(f"saved {name}  | {prompt[:70]}")
        with open(os.path.join(dump_dir, "prompts.txt"), "w") as f:
            f.write("\n".join(manifest) + "\n")
        log(f"DUMP COMPLETE -> {dump_dir}")
    finally:
        engine.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
