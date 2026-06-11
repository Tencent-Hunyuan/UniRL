#!/usr/bin/env python
"""Dump one GRPO group (1 prompt x 16 samples) from a vllm_omni engine.

Two variants of the same group, mirroring the e2e request shape:
  A  per-sample initial noise  (init_same_noise=false — what the recipes run)
  B  group-shared initial noise (init_same_noise=true — flow-GRPO canonical)

For each: 16 PNGs + a 4x4 grid annotated with PickScore, plus score stats.
This makes the within-group learning signal visible: are the 16 samples
meaningfully ranked (learnable) or interchangeable (noise-dominated)?

Usage (pod, venv active, GPUs free):
  DIFFRL_OMNI_BOOT_SERIALIZE=0 \\
  PRETRAINED_MODEL=/root/diffusionrl/models/local/Qwen-Image \\
  DUMP_DIR=/mnt/bj/dump/lin382-qwen-group \\
  python examples/vllm_omni_group_dump.py
"""

from __future__ import annotations

import os
import statistics as st
import time
from types import SimpleNamespace

import torch

T0 = time.time()
G = 16  # group size (samples_per_prompt in the recipes)


def log(msg: str) -> None:
    print(f"[group +{time.time() - T0:7.1f}s] {msg}", flush=True)


def build_req(prompt: str, *, shared_noise: bool, steps: int, hw: int, seed: int):
    from unirl.types.primitives import Texts
    from unirl.types.rollout_req import RolloutReq
    from unirl.types.sampling import DiffusionSamplingParams

    gids = ["g0"] * G if shared_noise else [f"s{i}" for i in range(G)]
    return RolloutReq(
        sample_ids=[f"p0/r{i}" for i in range(G)],
        group_ids=["g0"] * G,
        primitives={"text": Texts(texts=[prompt] * G)},
        sampling_params=DiffusionSamplingParams(
            num_inference_steps=steps,
            height=hw,
            width=hw,
            guidance_scale=1.0,
            eta=0.7,
            seed=seed,
            sde_indices=[1, 3, 5],  # 3 SDE steps in the recipes' [0, 0.5] window
        ),
        init_noise_group_ids=gids,
        init_noise_latent_shape=[16, hw // 8, hw // 8],
    )


def score_images(prompt: str, decoded) -> list:
    from unirl.reward.local.pickscore import PickScoreRewardScorer, PickScoreSpec
    from unirl.types.primitives import Texts
    from unirl.types.reward import RewardRequest

    scorer = PickScoreRewardScorer(config=PickScoreSpec(batch_size=8, device="auto"), base_device="cuda")
    request = RewardRequest(primitives={"text": Texts(texts=[prompt] * G)}, generated={"image": decoded})
    return list(scorer._compute_model_rewards(request))


def save_group(tag: str, dump_dir: str, pixels: torch.Tensor, scores: list) -> None:
    from PIL import Image as PILImage
    from PIL import ImageDraw

    d = os.path.join(dump_dir, tag)
    os.makedirs(d, exist_ok=True)
    tiles = []
    order = sorted(range(G), key=lambda i: -scores[i])
    for rank, i in enumerate(order):
        arr = (pixels[i].clamp(0, 1) * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        img = PILImage.fromarray(arr)
        img.save(os.path.join(d, f"{i:02d}_ps{scores[i]:.4f}.png"))
        tiles.append((img, scores[i]))
    # 4x4 grid, score-sorted (best first), annotated.
    w, h = tiles[0][0].size
    grid = PILImage.new("RGB", (4 * w, 4 * h), (0, 0, 0))
    for k, (img, s) in enumerate(tiles):
        img = img.copy()
        ImageDraw.Draw(img).text((6, 6), f"{s:.4f}", fill=(255, 60, 60))
        grid.paste(img, ((k % 4) * w, (k // 4) * h))
    grid.save(os.path.join(dump_dir, f"grid_{tag}.png"))
    mean, sd = st.mean(scores), st.stdev(scores)
    log(f"[{tag}] mean={mean:.4f} std={sd:.4f} min={min(scores):.4f} max={max(scores):.4f}")


def main() -> int:
    model_path = os.environ.get("PRETRAINED_MODEL", "/root/diffusionrl/models/local/Qwen-Image")
    dump_dir = os.environ.get("DUMP_DIR", "/mnt/bj/dump/lin382-qwen-group")
    prompt = os.environ.get("PROMPT", "a jung male cyborg with white hair sitting down on a throne in a dystopian city")
    steps = int(os.environ.get("STEPS", "12"))
    hw = int(os.environ.get("HW", "384"))
    seed = int(os.environ.get("SEED", "42"))
    os.makedirs(dump_dir, exist_ok=True)

    from unirl.models.qwen_image.config import _qwen_image_dynamic_overrides
    from unirl.rollout.engine.vllm_omni.config import VLLMOmniEngineConfig, VLLMOmniPorts

    cfg = VLLMOmniEngineConfig(model_path=model_path, modality="qwen_image_t2i", enable_sleep_mode=False)
    model_config = SimpleNamespace(
        shift=3.0,
        use_lora=False,
        use_dynamic_shifting=True,
        dynamic_shift_overrides=_qwen_image_dynamic_overrides(),
        max_sequence_length=512,
    )
    log("booting engine ...")
    engine = cfg.make_engine(model_config=model_config, ports=VLLMOmniPorts.reserve())
    log(f"boot OK; prompt: {prompt!r}")

    try:
        for tag, shared in (("A_per_sample_noise", False), ("B_shared_noise", True)):
            req = build_req(prompt, shared_noise=shared, steps=steps, hw=hw, seed=seed)
            t = time.time()
            resp = engine.generate(req)
            log(f"[{tag}] generated {G} in {time.time() - t:.1f}s")
            pixels = resp.tracks["image"].decoded.pixels
            scores = score_images(prompt, resp.tracks["image"].decoded)
            save_group(tag, dump_dir, pixels, scores)
        log(f"GROUP DUMP COMPLETE -> {dump_dir}")
    finally:
        engine.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
