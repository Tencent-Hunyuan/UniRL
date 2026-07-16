#!/usr/bin/env python3
"""Rollout-only e2e smoke for the sglang_diffusion engine (SD3, LIN-454).

The OTHER diffusion engine (sglang's DiffGenerator, distinct from vllm_omni):
boots the REAL ``SGLangDiffusionRolloutEngine`` (model_family ``sd3``,
SD3.5-medium), builds a request ``Sample`` by hand, runs ``generate``, asserts
a filled ``LatentSegment`` (trajectory) + decoded ``Images`` + replay
conditions, σ-verified. Forward-process (ODE) variant — no SDE strategy needed.

    PRETRAINED_MODEL=/root/unirl/models/local/stable-diffusion-3.5-medium \
    CUDA_VISIBLE_DEVICES=0 .venv-sglang/bin/python scripts/rollout_sd3_sglang_smoke.py
"""

from __future__ import annotations

import os
import sys
import traceback

import torch

from unirl.models.sd3.config import SD3PipelineConfig
from unirl.rollout.engine.sglang_diffusion.config import SGLangDiffusionEngineConfig
from unirl.rollout.engine.sglang_diffusion.engine import SGLangDiffusionRolloutEngine
from unirl.types.primitives import Images, Texts
from unirl.types.sample import Part, Sample
from unirl.types.sampling import DiffusionSamplingParams
from unirl.types.segments.latent import LatentSegment


def _log(msg: str) -> None:
    print(f"[sd3-sglang-smoke] {msg}", flush=True)


def build_request_sample() -> Sample:
    prompts = ["a photo of a red apple on a wooden table", "an astronaut riding a horse on the moon"]
    input_part = Part.input([f"p{i}" for i in range(len(prompts))], primitives={"text": Texts(texts=prompts)})
    diff_params = DiffusionSamplingParams(
        num_inference_steps=4,
        guidance_scale=1.0,  # CFG off → no negative prompt / neg-embed plumbing
        height=512,
        width=512,
        eta=0.7,
        samples_per_prompt=1,
        seed=42,
        init_same_noise=False,
        sde_indices=None,  # forward process (ODE): clean-latent trajectory, no SDE label needed
    )
    return Sample.request(input_part).fork(1, sampling_params=diff_params)


def main() -> int:
    model_path = os.environ.get("PRETRAINED_MODEL")
    if not model_path:
        _log("ERROR: set PRETRAINED_MODEL to a local SD3.5-medium dir")
        return 2
    _log(f"torch {torch.__version__} cuda={torch.version.cuda}; model_path={model_path}")

    model_config = SD3PipelineConfig(pretrained_model_ckpt_path=model_path, model_precision="bf16", shift=3.0)
    config = SGLangDiffusionEngineConfig(model_family="sd3", local_mode=True, populate_conditions=True)

    engine = None
    try:
        _log("constructing SGLangDiffusionRolloutEngine (boots sglang DiffGenerator + loads SD3.5) ...")
        engine = SGLangDiffusionRolloutEngine(config, device=torch.device("cuda:0"), rank=0, model_config=model_config)
        sample = build_request_sample()
        gen_in = sample.parts[-1]
        _log(f"request: {len(sample.parts)} parts; gen ids={list(gen_in.sample_ids)}")

        _log("calling engine.generate(sample) ...")
        out = engine.generate(sample)
        _log("generate returned; checking the filled Sample ...")

        assert len(out.parts) == 2, f"expected [input, gen]; got {len(out.parts)} parts"
        gen = out.parts[-1]
        assert list(gen.sample_ids) == list(gen_in.sample_ids), "gen ids changed"
        assert isinstance(gen.segment, LatentSegment), f"segment must be LatentSegment; got {type(gen.segment)}"
        assert gen.segment.latents is not None, "LatentSegment.latents is None"
        assert isinstance(gen.primitives.get("image"), Images), (
            f"decoded image primitive must be Images; got {type(gen.primitives.get('image'))}"
        )
        assert len(gen.primitives["image"]) == 2, f"expected 2 images; got {len(gen.primitives['image'])}"
        assert gen.conditions, "replay conditions empty"

        lat = gen.segment.latents
        _log(
            f"PASS: latents shape={tuple(lat.shape)} dtype={lat.dtype}; images={len(gen.primitives['image'])} "
            f"conditions={sorted(gen.conditions.keys())}"
        )
        _log("SD3-SGLANG ROLLOUT SMOKE PASSED ✅  (σ-verified; Sample filled correctly)")
        return 0
    except Exception:
        _log("SD3-SGLANG ROLLOUT SMOKE FAILED ❌")
        traceback.print_exc()
        return 1
    finally:
        if engine is not None:
            try:
                engine.shutdown()
                _log("engine shut down")
            except Exception:
                _log("engine.shutdown() raised (ignored)")


if __name__ == "__main__":
    sys.exit(main())
