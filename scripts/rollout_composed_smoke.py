#!/usr/bin/env python3
"""Rollout-only e2e smoke for the composed PE engine (AR→diffusion, LIN-454).

The unified MULTI-STAGE path: an AR child (sglang, Qwen3) rewrites prompts,
a diffusion child (sglang_diffusion, SD3.5) renders them. Boots the REAL
``ComposedRolloutEngine`` with both children, builds a pre-forked 3-part
request ``Sample`` ``[input, ar_shell, diffusion_shell]`` by hand, runs
``generate``, and asserts the returned ``[input, ar, diffusion]`` chain is
filled: the ar Part with a TextSegment + decoded Texts, the diffusion Part
with a LatentSegment + decoded Images, path ids chaining p0 → p0/0 → p0/0/0.

This is the only e2e that exercises the 3-part Sample chain (the composed
re-root + map-back + assemble — the hardest conversion).

    QWEN3_PATH=/root/unirl/models/local/Qwen3-4B-Base \
    PRETRAINED_MODEL=/root/unirl/models/local/stable-diffusion-3.5-medium \
    CUDA_VISIBLE_DEVICES=0 .venv-sglang/bin/python scripts/rollout_composed_smoke.py
"""

from __future__ import annotations

import os
import sys
import traceback

import torch

from unirl.models.sd3.config import SD3PipelineConfig
from unirl.rollout.engine.composed.config import ComposedRolloutEngineConfig
from unirl.rollout.engine.composed.engine import ComposedRolloutEngine
from unirl.rollout.engine.sglang.config import SGLangEngineConfig
from unirl.rollout.engine.sglang_diffusion.config import SGLangDiffusionEngineConfig
from unirl.types.primitives import Images, Texts
from unirl.types.sample import Part, Sample
from unirl.types.sampling import ARSamplingParams, DiffusionSamplingParams
from unirl.types.segments.latent import LatentSegment
from unirl.types.segments.text import TextSegment


def _log(msg: str) -> None:
    print(f"[composed-smoke] {msg}", flush=True)


def build_request_sample() -> Sample:
    """Pre-forked ``[input, ar_shell(N=1), diffusion_shell(M=1)]`` (the caller's job)."""
    prompts = ["a fluffy cat sitting on a sofa", "a beautiful sunset over the ocean"]
    input_part = Part.input(
        [f"p{i}" for i in range(len(prompts))],
        primitives={"text": Texts(texts=prompts)},
        control={"ar": {}, "chat": {}},
    )
    ar_params = ARSamplingParams(samples_per_prompt=1, temperature=0.7, max_new_tokens=32, top_p=0.9, top_k=20)
    diff_params = DiffusionSamplingParams(
        num_inference_steps=4,
        guidance_scale=1.0,
        height=512,
        width=512,
        eta=0.7,
        samples_per_prompt=1,
        seed=42,
        init_same_noise=False,
        sde_indices=None,
    )
    return Sample.request(input_part).fork(1, sampling_params=ar_params).fork(1, sampling_params=diff_params)


def main() -> int:
    qwen = os.environ.get("QWEN3_PATH")
    sd3 = os.environ.get("PRETRAINED_MODEL")
    if not qwen or not sd3:
        _log("ERROR: set QWEN3_PATH and PRETRAINED_MODEL to local model dirs")
        return 2
    _log(f"torch {torch.__version__} cuda={torch.version.cuda}; ar={qwen} diffusion={sd3}")

    ar_config = SGLangEngineConfig(
        pretrained_model_ckpt_path=qwen,
        backend="native",
        tp_size=1,
        max_new_tokens=32,
        temperature=0.7,
        top_p=0.9,
        concurrency=8,
        # Colocate: AR (Qwen3) + diffusion (SD3.5) share ONE GPU. Cap the sglang
        # KV reservation (default 0.88 ≈ 85 GiB) so SD3.5 fits alongside — this
        # is the engine_kwargs the colocate recipes set; nothing Sample-specific.
        engine_kwargs={"mem_fraction_static": 0.35},
    )
    diff_config = SGLangDiffusionEngineConfig(model_family="sd3", local_mode=True, populate_conditions=True)
    # model_config rides to BOTH children; the AR child ignores it, the diffusion child loads SD3.5.
    model_config = SD3PipelineConfig(pretrained_model_ckpt_path=sd3, model_precision="bf16", shift=3.0)
    config = ComposedRolloutEngineConfig(
        ar=ar_config, diffusion=diff_config, sleep_diffusion_on_start=False, pe_instruction=None, pe_marker=None
    )

    engine = None
    try:
        _log("constructing ComposedRolloutEngine (boots AR sglang + diffusion sglang_diffusion) ...")
        engine = ComposedRolloutEngine(config, device=torch.device("cuda:0"), rank=0, model_config=model_config)
        sample = build_request_sample()
        _log(
            f"request: {len(sample.parts)} parts; "
            f"ar ids={list(sample.parts[1].sample_ids)} diffusion ids={list(sample.parts[2].sample_ids)}"
        )

        _log("calling engine.generate(sample) — AR rewrite → diffusion render ...")
        out = engine.generate(sample)
        _log("generate returned; checking the filled 3-part Sample ...")

        assert len(out.parts) == 3, f"expected [input, ar, diffusion]; got {len(out.parts)} parts"
        inp, ar, diff = out.parts
        # lineage: input p0 → ar p0/0 → diffusion p0/0/0
        assert list(ar.sample_ids) == ["p0/0", "p1/0"], f"ar ids: {list(ar.sample_ids)}"
        assert list(diff.sample_ids) == ["p0/0/0", "p1/0/0"], f"diffusion ids: {list(diff.sample_ids)}"
        # ar Part filled
        assert isinstance(ar.segment, TextSegment), f"ar segment must be TextSegment; got {type(ar.segment)}"
        assert isinstance(ar.primitives.get("text"), Texts) and len(ar.primitives["text"].texts) == 2, (
            "ar decoded Texts wrong"
        )
        # diffusion Part filled
        assert isinstance(diff.segment, LatentSegment), (
            f"diffusion segment must be LatentSegment; got {type(diff.segment)}"
        )
        assert diff.segment.latents is not None, "diffusion LatentSegment.latents is None"
        assert isinstance(diff.primitives.get("image"), Images) and len(diff.primitives["image"]) == 2, (
            "diffusion decoded Images wrong"
        )

        _log(f"PASS: ar texts (PE)={[t[:50] for t in ar.primitives['text'].texts]}")
        _log(
            f"PASS: diffusion latents={tuple(diff.segment.latents.shape)} images={len(diff.primitives['image'])} "
            f"conditions={sorted(diff.conditions.keys())}"
        )
        _log("COMPOSED ROLLOUT SMOKE PASSED ✅  (3-part [input, ar, diffusion] chain filled; lineage preserved)")
        return 0
    except Exception:
        _log("COMPOSED ROLLOUT SMOKE FAILED ❌")
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
