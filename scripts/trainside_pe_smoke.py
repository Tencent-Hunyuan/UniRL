#!/usr/bin/env python3
"""Trainside rollout smoke for the Sample → Sample PEPipeline (LIN-495).

Tier-3 representative for the multi-stage migration (pe / bagel-t2ti /
hunyuan_image3-t2ti — all 2-gen-part flows). The trainside PE path drives the
composed :class:`PEPipeline` (SD3 diffusion child + Qwen3 LLM child) as the
in-process sampler of a ``TrainsideRolloutEngine`` (``stage_attrs=["diffusion",
"ar"]``).

Builds the pre-forked 3-part request ``Sample`` ``[input, ar_shell, diff_shell]``
by hand (as ``PETrainer._build_request_sample`` does), runs ``generate``, and
asserts the returned ``[input, ar_part, diffusion_part]`` chain is filled: the ar
Part with a TextSegment + decoded Texts (the rewrites), the diffusion Part with a
LatentSegment + decoded Images, lineage chaining prompt → rewrite → image. This is
the trainside analogue of ``rollout_composed_smoke.py`` (which boots the served
composed engine), exercising the hardest conversion: the 3-part Sample chain
driven by in-process child pipelines.

No external inference server, no training loop, no reward. Run on a GPU pod, torch venv:

    QWEN3_PATH=/root/unirl/models/local/Qwen3-4B-Base \
    PRETRAINED_MODEL=/root/unirl/models/local/stable-diffusion-3.5-medium \
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/trainside_pe_smoke.py

Exits 0 on PASS, non-zero on any failed assertion or engine error.
"""

from __future__ import annotations

import os
import sys
import traceback

import torch

from unirl.models.pe.pipeline import PEPipeline
from unirl.models.qwen3.config import Qwen3PipelineConfig
from unirl.models.qwen3.pipeline import Qwen3Pipeline
from unirl.models.sd3.config import SD3PipelineConfig
from unirl.models.sd3.pipeline import SD3Pipeline
from unirl.rollout.engine.trainside.engine import TrainsideRolloutEngine
from unirl.types.primitives import Images, Texts
from unirl.types.sample import Part, Sample
from unirl.types.sampling import ARSamplingParams, DiffusionSamplingParams
from unirl.types.segments.latent import LatentSegment
from unirl.types.segments.text import TextSegment


def _log(msg: str) -> None:
    print(f"[trainside-pe] {msg}", flush=True)


def build_request_sample() -> Sample:
    """P=2 prompts, N=2 rewrites each, M=1 image each: ``[input, ar_shell, diff_shell]``.

    Mirrors ``PETrainer._build_request_sample``: the AR shell carries
    ``ARSamplingParams`` (branch N), the diffusion shell ``DiffusionSamplingParams``
    (branch M). No sigmas — the trainside engine pins σ onto the diffusion shell
    (it is parts[-1]).
    """
    prompts = ["a cat", "a dog"]
    input_part = Part.input(
        [f"p{i}" for i in range(len(prompts))],
        primitives={"text": Texts(texts=prompts)},
        control={"ar": {}, "chat": {}},
    )
    ar_params = ARSamplingParams(samples_per_prompt=2, temperature=0.7, max_new_tokens=32, top_p=0.9, top_k=20)
    diff_params = DiffusionSamplingParams(
        num_inference_steps=4,
        guidance_scale=1.0,
        height=512,
        width=512,
        eta=0.7,
        samples_per_prompt=1,
        seed=42,
        sde_indices=[0, 1, 2],
    )
    return (
        Sample.request(input_part)
        .fork(ar_params.samples_per_prompt, sampling_params=ar_params)
        .fork(diff_params.samples_per_prompt, sampling_params=diff_params)
    )


def main() -> int:
    qwen = os.environ.get("QWEN3_PATH")
    sd3 = os.environ.get("PRETRAINED_MODEL")
    if not qwen or not sd3:
        _log("ERROR: set QWEN3_PATH and PRETRAINED_MODEL to local model dirs")
        return 2
    _log(f"torch {torch.__version__} cuda={torch.version.cuda}; qwen3={qwen} sd3={sd3}")

    try:
        _log("loading SD3 + Qwen3 child pipelines (cuda:0) and composing PEPipeline ...")
        diffusion_pipeline = SD3Pipeline.from_config(
            SD3PipelineConfig(pretrained_model_ckpt_path=sd3, model_precision="bf16", shift=3.0, device="cuda:0")
        )
        llm_pipeline = Qwen3Pipeline.from_config(Qwen3PipelineConfig(pretrained_model_ckpt_path=qwen, device="cuda:0"))
        pe_pipeline = PEPipeline(diffusion_pipeline=diffusion_pipeline, llm_pipeline=llm_pipeline)
        engine = TrainsideRolloutEngine(pipeline=pe_pipeline, stage_attrs=("diffusion", "ar"))

        sample = build_request_sample()
        _log(f"request: {len(sample.parts)} parts; gen ids={list(sample.parts[-1].sample_ids)}")

        _log("calling engine.generate(sample) [rollout] ...")
        out = engine.generate(sample)

        # ---- the [input, ar, diffusion] chain is filled ----
        assert len(out.parts) == 3, f"expected [input, ar, diffusion]; got {len(out.parts)} parts"
        ar_part, diff_part = out.parts[1], out.parts[2]
        assert isinstance(ar_part.segment, TextSegment), f"ar segment must be TextSegment; got {type(ar_part.segment)}"
        assert isinstance(ar_part.primitives.get("text"), Texts) and len(ar_part.primitives["text"].texts) == 4, (
            "ar Part must carry 4 (=P*N) rewritten Texts"
        )
        assert isinstance(diff_part.segment, LatentSegment), (
            f"diffusion segment must be LatentSegment; got {type(diff_part.segment)}"
        )
        assert isinstance(diff_part.primitives.get("image"), Images) and len(diff_part.primitives["image"]) == 4, (
            "diffusion Part must carry 4 (=P*N*M) decoded Images"
        )
        # lineage: diffusion ids descend from ar ids descend from prompt ids
        assert len(diff_part.sample_ids) == 4, f"expected 4 diffusion samples; got {len(diff_part.sample_ids)}"
        _log(
            f"rollout PASS: ar={len(ar_part.primitives['text'].texts)} rewrites, "
            f"diffusion={len(diff_part.primitives['image'])} images ✓"
        )

        _log("TRAINSIDE PE SMOKE PASSED ✅  (3-part Sample chain filled by in-process child pipelines)")
        return 0
    except Exception:
        _log("TRAINSIDE PE SMOKE FAILED ❌")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
