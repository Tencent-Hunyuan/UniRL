#!/usr/bin/env python3
"""Rollout-only e2e smoke for the Sample → Sample vllm_omni engine (LIN-454).

Boots the REAL ``VLLMOmniRolloutEngine`` (modality ``sd3_t2i``, SD3.5-medium)
in a single process, builds a request ``Sample`` BY HAND (the trainer that
normally builds requests is deferred in this refactor), runs ``generate``, and
asserts the returned ``Sample``'s frontier gen Part is filled with real image
latents + decoded images + replay conditions, σ-verified.

No training, no reward, no weight sync — purely "does the converted rollout
engine produce a valid Sample from a real backend rollout". The ``@distributed``
decorator on ``generate`` is a transparent passthrough off the Handle, so a
direct single-process call runs the real conversion logic.

Run on a GPU pod (1 free GPU), in the vllm venv:

    PRETRAINED_MODEL=/root/unirl/models/local/stable-diffusion-3.5-medium \
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/rollout_sd3_smoke.py

Exits 0 on PASS, non-zero on any failed assertion or engine error.
"""

from __future__ import annotations

import os
import sys
import traceback

import torch

from unirl.models.sd3.config import SD3PipelineConfig
from unirl.rollout.engine.vllm_omni.config import VLLMOmniEngineConfig
from unirl.rollout.engine.vllm_omni.engine import VLLMOmniRolloutEngine
from unirl.types.primitives import Images, Texts
from unirl.types.sample import Part, Sample
from unirl.types.sampling import DiffusionSamplingParams
from unirl.types.segments.latent import LatentSegment


def _log(msg: str) -> None:
    print(f"[sd3-smoke] {msg}", flush=True)


def build_request_sample() -> Sample:
    """A 2-prompt, 1-sample-each SD3 request: ``[input, diffusion gen-shell]``.

    vllm_omni runs ``num_outputs_per_prompt=1`` (the caller pre-expands), so a
    branch-1 fork gives one gen sample per prompt.
    """
    prompts = ["a photo of a red apple on a wooden table", "an astronaut riding a horse on the moon"]
    input_part = Part.input([f"p{i}" for i in range(len(prompts))], primitives={"text": Texts(texts=prompts)})
    diff_params = DiffusionSamplingParams(
        num_inference_steps=4,
        guidance_scale=1.0,  # CFG off (matches the sd3_vllmomni example)
        height=512,
        width=512,
        eta=0.7,
        samples_per_prompt=1,
        seed=42,
        init_same_noise=False,
        sde_indices=[0, 1, 2],  # record SDE log-probs on the first 3 of 4 steps
    )
    return Sample.request(input_part).fork(1, sampling_params=diff_params)


def main() -> int:
    model_path = os.environ.get("PRETRAINED_MODEL")
    if not model_path:
        _log("ERROR: set PRETRAINED_MODEL to a local SD3.5-medium dir")
        return 2
    _log(f"torch {torch.__version__} cuda={torch.version.cuda} device_count={torch.cuda.device_count()}")
    _log(f"model_path={model_path}")

    model_config = SD3PipelineConfig(pretrained_model_ckpt_path=model_path, model_precision="bf16", shift=3.0)
    engine_config = VLLMOmniEngineConfig(
        model_path=model_path,
        modality="sd3_t2i",
        enable_sleep_mode=False,  # standalone rollout: never sleep/wake
    )

    engine = None
    try:
        _log("constructing VLLMOmniRolloutEngine (boots the Omni orchestrator + loads SD3.5) ...")
        engine = VLLMOmniRolloutEngine(
            engine_config,
            device=torch.device("cuda:0"),
            rank=0,
            model_config=model_config,
        )
        _log("engine constructed; building request Sample ...")
        sample = build_request_sample()
        gen_in = sample.parts[-1]
        _log(f"request: {len(sample.parts)} parts; gen shell ids={list(gen_in.sample_ids)}")

        _log("calling engine.generate(sample) ...")
        out = engine.generate(sample)
        _log("generate returned; checking the filled Sample ...")

        # ---- structural assertions on the returned Sample ----
        assert len(out.parts) == 2, f"expected [input, gen]; got {len(out.parts)} parts"
        gen = out.parts[-1]
        assert list(gen.sample_ids) == list(gen_in.sample_ids), (
            f"gen ids changed: {list(gen.sample_ids)} != {list(gen_in.sample_ids)}"
        )
        assert isinstance(gen.segment, LatentSegment), f"segment must be LatentSegment; got {type(gen.segment)}"
        assert gen.segment.latents is not None, "LatentSegment.latents is None (no trajectory captured)"
        assert isinstance(gen.primitives.get("image"), Images), (
            f"decoded image primitive must be Images; got {type(gen.primitives.get('image'))}"
        )
        assert len(gen.primitives["image"]) == 2, f"expected 2 decoded images; got {len(gen.primitives['image'])}"
        assert gen.conditions, "replay conditions empty (expected at least 'text')"

        lat = gen.segment.latents
        _log(f"PASS: latents shape={tuple(lat.shape)} dtype={lat.dtype}")
        _log(f"PASS: images={len(gen.primitives['image'])} conditions={sorted(gen.conditions.keys())}")
        _log(
            f"PASS: sigmas pinned len={None if gen.sampling_params.sigmas is None else int(gen.sampling_params.sigmas.shape[0])}"
        )
        _log("ROLLOUT SMOKE PASSED ✅  (σ-verified inside build_image_segment; Sample filled correctly)")
        return 0
    except Exception:
        _log("ROLLOUT SMOKE FAILED ❌")
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
