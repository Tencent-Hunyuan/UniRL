#!/usr/bin/env python3
"""Rollout-only e2e smoke for IT2I — image+text → edited image (LIN-454).

The gold-standard multi-input run: boots the REAL ``VLLMOmniRolloutEngine``
(modality ``hi3_it2i``, HunyuanImage-3 80B) and feeds it a hand-built multi-input
request ``Sample`` ``[text_input, image_input, ar_gen, image_gen]`` — the input
image rides as a chained input Part via ``Part.input_child``. Asserts the
returned 4-part chain fills the ar Part (AR recaption: TextSegment + Texts) and
the image Part (edited image: LatentSegment latents + Images + fused conditions),
with lineage p0 → p0/0 → p0/0/0 → p0/0/0/0 preserved.

HI3 it2i runs TP=4 AR + TP=4 DiT across ALL 8 GPUs (the adapter clears
CUDA_VISIBLE_DEVICES and pins stages via the stage YAML), so run WITHOUT a
CUDA_VISIBLE_DEVICES restriction:

    PRETRAINED_MODEL=/root/unirl/models/local/HunyuanImage-3.0-Instruct \
    .venv/bin/python scripts/rollout_hi3_it2i_smoke.py
"""

from __future__ import annotations

import os
import sys
import traceback

import torch

from unirl.models.hunyuan_image3.config import HunyuanImage3PipelineConfig
from unirl.rollout.engine.vllm_omni.config import VLLMOmniEngineConfig
from unirl.rollout.engine.vllm_omni.engine import VLLMOmniRolloutEngine
from unirl.types.primitives import Image, Images, Texts
from unirl.types.sample import Part, Sample
from unirl.types.sampling import ARSamplingParams, DiffusionSamplingParams
from unirl.types.segments.latent import LatentSegment
from unirl.types.segments.text import TextSegment

_HW = 512  # output canvas = the input image dims (HI3 reads h/w off the conditioning PIL)


def _log(msg: str) -> None:
    print(f"[hi3-it2i-smoke] {msg}", flush=True)


def build_request_sample() -> Sample:
    """A 1-prompt IT2I request ``[text, image_input, ar_gen, image_gen]`` via input_child."""
    text = Part.input(["p0"], primitives={"text": Texts(texts=["Make the cat wear a small red hat."])}, control={})
    image_in = text.input_child({"image": Images.from_list([Image(pixels=torch.rand(3, _HW, _HW))])})
    ar_params = ARSamplingParams(samples_per_prompt=1, temperature=0.7, max_new_tokens=64, top_p=0.9, top_k=20)
    diff_params = DiffusionSamplingParams(
        num_inference_steps=4,
        guidance_scale=2.5,
        height=_HW,
        width=_HW,
        eta=0.7,
        samples_per_prompt=1,
        seed=42,
        init_same_noise=False,
        sde_indices=[0, 1, 2],
    )
    return Sample.request(text, image_in).fork(1, sampling_params=ar_params).fork(1, sampling_params=diff_params)


def main() -> int:
    model_path = os.environ.get("PRETRAINED_MODEL")
    if not model_path:
        _log("ERROR: set PRETRAINED_MODEL to a local HunyuanImage-3.0-Instruct dir")
        return 2
    _log(f"torch {torch.__version__} cuda={torch.version.cuda} device_count={torch.cuda.device_count()}")
    _log(f"model_path={model_path}")

    model_config = HunyuanImage3PipelineConfig(pretrained_model_ckpt_path=model_path, model_precision="bf16", shift=3.0)
    engine_config = VLLMOmniEngineConfig(
        model_path=model_path,
        modality="hi3_it2i",
        enable_sleep_mode=False,  # standalone rollout: never sleep/wake
    )

    engine = None
    try:
        _log("constructing VLLMOmniRolloutEngine (hi3_it2i; boots HI3 80B across 8 GPUs) ...")
        engine = VLLMOmniRolloutEngine(engine_config, device=torch.device("cuda:0"), rank=0, model_config=model_config)
        sample = build_request_sample()
        _log(
            f"request: {len(sample.parts)} parts; "
            f"image input ids={list(sample.parts[1].sample_ids)} ar ids={list(sample.parts[2].sample_ids)} "
            f"image gen ids={list(sample.parts[3].sample_ids)}"
        )

        _log("calling engine.generate(sample) — AR recaption (text+image) → DiT edited image ...")
        out = engine.generate(sample)
        _log("generate returned; checking the filled 4-part Sample ...")

        assert len(out.parts) == 4, f"expected [text, image, ar, image]; got {len(out.parts)} parts"
        _text, _image_in, ar, img = out.parts
        assert list(ar.sample_ids) == ["p0/0/0"], f"ar ids: {list(ar.sample_ids)}"
        assert list(img.sample_ids) == ["p0/0/0/0"], f"image ids: {list(img.sample_ids)}"
        # AR recaption Part filled
        assert isinstance(ar.segment, TextSegment), f"ar segment must be TextSegment; got {type(ar.segment)}"
        # Edited-image Part filled
        assert isinstance(img.segment, LatentSegment), f"image segment must be LatentSegment; got {type(img.segment)}"
        assert img.segment.latents is not None, "image LatentSegment.latents is None"
        assert isinstance(img.primitives.get("image"), Images) and len(img.primitives["image"]) == 1, (
            "edited image decoded wrong"
        )
        assert img.conditions, "image replay conditions empty (expected the fused HI3 conditions)"

        _log(f"PASS: edited-image latents={tuple(img.segment.latents.shape)} dtype={img.segment.latents.dtype}")
        _log(
            f"PASS: ar recaption decoded="
            f"{(ar.primitives['text'].texts[0][:80] if isinstance(ar.primitives.get('text'), Texts) else None)!r}"
        )
        _log(f"PASS: image conditions={sorted(img.conditions.keys())}")
        _log("HI3 IT2I ROLLOUT SMOKE PASSED ✅  (image+text → AR recaption → edited image; chain filled)")
        return 0
    except Exception:
        _log("HI3 IT2I ROLLOUT SMOKE FAILED ❌")
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
