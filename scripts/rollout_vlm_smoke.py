#!/usr/bin/env python3
"""Rollout-only e2e smoke for MULTI-INPUT image+text (sglang VLM, LIN-454).

The cheap multi-input mechanism check: boots the REAL ``SGLangRolloutEngine``
(model_family ``vlm``, Qwen2.5-VL) and feeds it a hand-built multi-input request
``Sample`` ``[text_input, image_input, ar_gen]`` (the image rides as a chained
input Part via ``Part.input_child``). Asserts the frontier gen Part is filled
with a ``TextSegment`` + decoded ``Texts`` + replay conditions carrying the
multimodal ``pixel_values`` / ``image_grid_thw``. Validates the multi-input
Sample + image-conditioning path that IT2I also needs, on a 7B/1-GPU model.

    QWEN_VL_PATH=/root/unirl/models/local/Qwen2.5-VL-7B-Instruct \
    CUDA_VISIBLE_DEVICES=0 .venv-sglang/bin/python scripts/rollout_vlm_smoke.py
"""

from __future__ import annotations

import os
import sys
import traceback

import torch

from unirl.rollout.engine.sglang.config import SGLangEngineConfig
from unirl.rollout.engine.sglang.engine import SGLangRolloutEngine
from unirl.types.primitives import Image, Images, Texts
from unirl.types.sample import Part, Sample
from unirl.types.sampling import ARSamplingParams
from unirl.types.segments.text import TextSegment

_IMAGE_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"


def _log(msg: str) -> None:
    print(f"[vlm-smoke] {msg}", flush=True)


def build_request_sample() -> Sample:
    """A 1-prompt image+text request: ``[text_input, image_input, ar_gen]`` via input_child."""
    text = Part.input(["p0"], primitives={"text": Texts(texts=["Describe this image in one short sentence."])})
    # One synthetic 224x224 RGB image (content irrelevant — this validates the path, not accuracy).
    image_in = text.input_child({"image": Images.from_list([Image(pixels=torch.rand(3, 224, 224))])})
    ar_params = ARSamplingParams(samples_per_prompt=1, temperature=0.7, max_new_tokens=48, top_p=0.9, top_k=20)
    return Sample.request(text, image_in).fork(1, sampling_params=ar_params)


def main() -> int:
    model_path = os.environ.get("QWEN_VL_PATH")
    if not model_path:
        _log("ERROR: set QWEN_VL_PATH to a local Qwen2.5-VL dir")
        return 2
    _log(f"torch {torch.__version__} cuda={torch.version.cuda}; model_path={model_path}")

    config = SGLangEngineConfig(
        pretrained_model_ckpt_path=model_path,
        backend="native",
        tp_size=1,
        image_token=_IMAGE_TOKEN,  # the VLM switch → model_family auto-derives to "vlm"
        max_new_tokens=48,
        temperature=0.7,
        top_p=0.9,
        concurrency=4,
        engine_kwargs={"mem_fraction_static": 0.6, "skip_server_warmup": True, "disable_cuda_graph": True},
    )

    engine = None
    try:
        _log("constructing SGLangRolloutEngine (vlm; boots sglang + loads Qwen2.5-VL) ...")
        engine = SGLangRolloutEngine(config, rank=0)
        sample = build_request_sample()
        _log(
            f"request: {len(sample.parts)} parts; image input ids={list(sample.parts[1].sample_ids)} "
            f"gen ids={list(sample.parts[-1].sample_ids)}"
        )

        _log("calling engine.generate(sample) ...")
        out = engine.generate(sample)
        _log("generate returned; checking the filled Sample ...")

        assert len(out.parts) == 3, f"expected [text, image, ar]; got {len(out.parts)} parts"
        gen = out.parts[-1]
        assert isinstance(gen.segment, TextSegment), f"segment must be TextSegment; got {type(gen.segment)}"
        assert isinstance(gen.primitives.get("text"), Texts) and len(gen.primitives["text"].texts) == 1, (
            "decoded Texts wrong"
        )
        assert gen.conditions, "replay conditions empty"
        # The multimodal replay conditions must carry the vision tensors.
        assert "pixel_values" in gen.conditions and "image_grid_thw" in gen.conditions, (
            f"VLM conditions must carry pixel_values + image_grid_thw; got {sorted(gen.conditions.keys())}"
        )

        _log(f"PASS: decoded={gen.primitives['text'].texts[0][:100]!r}")
        _log(f"PASS: conditions={sorted(gen.conditions.keys())} (multimodal vision tensors present)")
        _log("VLM MULTI-INPUT ROLLOUT SMOKE PASSED ✅  (image+text Sample → TextSegment + pixel_values)")
        return 0
    except Exception:
        _log("VLM MULTI-INPUT ROLLOUT SMOKE FAILED ❌")
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
