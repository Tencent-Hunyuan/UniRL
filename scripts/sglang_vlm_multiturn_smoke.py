#!/usr/bin/env python3
"""sglang VLM MULTI-TURN encode smoke (LIN-503 gap C — VLM path, on GPU).

Boots the real ``SGLangRolloutEngine`` (native, ``model_family=vlm``) on Qwen2.5-VL,
builds a ``[user(text), image, assistant, tool, gen]`` trajectory, runs ``generate``,
and proves the VLM engine conditions on the FULL multimodal trajectory: the captured
conditions carry ``pixel_values`` / ``image_grid_thw`` (the image) AND the decoded
``conditions["prompt"]`` contains the user/assistant/tool text turns — not just the
root prompt (the pre-fix behavior read ``parts[0]`` and emitted one user message).

Coexists with a running GPU burn via a high ``mem_fraction_static`` + no CUDA graphs.
Run on the pod with the sglang venv:

    QWEN_VL_PATH=/root/unirl/models/local/Qwen2.5-VL-3B-Instruct \
    CUDA_VISIBLE_DEVICES=0 .venv-sglang/bin/python scripts/sglang_vlm_multiturn_smoke.py

Exits 0 on PASS, non-zero on any failed assertion or engine error.
"""

from __future__ import annotations

import os
import re
import sys
import traceback

import torch

from unirl.rollout.engine.sglang.config import SGLangEngineConfig
from unirl.rollout.engine.sglang.engine import SGLangRolloutEngine
from unirl.types.primitives import Image, Images, Texts
from unirl.types.sample import Part, Sample
from unirl.types.sampling import ARSamplingParams

_USER = "What color is this image?"
_ASSISTANT = "Let me look at it."
_TOOL = "The image is solid red."
# Qwen2.5-VL VLM switch (selects model_family=vlm); the processor does the real expand.
_QWEN_VL_IMAGE_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"


def _log(msg: str) -> None:
    print(f"[sglang-vlm-mt] {msg}", flush=True)


def _red_image(h: int = 448, w: int = 448) -> Images:
    px = torch.zeros(3, h, w)
    px[0] = 1.0  # R channel high → solid red
    return Images.from_list([Image(pixels=px)])


def build_multiturn_vlm_sample() -> Sample:
    """A 4-turn multimodal trajectory ``[user(text), image, assistant, tool]`` + gen."""
    text = Part.input(["p0"], primitives={"text": Texts(texts=[_USER])}, role="user", control={})
    img = text.input_child({"image": _red_image()}, role="user")
    asst = img.fork(1, sampling_params=ARSamplingParams()).fill(primitives={"text": Texts(texts=[_ASSISTANT])})
    tool = asst.input_child({"text": Texts(texts=[_TOOL])}, role="tool")
    ar_params = ARSamplingParams(samples_per_prompt=1, temperature=0.7, max_new_tokens=32, top_p=0.9, top_k=20)
    return Sample(parts=[text, img, asst, tool, tool.fork(1, sampling_params=ar_params)])


def main() -> int:
    model_path = os.environ.get("QWEN_VL_PATH")
    if not model_path:
        _log("ERROR: set QWEN_VL_PATH to a local Qwen2.5-VL dir")
        return 2
    _log(f"torch {torch.__version__} cuda={torch.version.cuda}; model_path={model_path}")
    import unirl.rollout.engine.sglang.adapters.vlm as _v

    _log(f"sglang vlm adapter from: {_v.__file__}")

    config = SGLangEngineConfig(
        pretrained_model_ckpt_path=model_path,
        image_token=_QWEN_VL_IMAGE_TOKEN,  # → model_family=vlm
        backend="native",
        max_new_tokens=32,
        temperature=0.7,
        top_p=0.9,
        top_k=20,
        engine_kwargs={"mem_fraction_static": 0.9, "disable_cuda_graph": True},
    )
    _log(f"model_family resolved: {config.model_family}")
    engine = None
    try:
        _log("booting SGLangRolloutEngine (native, vlm, mem_fraction_static=0.9) ...")
        engine = SGLangRolloutEngine(config, rank=0)

        sample = build_multiturn_vlm_sample()
        turns, imgs = sample.vision_conditioning()
        _log(f"request: {len(sample.parts)} parts; turns={[t.role for t in turns]}; image_turns={len(imgs)}")

        _log("calling engine.generate(sample) [rollout] ...")
        out = engine.generate(sample)
        gen = out.parts[-1]
        assert gen.conditions and "prompt" in gen.conditions, "no stored prompt conditions"

        # ---- image carried into the replay conditions ----
        assert "pixel_values" in gen.conditions, "pixel_values DROPPED — image not carried into conditions"
        assert "image_grid_thw" in gen.conditions, "image_grid_thw missing from conditions"

        # ---- text turns carried: decode the processor-expanded prompt ----
        tok = engine.adapter._tokenizer
        prompt_ids = gen.conditions["prompt"].input_ids[0]
        prompt_text = tok.decode([int(t) for t in prompt_ids.tolist()], skip_special_tokens=False)
        shown = re.sub(r"(<\|image_pad\|>)+", "<|image_pad|>*N", prompt_text)
        _log("--- decoded conditions['prompt'] (image-pad run collapsed) ---")
        _log(shown)
        _log("-------------------------------------------------------------")

        assert "<|image_pad|>" in prompt_text, "image placeholder not expanded in the prompt"
        assert _USER in prompt_text, "USER text turn missing from the encoded prompt"
        assert _ASSISTANT in prompt_text, "ASSISTANT turn DROPPED — VLM encode flattened to the root prompt"
        assert _TOOL in prompt_text, "TOOL turn DROPPED — VLM encode flattened to the root prompt"
        assert prompt_text.index(_USER) < prompt_text.index(_ASSISTANT) < prompt_text.index(_TOOL), (
            "turns out of lineage order in the encoded prompt"
        )
        _log("VLM MULTI-TURN ENCODE PASS: image + user → assistant → tool all carried ✓")
        _log(f"completion: {gen.primitives['text'].texts[0]!r}")
        _log("SGLANG VLM MULTI-TURN SMOKE PASSED ✅  (the VLM engine conditions on the full multimodal trajectory)")
        return 0
    except Exception:
        _log("SGLANG VLM MULTI-TURN SMOKE FAILED ❌")
        traceback.print_exc()
        return 1
    finally:
        if engine is not None:
            try:
                engine.shutdown()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
