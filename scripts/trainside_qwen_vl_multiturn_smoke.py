#!/usr/bin/env python3
"""Trainside VLM MULTI-TURN encode smoke (LIN-503 gap C conjugate — VLM path).

Builds a ``[user(text), image, assistant, tool, gen]`` trajectory, runs the
in-process ``QwenVLPipeline``'s ``generate``, and proves it conditions on the FULL
multimodal trajectory: conditions carry ``pixel_values`` / ``image_grid_thw`` (the
image) AND the decoded ``conditions["prompt"]`` contains the user/assistant/tool
text turns — not just the root prompt. Also checks replay self-consistency
(ratio≈1) on the multi-turn conditions.

    QWEN_VL_PATH=/root/unirl/models/local/Qwen2.5-VL-3B-Instruct \
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/trainside_qwen_vl_multiturn_smoke.py

Exits 0 on PASS, non-zero on any failed assertion or engine error.
"""

from __future__ import annotations

import os
import re
import sys
import traceback

import torch

from unirl.algorithms.base import rollout_replay_logp_absdiff
from unirl.models.qwen_vl.config import QwenVLPipelineConfig
from unirl.models.qwen_vl.pipeline import QwenVLPipeline
from unirl.rollout.engine.trainside.engine import TrainsideRolloutEngine
from unirl.types.primitives import Image, Images, Texts
from unirl.types.sample import Part, Sample
from unirl.types.sampling import ARSamplingParams

_USER = "What color is this image?"
_ASSISTANT = "Let me look at it."
_TOOL = "The image is solid red."
_SELF_CONSIST_MAX = 1e-3


def _log(msg: str) -> None:
    print(f"[trainside-vlm-mt] {msg}", flush=True)


def _red_image(h: int = 448, w: int = 448) -> Images:
    px = torch.zeros(3, h, w)
    px[0] = 1.0  # solid red
    return Images.from_list([Image(pixels=px)])


def build_multiturn_vlm_sample() -> Sample:
    text = Part.input(["p0"], primitives={"text": Texts(texts=[_USER])}, role="user", control={})
    img = text.input_child({"image": _red_image()}, role="user")
    asst = img.fork(1, sampling_params=ARSamplingParams()).fill(primitives={"text": Texts(texts=[_ASSISTANT])})
    tool = asst.input_child({"text": Texts(texts=[_TOOL])}, role="tool")
    ar_params = ARSamplingParams(samples_per_prompt=1, temperature=0.7, max_new_tokens=32, top_p=0.9, top_k=20)
    return Sample(parts=[text, img, asst, tool, tool.fork(1, sampling_params=ar_params)])


def main() -> int:
    model_path = os.environ.get("QWEN_VL_PATH")
    if not model_path:
        _log("ERROR: set QWEN_VL_PATH to a local Qwen-VL dir")
        return 2
    _log(f"torch {torch.__version__} cuda={torch.version.cuda}; model_path={model_path}")
    import unirl.models.qwen_vl.chat_template as _ct

    _log(f"qwen_vl.chat_template from: {_ct.__file__}")

    config = QwenVLPipelineConfig(pretrained_model_ckpt_path=model_path, device="cuda:0")
    try:
        _log("loading QwenVLPipeline.from_config (bundle on cuda:0) ...")
        pipeline = QwenVLPipeline.from_config(config)
        engine = TrainsideRolloutEngine(pipeline=pipeline, stage_attrs=("ar",))

        sample = build_multiturn_vlm_sample()
        turns, imgs = sample.vision_conditioning()
        _log(f"request: {len(sample.parts)} parts; turns={[t.role for t in turns]}; image_turns={len(imgs)}")

        _log("calling engine.generate(sample) [rollout] ...")
        out = engine.generate(sample)
        gen = out.parts[-1]
        assert gen.conditions and "prompt" in gen.conditions, "no stored prompt conditions"
        assert gen.conditions.get("pixel_values") is not None, (
            "pixel_values DROPPED — image not carried into conditions"
        )
        assert gen.conditions.get("image_grid_thw") is not None, "image_grid_thw missing from conditions"

        tok = pipeline.bundle.processor.tokenizer
        prompt_ids = gen.conditions["prompt"].input_ids[0]
        prompt_text = tok.decode([int(t) for t in prompt_ids.tolist()], skip_special_tokens=False)
        shown = re.sub(r"(<\|image_pad\|>)+", "<|image_pad|>*N", prompt_text)
        _log("--- decoded conditions['prompt'] (image-pad run collapsed) ---")
        _log(shown)
        _log("-------------------------------------------------------------")

        assert "<|image_pad|>" in prompt_text, "image placeholder not expanded in the prompt"
        assert _USER in prompt_text, "USER text turn missing from the encoded prompt"
        assert _ASSISTANT in prompt_text, "ASSISTANT turn DROPPED — VLM trainside encode flattened to the root prompt"
        assert _TOOL in prompt_text, "TOOL turn DROPPED — VLM trainside encode flattened to the root prompt"
        assert prompt_text.index(_USER) < prompt_text.index(_ASSISTANT) < prompt_text.index(_TOOL), (
            "turns out of lineage order in the encoded prompt"
        )
        _log("VLM MULTI-TURN ENCODE PASS: image + user → assistant → tool all carried ✓")
        _log(f"completion: {gen.primitives['text'].texts[0]!r}")

        # ---- replay self-consistency on the multi-turn VLM conditions ----
        temperature = float(gen.sampling_params.temperature)
        model = pipeline.ar.trainable_module()
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                turns2, _imgs2 = out.vision_conditioning()
                conds = pipeline._conditions_for(turns2, out.parts[0].control)
                new1 = pipeline.ar.replay(conds, segment=gen.segment, temperature=temperature)
                new2 = pipeline.ar.replay(conds, segment=gen.segment, temperature=temperature)
        finally:
            model.train(was_training)
        assert torch.isfinite(new1).all(), "replay produced non-finite log-probs"
        sc = rollout_replay_logp_absdiff(new2, new1)["rollout_replay_logp_absdiff_mean"]
        _log(f"replay self-consistency (multi-turn VLM conds): mean|Δlogp|={sc:.3e} (threshold <{_SELF_CONSIST_MAX})")
        assert sc < _SELF_CONSIST_MAX, f"replay non-deterministic at fixed weights: mean|Δlogp|={sc:.3e}"

        _log("TRAINSIDE VLM MULTI-TURN SMOKE PASSED ✅  (encode carries image + full trajectory; replay deterministic)")
        return 0
    except Exception:
        _log("TRAINSIDE VLM MULTI-TURN SMOKE FAILED ❌")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
