#!/usr/bin/env python3
"""Generation-only smoke for the HI3 unified-model TWO-ENGINE path (LIN-480).

Exercises exactly the per-engine sub-requests
:meth:`unirl.trainer.unified_model.UnifiedModelTrainer._run_rollout_one` builds —
the consumer-side migration onto ``Sample``/``Part`` — WITHOUT the training-only
machinery (no Ray pool, FSDP base, reward, weight-sync, sleep/wake). In
``_run_rollout_one`` the two engines run SEQUENTIALLY (the AR engine fully
completes, then the DiT engine), so this boots ONE engine per process (set
``HI3_STAGE``), sidestepping the documented same-process two-engine device-env
collision. The two stage YAMLs pin DISJOINT cards (AR ``0,1,2,3`` / DiT
``4,5,6,7``) and ``clear_cuda_visible``, so run WITHOUT a CUDA_VISIBLE_DEVICES
restriction:

    # 1) AR think/recaption half (modality hi3_ar_recaption, cards 0-3):
    PRETRAINED_MODEL=/root/unirl/models/local/HunyuanImage-3.0-Instruct \
    HI3_STAGE=ar .venv/bin/python scripts/rollout_hi3_unified_smoke.py

    # 2) DiT image half (modality hi3_dit_recaption, cards 4-7):
    PRETRAINED_MODEL=/root/unirl/models/local/HunyuanImage-3.0-Instruct \
    HI3_STAGE=dit .venv/bin/python scripts/rollout_hi3_unified_smoke.py

The AR run writes its decoded recaptions to ``RECAP_PATH`` (default
``/tmp/hi3_recaptions.json``); the DiT run reads them if present (real AR→DiT
lineage chain) and otherwise falls back to a synthetic recaption so the DiT half
also runs standalone.

What each half proves about the migrated ``_run_rollout_one``:
- **ar**: the AR sub-request carries a PARAMS-ONLY diffusion shell ahead of the
  AR frontier (``.fork(1, image_shell.params).fork(1, ar_shell.params)``) so the
  ``carries_target_size`` ``hi3_ar_recaption`` adapter can read the DiT canvas
  height/width while only the AR stage runs — the bug fixed in LIN-480.
- **dit**: the DiT sub-request chains the recaption as a ``cot_text`` input Part
  via :meth:`Part.input_child`; the ``hi3_dit_recaption`` adapter reads it via
  ``cot_text_from_sample`` and renders M images per recaption.
"""

from __future__ import annotations

import json
import os
import sys
import traceback

import torch

from unirl.models.hunyuan_image3.config import HunyuanImage3PipelineConfig
from unirl.rollout.engine.vllm_omni.config import VLLMOmniEngineConfig
from unirl.rollout.engine.vllm_omni.engine import VLLMOmniRolloutEngine
from unirl.types.primitives import Images, Texts
from unirl.types.sample import Part, Sample
from unirl.types.sampling import ARSamplingParams, DiffusionSamplingParams
from unirl.types.segments.latent import LatentSegment
from unirl.types.segments.text import TextSegment

# Small but fan-out-exercising shape (cut from the recipe's N=2/M=2/4096-tok/16-step/1024px
# so the smoke is cheap): P prompts → P*N recaptions → P*N*M images.
_P = int(os.environ.get("HI3_P", "2"))
_N = int(os.environ.get("HI3_N", "2"))
_M = int(os.environ.get("HI3_M", "1"))
_HW = int(os.environ.get("HI3_HW", "512"))
_RECAP_PATH = os.environ.get("RECAP_PATH", "/tmp/hi3_recaptions.json")

_BASE_PROMPTS = [
    "a fluffy cat sitting on a sofa",
    "a serene mountain lake at sunrise",
    "a neon-lit street market at night",
    "a single red apple on a wooden table",
]


def _log(msg: str) -> None:
    print(f"[hi3-unified-smoke] {msg}", flush=True)


def _ar_params() -> ARSamplingParams:
    return ARSamplingParams(samples_per_prompt=_N, temperature=0.6, max_new_tokens=128, top_p=0.95, top_k=1024)


def _diff_params() -> DiffusionSamplingParams:
    # scheduler nulled, concrete sde_indices ride (mirrors _build_request_sample's
    # dataclasses.replace(diffusion, sde_indices=..., scheduler=None)).
    return DiffusionSamplingParams(
        num_inference_steps=8,
        guidance_scale=1.0,
        height=_HW,
        width=_HW,
        eta=1.0,
        samples_per_prompt=_M,
        seed=42,
        init_same_noise=False,
        sde_indices=[0, 1, 2],
    )


def _unified_request(prompts: list[str], rid: int = 0) -> Sample:
    """Pre-forked unified lineage ``[input, ar_shell(P*N), image_shell(P*N*M)]`` —
    mirrors :meth:`UnifiedModelTrainer._build_request_sample`."""
    input_part = Part.input([f"r{rid}:p{i}" for i in range(len(prompts))], primitives={"text": Texts(texts=prompts)})
    return (
        Sample.request(input_part)
        .fork(_ar_params().samples_per_prompt, sampling_params=_ar_params())
        .fork(_diff_params().samples_per_prompt, sampling_params=_diff_params())
    )


def _shells(sample: Sample):
    # gen_part raises a descriptive ValueError if a shell is missing (vs a bare
    # StopIteration leaking out of the helper).
    return sample.parts[0], sample.gen_part(ARSamplingParams), sample.gen_part(DiffusionSamplingParams)


def _build_engine(modality: str) -> VLLMOmniRolloutEngine:
    model_path = os.environ["PRETRAINED_MODEL"]
    model_config = HunyuanImage3PipelineConfig(pretrained_model_ckpt_path=model_path, model_precision="bf16", shift=3.0)
    engine_config = VLLMOmniEngineConfig(
        model_path=model_path,
        modality=modality,
        enable_sleep_mode=False,  # standalone rollout: never sleep/wake
    )
    _log(f"constructing VLLMOmniRolloutEngine (modality={modality}; boots HI3 80B on its stage-YAML cards) ...")
    return VLLMOmniRolloutEngine(engine_config, device=torch.device("cuda:0"), rank=0, model_config=model_config)


def run_ar(prompts: list[str], rid: int = 0) -> int:
    """AR half: P*N recaptions, with the params-only diffusion shell fix."""
    sample = _unified_request(prompts, rid)
    input_part, ar_shell, image_shell = _shells(sample)
    n_rec = int(ar_shell.sampling_params.samples_per_prompt)

    # Level 1: P*N pre-expanded prompts re-rooted flat (1:1); a params-only
    # diffusion shell rides AHEAD of the AR frontier so the carries_target_size
    # adapter can read height/width while only the AR stage runs.
    ar_texts = Texts(texts=[t for t in input_part.primitives["text"].texts for _ in range(n_rec)])
    n_ar = len(ar_texts.texts)
    ar_input = Part.input(
        [f"r{rid}:a{k}" for k in range(n_ar)], primitives={"text": ar_texts}, control=dict(input_part.control)
    )
    ar_request = (
        Sample.request(ar_input)
        .fork(1, sampling_params=image_shell.sampling_params)
        .fork(1, sampling_params=ar_shell.sampling_params)
    )
    _log(
        f"AR request: {len(ar_request.parts)} parts (input, params-only diff shell, ar frontier); "
        f"expecting {n_ar} recaptions (= P*N = {_P}*{_N})"
    )

    engine = None
    try:
        engine = _build_engine("hi3_ar_recaption")
        _log("calling engine.generate(ar_request) — think/recaption ...")
        ar_out = engine.generate(ar_request)
        ar_gen = ar_out.parts[-1]
        recaptions = ar_gen.primitives.get("text")

        assert isinstance(ar_gen.segment, TextSegment), f"ar segment must be TextSegment; got {type(ar_gen.segment)}"
        assert isinstance(recaptions, Texts), f"ar primitive must be Texts; got {type(recaptions)}"
        assert len(recaptions.texts) == n_ar, f"expected {n_ar} recaptions; got {len(recaptions.texts)}"

        with open(_RECAP_PATH, "w") as f:
            json.dump({"prompts": prompts, "n_rec": n_rec, "recaptions": list(recaptions.texts)}, f, ensure_ascii=False)
        _log(f"PASS: {n_ar} recaptions; ids={list(ar_gen.sample_ids)}")
        for k, t in enumerate(recaptions.texts):
            _log(f"  recap[{k}] (parent {ar_gen.group_ids[k]}) = {t[:80]!r}")
        _log(f"wrote recaptions -> {_RECAP_PATH}")
        _log("HI3 UNIFIED AR SMOKE PASSED ✅  (params-only diffusion shell → carries_target_size recaption)")
        return 0
    finally:
        if engine is not None:
            try:
                engine.shutdown()
                _log("engine shut down")
            except Exception:
                _log("engine.shutdown() raised (ignored)")


def run_dit(prompts: list[str], rid: int = 0) -> int:
    """DiT half: P*N*M images, recaption chained as cot_text via input_child."""
    sample = _unified_request(prompts, rid)
    input_part, ar_shell, image_shell = _shells(sample)
    n_rec = int(ar_shell.sampling_params.samples_per_prompt)
    n_img = int(image_shell.sampling_params.samples_per_prompt)
    n_ar = len(prompts) * n_rec

    # Source the recaptions: real AR output if the AR half wrote them, else synthetic.
    recaptions: list[str]
    if os.path.exists(_RECAP_PATH):
        with open(_RECAP_PATH) as f:
            blob = json.load(f)
        recaptions = list(blob.get("recaptions", []))
        if len(recaptions) == n_ar:
            _log(f"using {n_ar} REAL recaptions from {_RECAP_PATH} (AR→DiT lineage chain)")
        else:
            _log(f"{_RECAP_PATH} has {len(recaptions)} recaptions != P*N={n_ar}; falling back to synthetic")
            recaptions = []
    else:
        recaptions = []
    if not recaptions:
        recaptions = [
            f"A highly detailed, photorealistic, well-lit photograph of {prompts[i // n_rec]}." for i in range(n_ar)
        ]
        _log(f"using {n_ar} SYNTHETIC recaptions (DiT half runs standalone)")

    # Level 2: P*N*M pre-expanded (original prompt + recaption cot_text), re-rooted flat
    # (1:1) from the globally-unique image-shell lineage (flatten '/' into a legal root
    # id), mirroring _run_rollout_one so the engine-derived x_T key stays dp>1-unique.
    dit_prompts = Texts(texts=[prompts[i // n_rec] for i in range(n_ar) for _ in range(n_img)])
    dit_cot = Texts(texts=[recaptions[i] for i in range(n_ar) for _ in range(n_img)])
    n_total = len(dit_prompts.texts)
    dit_input = Part.input([sid.replace("/", "_") for sid in image_shell.sample_ids], primitives={"text": dit_prompts})
    cot_input = dit_input.input_child({"text": dit_cot})
    dit_request = Sample.request(dit_input, cot_input).fork(1, sampling_params=image_shell.sampling_params)
    _log(
        f"DiT request: {len(dit_request.parts)} parts (prompt input, cot_text input child, image frontier); "
        f"expecting {n_total} images (= P*N*M = {_P}*{_N}*{_M})"
    )

    engine = None
    try:
        engine = _build_engine("hi3_dit_recaption")
        _log("calling engine.generate(dit_request) — render images from injected recaption ...")
        dit_out = engine.generate(dit_request)
        img_gen = dit_out.parts[-1]

        assert len(img_gen.sample_ids) == n_total, (
            f"DiT must be 1:1: expected {n_total} images; got {len(img_gen.sample_ids)}"
        )
        assert isinstance(img_gen.segment, LatentSegment), (
            f"image segment must be LatentSegment; got {type(img_gen.segment)}"
        )
        assert img_gen.segment.latents is not None, "image LatentSegment.latents is None"
        assert isinstance(img_gen.primitives.get("image"), Images) and len(img_gen.primitives["image"]) == n_total, (
            "images decoded wrong"
        )
        assert img_gen.conditions, "image replay conditions empty (expected the fused HI3 conditions)"

        _log(
            f"PASS: {n_total} images; latents={tuple(img_gen.segment.latents.shape)} dtype={img_gen.segment.latents.dtype}"
        )
        _log(f"PASS: ids={list(img_gen.sample_ids)}")
        _log(f"PASS: image conditions={sorted(img_gen.conditions.keys())}")
        _log("HI3 UNIFIED DiT SMOKE PASSED ✅  (cot_text via input_child → rendered images)")
        return 0
    finally:
        if engine is not None:
            try:
                engine.shutdown()
                _log("engine shut down")
            except Exception:
                _log("engine.shutdown() raised (ignored)")


def main() -> int:
    if not os.environ.get("PRETRAINED_MODEL"):
        _log("ERROR: set PRETRAINED_MODEL to a local HunyuanImage-3.0-Instruct dir")
        return 2
    stage = os.environ.get("HI3_STAGE", "").lower()
    if stage not in ("ar", "dit"):
        _log("ERROR: set HI3_STAGE=ar or HI3_STAGE=dit")
        return 2
    _log(f"torch {torch.__version__} cuda={torch.version.cuda} device_count={torch.cuda.device_count()}")
    _log(f"stage={stage} P={_P} N={_N} M={_M} HW={_HW} model={os.environ['PRETRAINED_MODEL']}")
    prompts = [_BASE_PROMPTS[i % len(_BASE_PROMPTS)] for i in range(_P)]

    try:
        return run_ar(prompts) if stage == "ar" else run_dit(prompts)
    except Exception:
        _log(f"HI3 UNIFIED {stage.upper()} SMOKE FAILED ❌")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
