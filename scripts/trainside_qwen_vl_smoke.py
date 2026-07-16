#!/usr/bin/env python3
"""Trainside rollout+replay smoke for the Sample → Sample Qwen-VL AR pipeline (LIN-495).

Tier-2 representative for the AR / extra-input-modality migration (qwen_vl +
flux2_klein). Clone of ``trainside_ar_smoke.py`` with the Qwen-VL pipeline swapped
in. Runs a TEXT-ONLY request (the image-conditioning path is the same
``sample.conditioning()[1:]`` read; this smoke exercises the AR rollout→replay
contract and the new Sample boundary).

Asserts replay SELF-CONSISTENCY (not rollout==replay) for the same reason as the
Qwen3 AR smoke: the trainside production path uses ``old_logp_source="replay"``, so
old_logp and new_logp both come from replay and the GRPO ratio is 1.0 by
construction. The in-process autoregress bf16 record is an informational
rollout-fidelity diagnostic (generous catastrophe guard only). See
``trainside_ar_smoke.py`` for the full rationale.

No external inference server, no training loop, no reward. Run on a GPU pod, torch venv:

    QWEN_VL_PATH=/root/unirl/models/local/Qwen2.5-VL-7B-Instruct \
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/trainside_qwen_vl_smoke.py

Exits 0 on PASS, non-zero on any failed assertion or engine error.
"""

from __future__ import annotations

import os
import sys
import traceback

import torch

from unirl.algorithms.base import rollout_replay_logp_absdiff
from unirl.models.qwen_vl.config import QwenVLPipelineConfig
from unirl.models.qwen_vl.pipeline import QwenVLPipeline
from unirl.rollout.engine.trainside.engine import TrainsideRolloutEngine
from unirl.types.primitives import Texts
from unirl.types.sample import Part, Sample
from unirl.types.sampling import ARSamplingParams
from unirl.types.segments.text import TextSegment

# Replay must be deterministic at fixed weights (old_logp_source="replay" ⇒ ratio 1).
_SELF_CONSIST_MAX = 1e-3
# Generous catastrophe guard on the autoregress(bf16)-vs-replay(fp32) gap.
_ROLLOUT_GAP_SANITY_MAX = 15.0


def _log(msg: str) -> None:
    print(f"[trainside-qwen-vl] {msg}", flush=True)


def build_request_sample(n: int) -> Sample:
    """2 prompts, ``n`` completions each: ``[input, ar gen-shell]`` (P*n samples)."""
    prompts = ["Describe a sunny beach in one sentence.", "What is two plus two?"]
    input_part = Part.input(
        [f"p{i}" for i in range(len(prompts))], primitives={"text": Texts(texts=prompts)}, control={}
    )
    ar_params = ARSamplingParams(samples_per_prompt=n, temperature=0.7, max_new_tokens=48, top_p=0.9, top_k=20)
    return Sample.request(input_part).fork(n, sampling_params=ar_params)


def main() -> int:
    model_path = os.environ.get("QWEN_VL_PATH")
    if not model_path:
        _log("ERROR: set QWEN_VL_PATH to a local Qwen-VL dir")
        return 2
    _log(f"torch {torch.__version__} cuda={torch.version.cuda}; model_path={model_path}")

    config = QwenVLPipelineConfig(pretrained_model_ckpt_path=model_path, device="cuda:0")
    n = 2  # completions per prompt → exercises the sibling fan-out (n>1)
    try:
        _log("loading QwenVLPipeline.from_config (bundle on cuda:0) ...")
        pipeline = QwenVLPipeline.from_config(config)
        engine = TrainsideRolloutEngine(pipeline=pipeline, stage_attrs=("ar",))

        sample = build_request_sample(n)
        gen_in = sample.parts[-1]
        n_expect = 2 * n
        _log(f"request: {len(sample.parts)} parts; gen ids={list(gen_in.sample_ids)}")

        _log("calling engine.generate(sample) [rollout] ...")
        out = engine.generate(sample)

        # ---- rollout: the frontier gen Part is filled ----
        assert len(out.parts) == 2, f"expected [input, gen]; got {len(out.parts)} parts"
        gen = out.parts[-1]
        assert list(gen.sample_ids) == list(gen_in.sample_ids), "gen ids changed"
        assert len(gen.sample_ids) == n_expect, f"expected {n_expect} samples; got {len(gen.sample_ids)}"
        assert isinstance(gen.segment, TextSegment), f"segment must be TextSegment; got {type(gen.segment)}"
        assert gen.segment.log_probs is not None, "TextSegment.log_probs is None (no rollout logp recorded)"
        assert isinstance(gen.primitives.get("text"), Texts) and len(gen.primitives["text"].texts) == n_expect, (
            f"expected {n_expect} decoded texts"
        )
        assert gen.conditions, "trainside path stores Part.conditions for replay (re-typed via from_dict)"
        _log(f"rollout PASS: {n_expect} completions; conditions stored ✓")

        # ---- replay (twice) at fixed weights: the production old_logp_source=replay path ----
        _log("re-tokenizing conditions from the filled Sample and replaying (x2) ...")
        temperature = float(gen.sampling_params.temperature)
        turns, _images = out.vision_conditioning()
        control = out.parts[0].control
        model = pipeline.ar.trainable_module()
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                conds = pipeline._conditions_for(turns, control)
                new1 = pipeline.ar.replay(conds, segment=gen.segment, temperature=temperature)
                new2 = pipeline.ar.replay(conds, segment=gen.segment, temperature=temperature)
        finally:
            model.train(was_training)

        n_tokens = int(gen.segment.log_probs.numel())
        assert tuple(new1.shape) == (n_tokens,), f"replay logp shape {tuple(new1.shape)} != {n_tokens} response tokens"
        assert torch.isfinite(new1).all(), "replay produced non-finite log-probs"

        # PRIMARY invariant: replay deterministic at fixed weights → ratio 1.
        sc = rollout_replay_logp_absdiff(new2, new1)
        sc_mean, sc_max = sc["rollout_replay_logp_absdiff_mean"], sc["rollout_replay_logp_absdiff_max"]
        _log(
            f"replay self-consistency (old_logp_source=replay ⇒ ratio 1): mean|Δlogp|={sc_mean:.3e} max={sc_max:.3e} (threshold mean<{_SELF_CONSIST_MAX})"
        )
        assert sc_mean < _SELF_CONSIST_MAX, f"replay non-deterministic at fixed weights: mean|Δlogp|={sc_mean:.3e}"

        # INFORMATIONAL: autoregress(bf16) vs replay(fp32) rollout-fidelity gap.
        old = gen.segment.log_probs.to(device=new1.device, dtype=new1.dtype)
        rg = rollout_replay_logp_absdiff(new1, old)
        rg_mean, rg_max = rg["rollout_replay_logp_absdiff_mean"], rg["rollout_replay_logp_absdiff_max"]
        _log(
            f"[informational] autoregress(bf16)→replay(fp32) rollout-fidelity gap: mean|Δlogp|={rg_mean:.3e} max={rg_max:.3e} (expected ~5; NOT a ratio≈1 check)"
        )
        assert rg_mean < _ROLLOUT_GAP_SANITY_MAX, (
            f"autoregress↔replay gap {rg_mean:.3e} exceeds catastrophe guard {_ROLLOUT_GAP_SANITY_MAX} "
            f"— likely an alignment/conditions break, not precision"
        )

        _log(
            "TRAINSIDE QWEN-VL SMOKE PASSED ✅  (rollout fills the Sample; replay deterministic — old_logp_source=replay ratio≈1)"
        )
        return 0
    except Exception:
        _log("TRAINSIDE QWEN-VL SMOKE FAILED ❌")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
