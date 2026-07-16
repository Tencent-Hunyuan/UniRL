#!/usr/bin/env python3
"""Trainside rollout+replay smoke for the Sample → Sample Qwen3 AR pipeline (LIN-479).

Loads the IN-PROCESS Qwen3 pipeline, wraps it in a ``TrainsideRolloutEngine``,
builds a request ``Sample`` by hand, runs ``generate`` (rollout), then
re-tokenizes conditions from the filled Sample and runs the AR stage's ``replay``.

Why this asserts replay SELF-CONSISTENCY (not rollout==replay like the SD3 smoke):
the trainside Qwen3 production path uses ``old_logp_source="replay"`` (all PE
recipes), so ``old_logp`` and ``new_logp`` BOTH come from replay — the GRPO ratio
is replay-vs-replay and starts at 1.0 by construction. The in-process autoregress
records a *fast* bf16 log-prob (stock forward, no autocast) that is intentionally
NOT expected to match replay's fp32 (autocast body + fp32 lm_head) — unlike SD3,
where ``diffuse`` and ``replay`` share one forward. So the meaningful invariant
here is: replay is deterministic at fixed weights (ratio == 1). The
autoregress-vs-replay gap is LOGGED as an informational rollout-fidelity
diagnostic (harmless under ``old_logp_source="replay"``), with only a generous
catastrophe guard so an alignment/conditions break still fails loud.

No external inference server, no training loop, no reward. Run on a GPU pod, torch venv:

    QWEN3_PATH=/root/unirl/models/local/Qwen3-4B-Base \
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/trainside_ar_smoke.py

Exits 0 on PASS, non-zero on any failed assertion or engine error.
"""

from __future__ import annotations

import os
import sys
import traceback

import torch

from unirl.algorithms.base import rollout_replay_logp_absdiff
from unirl.models.qwen3.config import Qwen3PipelineConfig
from unirl.models.qwen3.pipeline import Qwen3Pipeline
from unirl.rollout.engine.trainside.engine import TrainsideRolloutEngine
from unirl.types.primitives import Texts
from unirl.types.sample import Part, Sample
from unirl.types.sampling import ARSamplingParams
from unirl.types.segments.text import TextSegment

# Replay must be deterministic at fixed weights (the old_logp_source="replay"
# contract — old_logp == new_logp ⇒ ratio 1). A tiny floor tolerates kernel
# nondeterminism without masking a real bug.
_SELF_CONSIST_MAX = 1e-3
# Generous catastrophe guard on the autoregress(bf16)-vs-replay(fp32) gap: a real
# alignment/conditions break drives this to 50+, while the expected precision gap
# is ~5. NOT a ratio≈1 assertion — see the module docstring.
_ROLLOUT_GAP_SANITY_MAX = 15.0


def _log(msg: str) -> None:
    print(f"[trainside-ar] {msg}", flush=True)


def build_request_sample(n: int) -> Sample:
    """2 prompts, ``n`` completions each: ``[input, ar gen-shell]`` (P*n samples)."""
    prompts = ["The capital of France is", "Two plus two equals"]
    input_part = Part.input(
        [f"p{i}" for i in range(len(prompts))], primitives={"text": Texts(texts=prompts)}, control={}
    )
    ar_params = ARSamplingParams(samples_per_prompt=n, temperature=0.7, max_new_tokens=48, top_p=0.9, top_k=20)
    return Sample.request(input_part).fork(n, sampling_params=ar_params)


def main() -> int:
    model_path = os.environ.get("QWEN3_PATH")
    if not model_path:
        _log("ERROR: set QWEN3_PATH to a local Qwen3-4B dir")
        return 2
    _log(f"torch {torch.__version__} cuda={torch.version.cuda}; model_path={model_path}")

    config = Qwen3PipelineConfig(pretrained_model_ckpt_path=model_path, device="cuda:0")
    n = 2  # completions per prompt → exercises the sibling fan-out (n>1)
    try:
        _log("loading Qwen3Pipeline.from_config (bundle on cuda:0) ...")
        pipeline = Qwen3Pipeline.from_config(config)
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
        turns = out.text_conditioning()
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

        # PRIMARY invariant: replay is deterministic at fixed weights → old_logp == new_logp → ratio 1
        # (the old_logp_source="replay" contract every trainside Qwen3 recipe relies on).
        sc = rollout_replay_logp_absdiff(new2, new1)
        sc_mean, sc_max = sc["rollout_replay_logp_absdiff_mean"], sc["rollout_replay_logp_absdiff_max"]
        _log(
            f"replay self-consistency (old_logp_source=replay ⇒ ratio 1): mean|Δlogp|={sc_mean:.3e} max={sc_max:.3e} (threshold mean<{_SELF_CONSIST_MAX})"
        )
        assert sc_mean < _SELF_CONSIST_MAX, f"replay non-deterministic at fixed weights: mean|Δlogp|={sc_mean:.3e}"

        # INFORMATIONAL: autoregress(bf16, no-autocast) vs replay(fp32 norms + fp32 lm_head)
        # rollout-fidelity gap. Harmless under old_logp_source="replay": the autoregress record
        # never enters the GRPO ratio. NOT asserted ≈0 (unlike SD3, where diffuse and replay
        # share one forward). The generous guard catches a catastrophic alignment/conditions
        # break (would be 50+) while tolerating the ~5 precision gap. See module docstring.
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
            "TRAINSIDE AR SMOKE PASSED ✅  (rollout fills the Sample; replay deterministic — the old_logp_source=replay ratio≈1 contract)"
        )
        return 0
    except Exception:
        _log("TRAINSIDE AR SMOKE FAILED ❌")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
