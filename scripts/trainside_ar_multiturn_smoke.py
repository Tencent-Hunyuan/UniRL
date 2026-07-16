#!/usr/bin/env python3
"""Trainside MULTI-TURN encode smoke (LIN-503 — gap C conjugate).

Builds a ``[user, assistant, tool, gen]`` request ``Sample`` by hand, runs the
in-process Qwen3 pipeline's ``generate``, and proves the engine now CONDITIONS ON
THE FULL TRAJECTORY: the stored ``conditions["prompt"]`` decodes to a conversation
containing the **assistant and tool** turns — not just the root user prompt (the
pre-fix behavior flattened to ``conditioning()[0]``). Also checks replay
self-consistency (ratio≈1) on the multi-turn conditions.

No inference server, no training loop, no reward. Run on a GPU pod, torch venv:

    QWEN3_PATH=/root/unirl/models/local/Qwen3-4B-Instruct \
    CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/trainside_ar_multiturn_smoke.py

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

_SELF_CONSIST_MAX = 1e-3

_USER = "What is 19 times 23?"
_ASSISTANT = "Let me calculate that for you."
_TOOL = "437"


def _log(msg: str) -> None:
    print(f"[trainside-mt] {msg}", flush=True)


def build_multiturn_sample() -> Sample:
    """A 3-turn trajectory ``[user, assistant, tool]`` + a frontier gen shell."""
    inp = Part.input(["p0"], primitives={"text": Texts(texts=[_USER])}, role="user", control={})
    asst = inp.fork(1, sampling_params=ARSamplingParams()).fill(primitives={"text": Texts(texts=[_ASSISTANT])})
    tool = asst.input_child({"text": Texts(texts=[_TOOL])}, role="tool")
    ar_params = ARSamplingParams(samples_per_prompt=1, temperature=0.7, max_new_tokens=48, top_p=0.9, top_k=20)
    gen = tool.fork(1, sampling_params=ar_params)
    return Sample(parts=[inp, asst, tool, gen])


def main() -> int:
    model_path = os.environ.get("QWEN3_PATH")
    if not model_path:
        _log("ERROR: set QWEN3_PATH to a local Qwen3 dir")
        return 2
    _log(f"torch {torch.__version__} cuda={torch.version.cuda}; model_path={model_path}")
    # Prove WHICH code is loaded (worktree vs editable install).
    import unirl.models.qwen3.chat_template as _ct

    _log(f"qwen3.chat_template from: {_ct.__file__}")

    config = Qwen3PipelineConfig(pretrained_model_ckpt_path=model_path, device="cuda:0")
    try:
        _log("loading Qwen3Pipeline.from_config (bundle on cuda:0) ...")
        pipeline = Qwen3Pipeline.from_config(config)
        engine = TrainsideRolloutEngine(pipeline=pipeline, stage_attrs=("ar",))

        sample = build_multiturn_sample()
        _log(f"request: {len(sample.parts)} parts [user, assistant, tool, gen]")
        _log(f"trajectory turns: {[t.role for t in sample.text_conditioning()]}")

        _log("calling engine.generate(sample) [rollout] ...")
        out = engine.generate(sample)
        gen = out.parts[-1]
        assert gen.conditions and "prompt" in gen.conditions, "no stored prompt conditions"

        # ---- THE multi-turn encode proof: decode the prompt the model conditioned on ----
        prompt_ids = gen.conditions["prompt"].input_ids[0]
        tok = pipeline.bundle.tokenizer
        prompt_text = tok.decode([int(t) for t in prompt_ids.tolist()], skip_special_tokens=False)
        _log("--- decoded conditions['prompt'] (what the model attended) ---")
        _log(prompt_text)
        _log("--------------------------------------------------------------")

        assert _USER in prompt_text, "USER turn missing from the encoded prompt"
        assert _ASSISTANT in prompt_text, (
            "ASSISTANT turn DROPPED — the encode flattened to the root prompt (gap C not applied trainside)"
        )
        assert _TOOL in prompt_text, (
            "TOOL turn DROPPED — the encode flattened to the root prompt (gap C not applied trainside)"
        )
        # Order sanity: user → assistant → tool in the rendered prompt.
        assert prompt_text.index(_USER) < prompt_text.index(_ASSISTANT) < prompt_text.index(_TOOL), (
            "turns out of lineage order in the encoded prompt"
        )
        _log("MULTI-TURN ENCODE PASS: prompt carries user → assistant → tool ✓")
        _log(f"completion: {gen.primitives['text'].texts[0]!r}")

        # ---- replay self-consistency on the multi-turn conditions (ratio≈1) ----
        temperature = float(gen.sampling_params.temperature)
        model = pipeline.ar.trainable_module()
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                conds = pipeline._conditions_for(out.text_conditioning(), out.parts[0].control)
                new1 = pipeline.ar.replay(conds, segment=gen.segment, temperature=temperature)
                new2 = pipeline.ar.replay(conds, segment=gen.segment, temperature=temperature)
        finally:
            model.train(was_training)
        assert torch.isfinite(new1).all(), "replay produced non-finite log-probs"
        sc = rollout_replay_logp_absdiff(new2, new1)
        sc_mean = sc["rollout_replay_logp_absdiff_mean"]
        _log(f"replay self-consistency (multi-turn conds): mean|Δlogp|={sc_mean:.3e} (threshold <{_SELF_CONSIST_MAX})")
        assert sc_mean < _SELF_CONSIST_MAX, f"replay non-deterministic at fixed weights: mean|Δlogp|={sc_mean:.3e}"

        _log("TRAINSIDE MULTI-TURN SMOKE PASSED ✅  (encode carries the full trajectory; replay deterministic)")
        return 0
    except Exception:
        _log("TRAINSIDE MULTI-TURN SMOKE FAILED ❌")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
