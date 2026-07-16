#!/usr/bin/env python3
"""GPU smoke: drive the env-driven ``AgentLoop`` over the REAL SGLang AR engine (Qwen3, LIN-492).

Boots the real ``SGLangRolloutEngine`` (model_family "text", Qwen3-4B) and runs an ``AgentLoop`` over
it, driven by a stub ``FixedTurnsEnv`` — proving the env-driven loop drives a real rollout engine
end-to-end (not just the ``FakeEngine`` in ``rollout_loop_smoke.py``). PRIMARY (hard-asserted): one AR
turn. SECONDARY (informational): a 2-turn loop with an observation between turns — the real agentic
shape ``gen -> observation -> gen`` — since real-engine multi-turn is a later phase, not prototype scope.

    QWEN3_PATH=/root/unirl/models/local/Qwen3-4B-Base \
    CUDA_VISIBLE_DEVICES=0 .venv-sglang/bin/python scripts/rollout_loop_ar_smoke.py

Exits 0 on PASS (PRIMARY single-turn over the real engine), non-zero on failure.
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import Optional, Tuple

import torch

from unirl.rollout.engine.sglang.config import SGLangEngineConfig
from unirl.rollout.engine.sglang.engine import SGLangRolloutEngine
from unirl.rollout.loop import AgentLoop
from unirl.types.primitives import Texts
from unirl.types.sample import Part, Primitive, Sample
from unirl.types.sample_id import parent_id
from unirl.types.sampling import ARSamplingParams
from unirl.types.segments.text import TextSegment


def _log(msg: str) -> None:
    print(f"[loop-ar-smoke] {msg}", flush=True)


class FixedTurnsEnv:
    """Stub environment: drives exactly ``turns`` generations, then ``done``. Stands in for a real
    Environment so the loop runs a fixed number of turns over the real engine."""

    def __init__(self, turns: int) -> None:
        self._remaining = turns

    def reset(self, request: Sample) -> Sample:
        return request

    def step(self, sample: Sample) -> Tuple[Optional[Primitive], bool, dict]:
        self._remaining -= 1
        if self._remaining <= 0:
            return None, True, {}
        ids = sample.parts[-1].sample_ids
        return Texts(texts=[f"<obs>{sid}</obs>" for sid in ids]), False, {}


def build_request(n: int) -> Tuple[Sample, ARSamplingParams]:
    prompts = ["The capital of France is", "Two plus two equals"]
    input_part = Part.input(
        [f"p{i}" for i in range(len(prompts))], primitives={"text": Texts(texts=prompts)}, control={}
    )
    ar_params = ARSamplingParams(samples_per_prompt=n, temperature=0.7, max_new_tokens=48, top_p=0.9, top_k=20)
    return Sample.request(input_part), ar_params


def main() -> int:
    model_path = os.environ.get("QWEN3_PATH")
    if not model_path:
        _log("ERROR: set QWEN3_PATH to a local Qwen3-4B dir")
        return 2
    _log(f"torch {torch.__version__} cuda={torch.version.cuda}; model_path={model_path}")

    config = SGLangEngineConfig(
        pretrained_model_ckpt_path=model_path,
        backend="native",  # in-process sglang Engine (no separate server)
        tp_size=1,
        max_new_tokens=48,
        temperature=0.7,
        top_p=0.9,
        concurrency=8,
    )
    n = 2  # completions per prompt → exercises sibling fan-out (n>1)
    engine = None
    try:
        _log("constructing SGLangRolloutEngine (boots sglang + loads Qwen3) ...")
        engine = SGLangRolloutEngine(config, rank=0)

        # ---- PRIMARY: env-driven single-turn AgentLoop over the real engine (hard-asserted) ----
        request, ar_params = build_request(n)
        input_ids = set(request.parts[0].sample_ids)
        loop = AgentLoop(environment=FixedTurnsEnv(1), sampling_params=ar_params, max_turns=4)
        _log("running env-driven AgentLoop (1 turn) over the real engine ...")
        out = loop.run(engine, request)

        assert len(out.parts) == 2, f"expected [input, ar_gen]; got {len(out.parts)} parts"
        gen = out.parts[-1]
        assert len(gen.sample_ids) == 2 * n, f"expected {2 * n} samples; got {len(gen.sample_ids)}"
        assert all(parent_id(s) in input_ids for s in gen.sample_ids), "gen ids must be children of the prompts"
        assert isinstance(gen.segment, TextSegment), f"segment must be TextSegment; got {type(gen.segment)}"
        assert isinstance(gen.primitives.get("text"), Texts) and len(gen.primitives["text"].texts) == 2 * n, (
            "decoded Texts missing/wrong"
        )
        assert len(out.gen_parts()) == 1 and out.gen_parts()[0] is gen, "gen_parts must be exactly the AR gen Part"
        _log(f"PRIMARY PASS: env-driven single-turn loop produced {2 * n} Qwen completions via the real engine")
        for i, t in enumerate(gen.primitives["text"].texts):
            _log(f"  sample[{i}] id={gen.sample_ids[i]} text={t[:80]!r}")

        # ---- SECONDARY (informational): 2-turn loop, gen -> observation -> gen (real agentic shape) ----
        _log("probing 2-turn env-driven loop (gen -> observation -> gen) — informational ...")
        try:
            request2, ar2 = build_request(n)
            loop2 = AgentLoop(environment=FixedTurnsEnv(2), sampling_params=ar2, max_turns=4)
            out2 = loop2.run(engine, request2)
            ok2 = (
                len(out2.parts) == 4 and len(out2.gen_parts()) == 2 and isinstance(out2.parts[-1].segment, TextSegment)
            )
            _log(
                f"SECONDARY: 2-turn loop -> {len(out2.parts)} parts (expect [input, gen, obs, gen]), "
                f"gen_parts={len(out2.gen_parts())}, frontier seg={type(out2.parts[-1].segment).__name__} "
                f"=> {'PASS ✅' if ok2 else 'unexpected shape'}"
            )
            if not ok2:
                _log("SECONDARY: real-engine multi-turn-with-observation is a follow-up phase (not prototype scope)")
        except Exception as e2:  # noqa: BLE001
            _log(
                f"SECONDARY: multi-turn raised ({type(e2).__name__}: {e2}) — expected; "
                f"real-engine multi-turn-with-observation is a later phase, not in this prototype's scope"
            )

        _log("AGENT-LOOP AR SMOKE PASSED ✅  (env-driven AgentLoop drives the real Qwen SGLang engine)")
        return 0
    except Exception:
        _log("AGENT-LOOP AR SMOKE FAILED ❌")
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
