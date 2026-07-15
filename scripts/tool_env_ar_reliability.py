#!/usr/bin/env python3
"""GPU reliability suite for the agent loop (LIN-492).

Runs ``AgentLoop`` + ``ToolEnvironment`` over the REAL Qwen3-4B-Instruct across a diverse battery
of arithmetic tasks, each with **n>1 GRPO fan-out** (the real training batch shape), and reports
aggregate behaviour:
  - tool-call rate   — did the loop parse + drive a calculator call (the LOOP's job),
  - correct rate     — did the final answer contain the exact true result (loop + model copying),
  - turn distribution — did the loop terminate cleanly (mostly 2 turns: call -> answer),
  - heterogeneity    — n>1 siblings that disagreed on turn-1 tool use (the known n>1 edge).

Beyond the single closed-loop smoke: it characterizes reliability across inputs and stochastic
samples. Ground truth for each task is computed by the calculator itself (not hardcoded).

    QWEN3_INSTRUCT_PATH=/root/unirl/models/local/Qwen3-4B-Instruct \
    CUDA_VISIBLE_DEVICES=0 N_SAMPLES=6 .venv-sglang/bin/python scripts/tool_env_ar_reliability.py
"""

from __future__ import annotations

import os
import sys
import traceback
from collections import Counter

import torch

from unirl.rollout.engine.sglang.config import SGLangEngineConfig
from unirl.rollout.engine.sglang.engine import SGLangRolloutEngine
from unirl.rollout.loop import AgentLoop, CalculatorTool, ToolEnvironment, parse_tool_call
from unirl.types.primitives import Texts
from unirl.types.sample import Part, Sample
from unirl.types.sampling import ARSamplingParams

# Single-expression tasks (integer results → clean exact-match). Mix of magnitudes, ops, parens,
# powers, negatives — several too large to compute mentally (must come from the tool).
EXPRS = [
    "12 * 13",
    "987654321 * 123456789",
    "(123 + 456) * 789",
    "1000000 / 8",
    "888888 - 999999",
    "2 ** 20",
    "456789 * 987654",
    "(999 - 333) * (111 + 222)",
    "314159 + 271828",
    "98765 * 43210",
]
# Chained tasks — may induce 3+ turns (the model calls the tool more than once).
CHAINED = [
    (
        "First multiply 123 by 456, then add 7890 to that result. Use the calculator for each step, "
        "then give the final number.",
        "123 * 456 + 7890",
    ),
    (
        "Compute 5000 minus 1234, then multiply the result by 6, using the calculator. State the final number.",
        "(5000 - 1234) * 6",
    ),
]


def _log(msg: str) -> None:
    print(f"[reliability] {msg}", flush=True)


def _request(prompt: str) -> Sample:
    return Sample.request(Part.input(["p0"], primitives={"text": Texts(texts=[prompt])}))


def _norm(s: str) -> str:
    return s.replace(",", "").replace(" ", "")


def evaluate(out, expected):
    """One record per sibling: did it call the tool (turn 1), is its final answer correct, #turns."""
    gens = out.gen_parts()
    first_texts = gens[0].primitives["text"].texts
    final_texts = gens[-1].primitives["text"].texts
    n = len(final_texts)
    recs = []
    for i in range(n):
        first = first_texts[i] if i < len(first_texts) else ""
        final = final_texts[i] if i < len(final_texts) else ""
        call = parse_tool_call(first)
        called = call is not None and call.get("name") == "calculator"
        recs.append({"called": called, "correct": _norm(expected) in _norm(final), "turns": len(gens), "final": final})
    return recs


def main() -> int:
    model_path = os.environ.get("QWEN3_INSTRUCT_PATH")
    if not model_path:
        _log("ERROR: set QWEN3_INSTRUCT_PATH to a local Qwen3-4B-Instruct dir")
        return 2
    n = int(os.environ.get("N_SAMPLES", "6"))
    _log(f"torch {torch.__version__} cuda={torch.version.cuda}; model={model_path}; n={n}")

    calc = CalculatorTool()
    suite = [
        (f"Compute {e} using the calculator tool, then state the final number.", e, calc.execute({"expression": e}))
        for e in EXPRS
    ]
    suite += [(prompt, e, calc.execute({"expression": e})) for prompt, e in CHAINED]

    schemas = ToolEnvironment([calc]).tool_schemas()
    config = SGLangEngineConfig(
        pretrained_model_ckpt_path=model_path,
        backend="native",
        tp_size=1,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        concurrency=8,
        chat_template_kwargs={"tools": schemas},
        # Leave headroom on the 95 GB H20: the default mem_fraction_static=0.88 reserves ~92 GB and
        # OOMs on the growing multi-turn prefill's transient logprob allocation (chained tasks).
        engine_kwargs={"mem_fraction_static": 0.70},
    )

    engine = None
    try:
        _log("constructing SGLangRolloutEngine (boots sglang + loads Qwen3-4B-Instruct) ...")
        engine = SGLangRolloutEngine(config, rank=0)
        ar = ARSamplingParams(samples_per_prompt=n, temperature=0.7, max_new_tokens=512, top_p=0.9, top_k=20)

        all_recs = []
        turn_hist = Counter()
        het_prompts = 0
        _log(f"running {len(suite)} prompts x n={n} = {len(suite) * n} trajectories ...")
        for idx, (prompt, expr, expected) in enumerate(suite):
            loop = AgentLoop(environment=ToolEnvironment([calc], max_turns=6), sampling_params=ar, max_turns=6)
            out = loop.run(engine, _request(prompt))
            recs = evaluate(out, expected)
            all_recs.extend(recs)
            for r in recs:
                turn_hist[r["turns"]] += 1
            called = sum(r["called"] for r in recs)
            correct = sum(r["correct"] for r in recs)
            het = 0 < called < len(recs)
            het_prompts += het
            _log(
                f"  [{idx + 1:2}/{len(suite)}] {expr:<26} (={str(expected)[:18]}): "
                f"tool {called}/{len(recs)}  correct {correct}/{len(recs)}  turns~{recs[0]['turns']}"
                f"{'  <-- heterogeneous turn-1' if het else ''}"
            )

        total = len(all_recs)
        called = sum(r["called"] for r in all_recs)
        correct = sum(r["correct"] for r in all_recs)
        _log("=" * 72)
        _log(f"TRAJECTORIES: {total}  ({len(suite)} prompts x n={n})")
        _log(f"  tool-call rate (loop drove a calculator call): {called}/{total} = {100 * called / total:.1f}%")
        _log(f"  end-to-end correct (answer has exact result):  {correct}/{total} = {100 * correct / total:.1f}%")
        _log(f"  turn distribution: {dict(sorted(turn_hist.items()))}")
        _log(f"  prompts with heterogeneous turn-1 tool use:    {het_prompts}/{len(suite)}")
        fails = [r for r in all_recs if not r["correct"]]
        if fails:
            _log(f"  {len(fails)} incorrect-final — first few:")
            for r in fails[:6]:
                _log(f"     called={r['called']} turns={r['turns']} final={r['final'][:110]!r}")
        loop_ok = called / total >= 0.90
        e2e_ok = correct / total >= 0.85
        verdict = "RELIABLE ✅" if (loop_ok and e2e_ok) else "REVIEW ⚠️"
        _log(
            f"VERDICT: {verdict}  (loop bar >=90% tool-call: {'ok' if loop_ok else 'MISS'}; "
            f"e2e bar >=85% correct: {'ok' if e2e_ok else 'MISS'})"
        )
        return 0
    except Exception:
        _log("RELIABILITY SUITE CRASHED ❌")
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
