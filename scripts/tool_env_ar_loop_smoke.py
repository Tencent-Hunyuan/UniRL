#!/usr/bin/env python3
"""GPU CLOSED-LOOP smoke: a real Qwen3-4B-Instruct calls a tool, SEES the result, and answers (LIN-492).

The first fully-closed agentic tool loop over the real engine — the payoff of rebasing onto
LIN-444/main (LIN-503 multi-turn conditioning) plus `Sample.observe(role="tool")`. Boots SGLang
with the calculator advertised via `chat_template_kwargs`, then runs `AgentLoop` + `ToolEnvironment`
end to end:

  turn 1: the model emits  <tool_call>{calculator: 987654321 * 123456789}</tool_call>
  env:    parse -> CalculatorTool -> observation (role 'tool') carrying the exact product
  turn 2: LIN-503 renders [user, assistant(tool_call), tool(<product>)] -> the model SEES the
          product and answers (no tool call) -> ToolEnvironment.step returns done.

The expression is deliberately a 9-digit × 9-digit product the model cannot compute mentally, so a
final answer containing the EXACT 18-digit result proves the model consumed the tool output (not its
own arithmetic) — i.e. the multi-turn conditioning truly closed the loop.

    QWEN3_INSTRUCT_PATH=/root/unirl/models/local/Qwen3-4B-Instruct \
    CUDA_VISIBLE_DEVICES=0 .venv-sglang/bin/python scripts/tool_env_ar_loop_smoke.py
"""

from __future__ import annotations

import os
import sys
import traceback

import torch

from unirl.rollout.engine.sglang.config import SGLangEngineConfig
from unirl.rollout.engine.sglang.engine import SGLangRolloutEngine
from unirl.rollout.loop import AgentLoop, CalculatorTool, ToolEnvironment, parse_tool_call
from unirl.types.primitives import Texts
from unirl.types.sample import Part, Sample
from unirl.types.sampling import ARSamplingParams

EXPR = "987654321 * 123456789"
PROMPT = f"What is {EXPR}? Use the calculator tool to compute it, then state the final answer."


def _log(msg: str) -> None:
    print(f"[tool-loop-ar-smoke] {msg}", flush=True)


def main() -> int:
    model_path = os.environ.get("QWEN3_INSTRUCT_PATH")
    if not model_path:
        _log("ERROR: set QWEN3_INSTRUCT_PATH to a local Qwen3-4B-Instruct dir")
        return 2
    _log(f"torch {torch.__version__} cuda={torch.version.cuda}; model={model_path}")

    # Ground truth from the tool itself (not hardcoded): the exact product the model must echo.
    expected = CalculatorTool().execute({"expression": EXPR})
    _log(f"expression={EXPR!r}  expected tool result={expected!r}")

    env = ToolEnvironment([CalculatorTool()], max_turns=4)
    config = SGLangEngineConfig(
        pretrained_model_ckpt_path=model_path,
        backend="native",
        tp_size=1,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        concurrency=8,
        chat_template_kwargs={"tools": env.tool_schemas()},
    )

    engine = None
    try:
        _log("constructing SGLangRolloutEngine (boots sglang + loads Qwen3-4B-Instruct) ...")
        engine = SGLangRolloutEngine(config, rank=0)

        ar = ARSamplingParams(samples_per_prompt=1, temperature=0.7, max_new_tokens=512, top_p=0.9, top_k=20)
        loop = AgentLoop(environment=env, sampling_params=ar, max_turns=4)
        request = Sample.request(Part.input(["p0"], primitives={"text": Texts(texts=[PROMPT])}))

        _log("running the closed agentic loop (tool call -> observe -> answer) ...")
        out = loop.run(engine, request)

        # Show the rendered conversation (role-tagged turns) — the closed loop, visualized.
        try:
            _log("trajectory turns (role: content):")
            for t in out.turns():
                content = t.content.texts[0] if isinstance(t.content, Texts) else str(t.content)
                _log(f"  [{t.role:<9}] {content[:200]!r}")
        except Exception as e:  # noqa: BLE001 — diagnostic only
            _log(f"(turns() diagnostic skipped: {type(e).__name__}: {e})")

        gens = out.gen_parts()
        _log(f"gen turns: {len(gens)}")

        # ---- assert the closed loop ----
        assert len(gens) >= 2, f"expected >=2 gen turns (tool call + answer); got {len(gens)}"

        first_call = parse_tool_call(gens[0].primitives["text"].texts[0])
        assert first_call is not None and first_call["name"] == "calculator", "turn 1 must call the calculator"
        _log(f"turn 1 tool call: {first_call}")

        tool_obs = [p for p in out.parts if p.resolved_role() == "tool"]
        assert tool_obs, "no observation Part rendered as role 'tool'"
        assert expected in tool_obs[0].primitives["text"].texts[0], f"tool observation must carry {expected}"
        _log(f"tool observation (role 'tool'): {tool_obs[0].primitives['text'].texts[0]!r}")

        final = gens[-1].primitives["text"].texts[0]
        _log(f"final answer: {final[:300]!r}")
        assert parse_tool_call(final) is None, "the final turn must be an answer, not another tool call"
        assert expected in final.replace(",", ""), (
            f"the final answer must contain the EXACT tool result {expected} (proving the model saw it); "
            f"got {final[:200]!r}"
        )

        _log(f"CLOSED-LOOP SMOKE PASSED ✅  (model called calculator, saw {expected}, and answered with it)")
        return 0
    except Exception:
        _log("CLOSED-LOOP SMOKE FAILED ❌")
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
