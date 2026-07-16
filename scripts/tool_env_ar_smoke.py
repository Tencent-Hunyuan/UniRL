#!/usr/bin/env python3
"""GPU single-turn smoke: a REAL Qwen3-4B-Instruct emits a real tool call the ToolEnvironment runs (LIN-492).

Boots the real ``SGLangRolloutEngine`` with the calculator tool advertised to the model via
``chat_template_kwargs={"tools": env.tool_schemas()}``, asks it an arithmetic question, and proves
the **first half** of an agentic turn end-to-end: the model emits a parseable
``<tool_call>{"name": "calculator", ...}</tool_call>``, ``parse_tool_call`` extracts it, and
``CalculatorTool`` computes the right answer — which ``ToolEnvironment.step`` hands back as the
observation. The **continuation** (the model *sees* the observation and answers) is deferred to the
separately-built multi-turn conditioning in the engine adapter (today it conditions only on the root
prompt), so this smoke asserts turn-0 only.

    QWEN3_INSTRUCT_PATH=/root/unirl/models/local/Qwen3-4B-Instruct \
    CUDA_VISIBLE_DEVICES=0 .venv-sglang/bin/python scripts/tool_env_ar_smoke.py

Exits 0 on PASS (model emitted a calculator call → 7006652), non-zero on failure.
"""

from __future__ import annotations

import os
import sys
import traceback

import torch

from unirl.rollout.engine.sglang.config import SGLangEngineConfig
from unirl.rollout.engine.sglang.engine import SGLangRolloutEngine
from unirl.rollout.loop import CalculatorTool, ToolEnvironment, parse_tool_call
from unirl.types.primitives import Texts
from unirl.types.sample import Part, Sample
from unirl.types.sampling import ARSamplingParams

PROMPT = "What is 1234 multiplied by 5678? Use the calculator tool to compute it, then give the answer."
EXPECTED = "7006652"


def _log(msg: str) -> None:
    print(f"[tool-env-ar-smoke] {msg}", flush=True)


def main() -> int:
    model_path = os.environ.get("QWEN3_INSTRUCT_PATH")
    if not model_path:
        _log("ERROR: set QWEN3_INSTRUCT_PATH to a local Qwen3-4B-Instruct dir")
        return 2
    _log(f"torch {torch.__version__} cuda={torch.version.cuda}; model_path={model_path}")

    env = ToolEnvironment([CalculatorTool()])
    config = SGLangEngineConfig(
        pretrained_model_ckpt_path=model_path,
        backend="native",  # in-process sglang Engine (no separate server)
        tp_size=1,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        concurrency=8,
        # Advertise the calculator to the model — the tokenizer's chat template renders it into the
        # prompt and Qwen3-Instruct emits <tool_call>{...}</tool_call>.
        chat_template_kwargs={"tools": env.tool_schemas()},
    )

    engine = None
    try:
        _log("constructing SGLangRolloutEngine (boots sglang + loads Qwen3-4B-Instruct) ...")
        engine = SGLangRolloutEngine(config, rank=0)

        # Stop right after the tool call so the model can't hallucinate its own tool response;
        # parse_tool_call's balanced-brace fallback recovers the stop-trimmed </tool_call>.
        request = Sample.request(
            Part.input(
                ["p0"],
                primitives={"text": Texts(texts=[PROMPT])},
                control={"ar": {"stop": ["</tool_call>"]}},
            )
        )
        ar = ARSamplingParams(samples_per_prompt=1, temperature=0.7, max_new_tokens=512, top_p=0.9, top_k=20)

        _log("generating turn 0 (the model should call the calculator) ...")
        out = engine.generate(request.fork(1, sampling_params=ar))
        text = out.parts[-1].primitives["text"].texts[0]
        _log(f"raw model output:\n{text!r}")

        # ---- assert the model emitted a parseable calculator tool call ----
        call = parse_tool_call(text)
        assert call is not None, "model did not emit a parseable <tool_call>"
        _log(f"parsed tool call: {call}")
        assert call["name"] == "calculator", f"expected a 'calculator' call; got {call['name']!r}"

        # ---- the ToolEnvironment runs the call and hands the result back as the observation ----
        observation, done, info = env.step(out)
        _log(f"env.step -> done={done}  info.turn={info['turn']}  result={info['results'][0]!r}")
        assert isinstance(observation, Texts), "expected a Texts observation"
        assert EXPECTED in observation.texts[0], f"observation must carry {EXPECTED}; got {observation.texts[0]!r}"
        assert not done, "a tool call should NOT end the episode (the model still needs to answer)"

        # And the calculator computes it directly from the parsed args, too.
        assert CalculatorTool().execute(call["arguments"]) == EXPECTED, "calculator result mismatch"

        _log(f"TOOL-ENV AR SMOKE PASSED ✅  (real Qwen3-4B-Instruct called calculator -> {EXPECTED})")
        _log("note: the continuation (model sees the observation and answers) needs the engine's")
        _log("      multi-turn conditioning (built separately) — out of scope for this turn-0 smoke.")
        return 0
    except Exception:
        _log("TOOL-ENV AR SMOKE FAILED ❌")
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
