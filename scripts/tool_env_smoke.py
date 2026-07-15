#!/usr/bin/env python3
"""CPU oracle for the ToolEnvironment (LIN-492).

Drives the real ``AgentLoop`` + ``ToolEnvironment`` + ``CalculatorTool`` with a model-free
``ScriptedEngine`` that emits real ``<tool_call>`` payloads — no GPU, no model. Mirrors
``scripts/rollout_loop_smoke.py`` (``check_*`` contracts, ``main() -> int``, non-zero exit on first
failure). The load-bearing checks: tool-call-driven termination (the loop runs exactly as long as
the model emits a tool call) and that the calculator's result rides back as the next observation.

    python scripts/tool_env_smoke.py
"""

from __future__ import annotations

import sys
from typing import Callable, List, Tuple

from unirl.rollout.loop import AgentLoop, CalculatorTool, ToolEnvironment, parse_tool_call
from unirl.types.primitives import Texts
from unirl.types.sample import Part, Sample
from unirl.types.sampling import ARSamplingParams

# A real calculator tool call (1234 * 5678 = 7006652), a final answer (no call), and an
# unknown-tool call — the three frontier "model outputs" the scripted engine replays.
TOOLCALL = '<tool_call>{"name": "calculator", "arguments": {"expression": "1234 * 5678"}}</tool_call>'
FINAL = "The product is 7006652. <answer>7006652</answer>"
UNKNOWN = '<tool_call>{"name": "search", "arguments": {"query": "anything"}}</tool_call>'


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


class ScriptedEngine:
    """Model-free engine: fills the frontier shell with a scripted body per turn (clamped to the
    last). Lets the ToolEnvironment drive the loop off the body's *content* (tool call vs final)."""

    def __init__(self, bodies: List[str]) -> None:
        self.bodies = bodies
        self.calls = 0

    def generate(self, sample: Sample) -> Sample:
        shell = sample.parts[-1]
        body = self.bodies[min(self.calls, len(self.bodies) - 1)]
        self.calls += 1
        filled = shell.fill(primitives={"text": Texts(texts=[body for _ in shell.sample_ids])})
        return sample.with_parts([*sample.parts[:-1], filled])


def _request(prompts: List[str]) -> Sample:
    ids = [f"p{i}" for i in range(len(prompts))]
    return Sample.request(Part.input(ids, primitives={"text": Texts(texts=prompts)}))


def _loop(env: ToolEnvironment) -> AgentLoop:
    return AgentLoop(environment=env, sampling_params=ARSamplingParams(samples_per_prompt=1), max_turns=16)


def check_tool_call_driven_termination() -> None:
    """THE load-bearing check: 1 tool call then a final answer -> exactly 2 gen turns, with the
    calculator result (7006652) carried back as the mask-0 observation between them."""
    out = _loop(ToolEnvironment([CalculatorTool()])).run(ScriptedEngine([TOOLCALL, FINAL]), _request(["1234*5678?"]))
    _check(len(out.parts) == 4, f"[input, gen, obs, gen]; got {len(out.parts)}")
    _check(len(out.gen_parts()) == 2, f"1 call + 1 final -> 2 gen turns; got {len(out.gen_parts())}")
    obs = out.parts[2]
    _check(obs.sampling_params is None, "the middle Part is a mask-0 observation (not trained)")
    _check(
        "7006652" in obs.primitives["text"].texts[0],
        f"observation carries the tool result; got {obs.primitives['text'].texts[0]!r}",
    )
    _check(
        parse_tool_call(out.parts[-1].primitives["text"].texts[0]) is None,
        "final gen has no tool call (loop stopped there)",
    )


def check_multi_tool_turns() -> None:
    """3 tool calls then a final -> 4 gen turns and 3 observation Parts, each carrying the result."""
    out = _loop(ToolEnvironment([CalculatorTool()])).run(
        ScriptedEngine([TOOLCALL, TOOLCALL, TOOLCALL, FINAL]), _request(["go"])
    )
    _check(len(out.gen_parts()) == 4, f"3 calls + final -> 4 gens; got {len(out.gen_parts())}")
    obs_parts = [p for p in out.parts if p.sampling_params is None and not p.is_root]
    _check(len(obs_parts) == 3, f"3 observation Parts; got {len(obs_parts)}")
    _check(
        all("7006652" in p.primitives["text"].texts[0] for p in obs_parts),
        "each observation carries the calculator result",
    )


def check_immediate_final() -> None:
    """A final answer with no tool call -> exactly one gen turn, no observation, done immediately."""
    out = _loop(ToolEnvironment([CalculatorTool()])).run(ScriptedEngine([FINAL]), _request(["hi"]))
    _check(len(out.parts) == 2, f"[input, gen]; got {len(out.parts)}")
    _check(len(out.gen_parts()) == 1, "one gen, no tool call -> immediate done")


def check_max_turns_cap() -> None:
    """A model that never stops calling tools is capped by the environment's ``max_turns``."""
    out = _loop(ToolEnvironment([CalculatorTool()], max_turns=3)).run(ScriptedEngine([TOOLCALL]), _request(["loop"]))
    _check(len(out.gen_parts()) == 3, f"env max_turns=3 caps the loop at 3 gens; got {len(out.gen_parts())}")


def check_unknown_tool_error_obs() -> None:
    """An unknown tool name surfaces as an ``Error: ...`` observation (model can recover); not a crash."""
    out = _loop(ToolEnvironment([CalculatorTool()])).run(ScriptedEngine([UNKNOWN, FINAL]), _request(["q"]))
    obs = out.parts[2]
    _check(obs.sampling_params is None, "observation Part follows the unknown-tool call")
    _check(
        "Error: unknown tool" in obs.primitives["text"].texts[0],
        f"unknown tool -> error obs; got {obs.primitives['text'].texts[0]!r}",
    )
    _check(len(out.gen_parts()) == 2, "loop continues past the error, then the final -> 2 gens")


def check_parse_tool_call() -> None:
    """``parse_tool_call`` handles valid / absent / malformed / args-as-string / stop-trimmed / last-wins."""
    valid = '<tool_call>{"name": "calculator", "arguments": {"expression": "2+2"}}</tool_call>'
    _check(parse_tool_call(valid) == {"name": "calculator", "arguments": {"expression": "2+2"}}, "valid parse")
    _check(parse_tool_call("a plain final answer, no call") is None, "no tool call -> None")
    _check(parse_tool_call("<tool_call>{not valid json}</tool_call>") is None, "malformed JSON -> None")
    as_string = '<tool_call>{"name": "calculator", "arguments": "{\\"expression\\": \\"3*3\\"}"}</tool_call>'
    _check(parse_tool_call(as_string) == {"name": "calculator", "arguments": {"expression": "3*3"}}, "args-as-string")
    trimmed = '<tool_call>{"name": "calculator", "arguments": {"expression": "7*6"}}'  # no </tool_call>
    _check(
        parse_tool_call(trimmed) == {"name": "calculator", "arguments": {"expression": "7*6"}}, "stop-trimmed recover"
    )
    two = valid + ' <tool_call>{"name": "calculator", "arguments": {"expression": "9*9"}}</tool_call>'
    _check(parse_tool_call(two)["arguments"]["expression"] == "9*9", "last tool call wins")


def check_calculator() -> None:
    """``CalculatorTool`` computes correctly and rejects anything outside the arithmetic whitelist."""
    calc = CalculatorTool()
    _check(calc.execute({"expression": "1234 * 5678"}) == "7006652", "1234*5678")
    _check(calc.execute({"expression": "2 + 2"}) == "4", "2+2")
    _check(calc.execute({"expression": "10 / 4"}) == "2.5", "10/4 -> 2.5")
    _check(calc.execute({"expression": "(3 + 4) * 2 - 5"}) == "9", "precedence + parens")
    _check(calc.execute({"expression": "-7 + 3"}) == "-4", "unary minus")
    for bad in ("__import__('os')", "x + 1", "len([1, 2])", "1 / 0", "1 < 2"):
        try:
            calc.execute({"expression": bad})
        except Exception:  # noqa: BLE001 — every one of these must raise
            continue
        raise AssertionError(f"calculator must reject {bad!r}")


def check_tool_schemas() -> None:
    """``tool_schemas()`` returns OpenAI function-tool dicts (for ``apply_chat_template(tools=...)``)."""
    schemas = ToolEnvironment([CalculatorTool()]).tool_schemas()
    _check(len(schemas) == 1, f"one tool schema; got {len(schemas)}")
    _check(schemas[0]["type"] == "function", "schema is an OpenAI function tool")
    fn = schemas[0]["function"]
    _check(fn["name"] == "calculator", "schema names the calculator")
    _check("expression" in fn["parameters"]["properties"], "calculator advertises 'expression'")
    _check(fn["parameters"]["required"] == ["expression"], "'expression' is required")


def check_observation_role_is_tool() -> None:
    """The tool result rides back as a 'tool'-role turn (LIN-503 role-aware rendering) so the chat
    template wraps it as <tool_response>; the generation is 'assistant' and the prompt is 'user'."""
    out = _loop(ToolEnvironment([CalculatorTool()])).run(ScriptedEngine([TOOLCALL, FINAL]), _request(["1234*5678?"]))
    _check(out.parts[0].resolved_role() == "user", f"prompt renders as 'user'; got {out.parts[0].resolved_role()!r}")
    _check(
        out.parts[1].resolved_role() == "assistant", f"gen renders as 'assistant'; got {out.parts[1].resolved_role()!r}"
    )
    _check(
        out.parts[2].resolved_role() == "tool", f"observation renders as 'tool'; got {out.parts[2].resolved_role()!r}"
    )


_CHECKS: Tuple[Callable[[], None], ...] = (
    check_tool_call_driven_termination,
    check_multi_tool_turns,
    check_immediate_final,
    check_max_turns_cap,
    check_unknown_tool_error_obs,
    check_parse_tool_call,
    check_calculator,
    check_tool_schemas,
    check_observation_role_is_tool,
)


def main() -> int:
    failures: List[str] = []
    for check in _CHECKS:
        try:
            check()
        except Exception as exc:  # noqa: BLE001 — the oracle reports, doesn't crash
            failures.append(f"{check.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"  ok  {check.__name__}")
    if failures:
        print("tool-env-smoke: FAILED", file=sys.stderr)
        for f in failures:
            print(f"  FAIL {f}", file=sys.stderr)
        return 1
    print(f"tool-env-smoke: {len(_CHECKS)} contracts hold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
