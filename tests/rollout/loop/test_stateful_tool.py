"""CPU tests for StatefulTool + guaranteed session teardown (LIN-533).

A :class:`~unirl.rollout.loop.tools.tool.StatefulTool` holds per-trajectory state behind a session
id minted at ``reset`` and carried in the root Sample's *control* bag. These tests drive the real
:class:`~unirl.rollout.loop.tool_environment.ToolEnvironment` protocol with a trivial in-file
counter tool:

- ``reset`` stamps ``control["tool_sessions"]`` and fires ``session_start`` once;
- ``step`` dispatches ``execute_session`` with state that persists across turns;
- ``close`` ends the session exactly once — on success AND on a forced-exception (``finally``)
  path — and never raises, even when the tool's ``session_end`` throws.

The stateless :class:`~unirl.rollout.loop.tools.calculator.CalculatorTool` path stays byte-identical.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

pytest.importorskip("torch")  # the unirl types import torch at module load

from unirl.rollout.loop import ToolEnvironment  # noqa: E402
from unirl.rollout.loop.tools.calculator import CalculatorTool  # noqa: E402
from unirl.rollout.loop.tools.tool import StatefulTool  # noqa: E402
from unirl.types.primitives import Texts  # noqa: E402
from unirl.types.sample import Part, Sample  # noqa: E402
from unirl.types.sampling import ARSamplingParams  # noqa: E402

_SP = ARSamplingParams(samples_per_prompt=1, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=8)
_CALL = '<tool_call>{"name": "counter", "arguments": {}}</tool_call>'


class _CounterTool(StatefulTool):
    """Trivial stateful tool: each ``execute_session`` increments a per-session counter and returns
    the new value, so a returned value ``> 1`` proves state persisted across turns. Records the
    lifecycle (``started``/``ended``) for assertions; ``session_end`` is idempotent (a live session
    ends at most once). ``end_raises`` makes ``session_end`` throw, to test that ``close`` swallows.
    """

    name = "counter"

    def __init__(self, *, end_raises: bool = False) -> None:
        self.started: List[str] = []
        self.ended: List[str] = []
        self._counts: Dict[str, int] = {}
        self._end_raises = end_raises

    def json_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {"name": self.name, "parameters": {"type": "object", "properties": {}}},
        }

    def session_start(self, session_id: str, context: Dict[str, Any]) -> None:
        self.started.append(session_id)
        self._counts[session_id] = 0

    def execute_session(self, session_id: str, arguments: Dict[str, Any]) -> str:
        self._counts[session_id] += 1
        return str(self._counts[session_id])

    def session_end(self, session_id: str) -> None:
        if self._end_raises:
            raise RuntimeError("boom in session_end")
        if session_id in self._counts:  # idempotent: end a live session at most once
            self.ended.append(session_id)
            del self._counts[session_id]


def _request(sid: str) -> Sample:
    return Sample.request(Part.input([sid], primitives={"text": Texts(texts=["prompt"])}))


def _turn(sample: Sample, body: str = _CALL) -> Sample:
    return sample.fork(1, sampling_params=_SP).with_filled_frontier(primitives={"text": Texts(texts=[body])})


def test_reset_mints_and_starts_session():
    counter = _CounterTool()
    base = ToolEnvironment([counter]).reset(_request("r0"))
    sessions = base.parts[0].control["tool_sessions"]
    assert sessions["counter"]  # a session id was stamped into the control bag
    assert counter.started == [sessions["counter"]]  # session_start fired once with that id


def test_cross_turn_state_via_step():
    """The same session id (carried on parts[0].control across fork/observe) reaches every step,
    so the tool's per-session state accumulates — the capability stateless tools cannot express."""
    counter = _CounterTool()
    env = ToolEnvironment([counter])
    base = env.reset(_request("r0"))

    s1 = _turn(base)
    obs1, done1, _ = env.step(s1)
    assert done1 is False and isinstance(obs1, Texts) and obs1.texts == ["1"]

    s2 = _turn(s1.observe(obs1))
    obs2, done2, _ = env.step(s2)
    assert done2 is False and obs2.texts == ["2"]  # state persisted across turns


def test_teardown_fires_once_on_success():
    counter = _CounterTool()
    env = ToolEnvironment([counter])
    base = env.reset(_request("r0"))
    sid = base.parts[0].control["tool_sessions"]["counter"]

    env.close(base)
    assert counter.ended == [sid]  # session_end fired exactly once


def test_teardown_fires_on_forced_exception():
    """The engine calls close from a ``finally`` — teardown must fire even when the turn loop
    raises. Mirrors AgenticRolloutEngine._run_one's finally-hook."""
    counter = _CounterTool()
    env = ToolEnvironment([counter])
    base = env.reset(_request("r0"))
    sid = base.parts[0].control["tool_sessions"]["counter"]

    def _crash_then_close():
        try:
            raise RuntimeError("trajectory blew up mid-loop")
        finally:
            env.close(base)

    with pytest.raises(RuntimeError, match="blew up"):
        _crash_then_close()
    assert counter.ended == [sid]  # session_end still fired despite the exception


def test_close_is_idempotent():
    counter = _CounterTool()
    env = ToolEnvironment([counter])
    base = env.reset(_request("r0"))
    sid = base.parts[0].control["tool_sessions"]["counter"]

    env.close(base)
    env.close(base)  # second call is a no-op (session already ended)
    assert counter.ended == [sid]


def test_close_swallows_session_end_errors():
    """A raising session_end must never propagate out of close — the engine's drain relies on
    _run_one never raising."""
    env = ToolEnvironment([_CounterTool(end_raises=True)])
    base = env.reset(_request("r0"))
    env.close(base)  # must not raise


def test_stateless_tool_is_untouched():
    """Zero regression: with no StatefulTool, reset returns the request object unchanged (no
    control stamping) and close is a no-op."""
    env = ToolEnvironment([CalculatorTool()])
    req = _request("r0")
    assert env.reset(req) is req  # same object back — no session plumbing
    env.close(req)  # no-op, must not raise


def test_mixed_stateful_and_stateless():
    counter = _CounterTool()
    env = ToolEnvironment([CalculatorTool(), counter])
    base = env.reset(_request("r0"))
    assert set(base.parts[0].control["tool_sessions"]) == {"counter"}  # only stateful tools get a session

    calc = '<tool_call>{"name": "calculator", "arguments": {"expression": "6 * 7"}}</tool_call>'
    obs, done, _ = env.step(_turn(base, body=calc))
    assert done is False and obs.texts == ["42"]  # stateless dispatch still works alongside a session bag
