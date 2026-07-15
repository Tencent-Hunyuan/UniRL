"""Tests for ToolEnvironment.step re-entrancy under threads (LIN-522).

``step`` is the tool boundary the agentic drain calls from one thread per
trajectory. It must be re-entrant — one env instance serves many concurrent
trajectories, so the turn count must be derived per-sample, not held on the
instance — and a blocking tool must only block ITS trajectory's thread, never
serialize siblings against each other.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict

import pytest

pytest.importorskip("torch")  # the unirl types import torch at module load

from unirl.rollout.loop import ToolEnvironment  # noqa: E402
from unirl.rollout.loop.tools.calculator import CalculatorTool  # noqa: E402
from unirl.rollout.loop.tools.tool import Tool  # noqa: E402
from unirl.types.primitives import Texts  # noqa: E402
from unirl.types.sample import Part, Sample  # noqa: E402
from unirl.types.sampling import ARSamplingParams  # noqa: E402

TOOLCALL = '<tool_call>{"name": "calculator", "arguments": {"expression": "1234 * 5678"}}</tool_call>'
ANSWER = "7006652"


def _sample_at_turn(root: str, n_turns: int, body: str = TOOLCALL) -> Sample:
    """A trajectory with ``n_turns`` filled gen Parts (frontier carries ``body``)."""
    s = Sample.request(Part.input([root], primitives={"text": Texts(texts=[f"prompt-{root}"])}))
    for i in range(n_turns):
        s = s.fork(1, sampling_params=ARSamplingParams(samples_per_prompt=1))
        s = s.with_filled_frontier(primitives={"text": Texts(texts=[body])})
        if i < n_turns - 1:
            s = s.observe(Texts(texts=[f"obs{i}"]))
    return s


def test_step_is_reentrant_across_concurrent_trajectory_threads():
    """One env instance, two trajectories at DIFFERENT depths stepped from two
    threads: each derives its own turn (no shared-state clobbering). A stateful
    ``self._turn`` would race here."""
    env = ToolEnvironment([CalculatorTool()])
    a = _sample_at_turn("p0", 1)  # turn 1
    b = _sample_at_turn("p1", 2)  # turn 2

    results: dict = {}
    ta = threading.Thread(target=lambda: results.__setitem__("a", env.step(a)))
    tb = threading.Thread(target=lambda: results.__setitem__("b", env.step(b)))
    ta.start()
    tb.start()
    ta.join(timeout=5)
    tb.join(timeout=5)

    obs_a, done_a, info_a = results["a"]
    obs_b, done_b, info_b = results["b"]
    assert info_a["turn"] == 1
    assert info_b["turn"] == 2  # not clobbered by a's concurrent step
    assert obs_a.texts == obs_b.texts == [ANSWER]
    assert done_a is False and done_b is False


class _SlowTool(Tool):
    """A tool whose ``execute`` blocks (synchronously) for a beat — stands in for a
    browser/sandbox/search tool."""

    name = "slow"

    def json_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {"name": self.name, "parameters": {"type": "object", "properties": {}}},
        }

    def execute(self, arguments: Dict[str, Any]) -> str:
        del arguments
        time.sleep(0.2)
        return "slow-done"


def test_slow_tool_blocks_only_its_own_thread():
    """While a blocking tool runs on one trajectory's thread, a sibling
    trajectory's step on another thread completes first — the env never
    serializes siblings behind a slow tool."""
    env = ToolEnvironment([_SlowTool(), CalculatorTool()])
    slow_call = '<tool_call>{"name": "slow", "arguments": {}}</tool_call>'
    slow_sample = _sample_at_turn("p0", 1, body=slow_call)
    fast_sample = _sample_at_turn("p1", 1)  # calculator: effectively instant

    order = []
    lock = threading.Lock()

    def run(sample: Sample, tag: str, out: dict) -> None:
        out[tag] = env.step(sample)
        with lock:
            order.append(tag)

    results: dict = {}
    slow = threading.Thread(target=run, args=(slow_sample, "slow", results))
    fast = threading.Thread(target=run, args=(fast_sample, "fast", results))
    slow.start()
    time.sleep(0.02)  # the slow tool is now blocking its thread
    fast.start()
    fast.join(timeout=5)
    with lock:
        first_done = list(order)
    slow.join(timeout=5)

    assert first_done == ["fast"]  # the sibling finished while the slow tool blocked
    assert order == ["fast", "slow"]
    assert results["slow"][0].texts == ["slow-done"]
    assert results["fast"][0].texts == [ANSWER]
    assert results["slow"][2]["turn"] == 1 and results["fast"][2]["turn"] == 1
