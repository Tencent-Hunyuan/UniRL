#!/usr/bin/env python3
"""CPU oracle for the environment-driven AgentLoop (LIN-492).

Exercises the synchronous, env-driven ``AgentLoop`` on fabricated Samples with a model-free
``FakeEngine`` and stub environments — no GPU, no backend. Mirrors
``scripts/check_sample_roundtrip.py`` (``check_*`` contracts, ``main() -> int``, non-zero exit on
first failure). The load-bearing check is **tool-call-driven termination**: the loop runs exactly as
long as the (fake) model emits a marker, proving the loop is driven by the environment's ``done``,
not a fixed plan or counter.

    python scripts/rollout_loop_smoke.py
"""

from __future__ import annotations

import sys
from typing import Callable, List, Optional, Tuple

from unirl.rollout.loop import AgentLoop
from unirl.types.primitives import Texts
from unirl.types.sample import Part, Primitive, Sample
from unirl.types.sample_id import parent_id
from unirl.types.sampling import ARSamplingParams


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


class FakeEngine:
    """Model-free engine: fills the frontier gen shell with a deterministic ``Texts`` primitive
    keyed off its ``sample_ids`` (segment left None). Mirrors ``check_generate_fills_frontier``."""

    def generate(self, sample: Sample) -> Sample:
        shell = sample.parts[-1]
        filled = shell.fill(primitives={"text": Texts(texts=[f"gen::{sid}" for sid in shell.sample_ids])})
        return sample.with_parts([*sample.parts[:-1], filled])


class MarkerFakeEngine:
    """Fake engine that emits a ``<call>`` marker for the first ``marker_turns`` generations, then
    plain text — lets ``MarkerEnv`` terminate the loop based on the model's *output content*."""

    def __init__(self, marker_turns: int) -> None:
        self.marker_turns = marker_turns
        self.calls = 0

    def generate(self, sample: Sample) -> Sample:
        shell = sample.parts[-1]
        emit = self.calls < self.marker_turns
        self.calls += 1
        body = "<call>tool</call>" if emit else "final"
        filled = shell.fill(primitives={"text": Texts(texts=[f"{body}::{sid}" for sid in shell.sample_ids])})
        return sample.with_parts([*sample.parts[:-1], filled])


class FixedTurnsEnv:
    """Drives exactly ``turns`` generations, then ``done`` — an environment that owns the turn count."""

    def __init__(self, turns: int) -> None:
        self._remaining = turns

    def reset(self, request: Sample) -> Sample:
        return request

    def step(self, sample: Sample) -> Tuple[Optional[Primitive], bool, dict]:
        self._remaining -= 1
        if self._remaining <= 0:
            return None, True, {}
        ids = sample.parts[-1].sample_ids
        return Texts(texts=[f"obs::{sid}" for sid in ids]), False, {}


class MarkerEnv:
    """Tool-call-driven: continue while the model's output contains ``<call>``, else ``done``."""

    def reset(self, request: Sample) -> Sample:
        return request

    def step(self, sample: Sample) -> Tuple[Optional[Primitive], bool, dict]:
        frontier = sample.parts[-1]
        if not any("<call>" in t for t in frontier.primitives["text"].texts):
            return None, True, {}  # no tool call -> trajectory done
        ids = frontier.sample_ids
        return Texts(texts=[f"<tool_response>ok</tool_response>::{sid}" for sid in ids]), False, {}


def _request(prompts: List[str]) -> Sample:
    ids = [f"p{i}" for i in range(len(prompts))]
    return Sample.request(Part.input(ids, primitives={"text": Texts(texts=prompts)}))


def check_grpo_fanout_and_turns() -> None:
    """Turn 0 fans out to n=``samples_per_prompt``; continuations fork 1 each. ``FixedTurnsEnv(2)``
    drives 2 gen turns with one observation between them: ``[input, gen0(n), obs0(n), gen1(n)]``."""
    sp = ARSamplingParams(samples_per_prompt=3)
    loop = AgentLoop(environment=FixedTurnsEnv(2), sampling_params=sp, max_turns=8)
    out = loop.run(FakeEngine(), _request(["solve"]))
    _check(len(out.parts) == 4, f"FixedTurnsEnv(2) -> [input, gen, obs, gen]; got {len(out.parts)} parts")
    _check(len(out.parts[1].sample_ids) == 3, "turn 0 fans out to samples_per_prompt=3")
    _check(len(out.parts[3].sample_ids) == 3, "continuation forks 1 per sample (3 -> 3)")
    _check(
        all(parent_id(s) in set(out.parts[2].sample_ids) for s in out.parts[3].sample_ids),
        "gen1 ids are children of the observation Part (continuation off the frontier)",
    )


def check_single_turn() -> None:
    """``FixedTurnsEnv(1)`` -> exactly one gen turn, no observation: ``[input, gen]``."""
    loop = AgentLoop(environment=FixedTurnsEnv(1), sampling_params=ARSamplingParams(samples_per_prompt=2), max_turns=8)
    out = loop.run(FakeEngine(), _request(["a", "b"]))
    _check(len(out.parts) == 2, f"single turn -> [input, gen]; got {len(out.parts)}")
    _check(len(out.gen_parts()) == 1, "exactly one gen Part")


def check_gen_parts_masking() -> None:
    """gen Parts are trainable; observation Parts (no sampling_params) are excluded, even mid-chain."""
    loop = AgentLoop(environment=FixedTurnsEnv(3), sampling_params=ARSamplingParams(samples_per_prompt=1), max_turns=8)
    out = loop.run(FakeEngine(), _request(["x"]))
    _check(len(out.parts) == 6, f"3 turns -> [input, gen, obs, gen, obs, gen]; got {len(out.parts)}")
    _check(len(out.gen_parts()) == 3, "exactly the 3 gen Parts are trainable")
    obs_parts = [p for p in out.parts[1:] if p.sampling_params is None]
    _check(len(obs_parts) == 2, "2 mask-0 observation Parts present and excluded from gen_parts")


def check_conditioning() -> None:
    """``conditioning()`` surfaces every ancestor primitive after N turns (prompt + gens + observations)."""
    loop = AgentLoop(environment=FixedTurnsEnv(3), sampling_params=ARSamplingParams(samples_per_prompt=1), max_turns=8)
    out = loop.run(FakeEngine(), _request(["solve it"]))
    cond = out.conditioning()
    _check(len(cond) == 5, f"frontier conditioning sees 5 ancestors (prompt + 2 gen + 2 obs); got {len(cond)}")
    _check(all(isinstance(c, Texts) for c in cond), "conditioning surfaces the Texts primitives")


def check_split_concat_roundtrip() -> None:
    """``split()`` -> ``concat()`` round-trips the multi-part agentic tree ids (dp-shard safety)."""
    loop = AgentLoop(environment=FixedTurnsEnv(2), sampling_params=ARSamplingParams(samples_per_prompt=1), max_turns=8)
    out = loop.run(FakeEngine(), _request(["a", "b"]))
    merged = Sample.concat(out.split())
    _check(
        [list(p.sample_ids) for p in merged.parts] == [list(p.sample_ids) for p in out.parts],
        "split -> concat round-trips the per-part sample ids in order",
    )


def check_tool_call_driven_termination() -> None:
    """THE load-bearing check: the loop runs exactly as long as the model emits ``<call>``. The
    environment's ``done`` (not a plan / counter) drives termination. ``MarkerFakeEngine`` emits the
    marker for 2 turns then stops -> the loop must produce exactly 3 gen turns (2 with a call + 1 final)."""
    loop = AgentLoop(environment=MarkerEnv(), sampling_params=ARSamplingParams(samples_per_prompt=1), max_turns=16)
    out = loop.run(MarkerFakeEngine(marker_turns=2), _request(["question"]))
    _check(
        len(out.gen_parts()) == 3, f"loop should run 3 gen turns (2 with <call> + 1 final); got {len(out.gen_parts())}"
    )
    _check("final" in out.parts[-1].primitives["text"].texts[0], "final gen Part has no tool call (loop stopped there)")
    _check(len(out.parts) == 6, f"[input, gen, obs, gen, obs, gen]; got {len(out.parts)}")


_CHECKS: Tuple[Callable[[], None], ...] = (
    check_grpo_fanout_and_turns,
    check_single_turn,
    check_gen_parts_masking,
    check_conditioning,
    check_split_concat_roundtrip,
    check_tool_call_driven_termination,
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
        print("rollout-loop-smoke: FAILED", file=sys.stderr)
        for f in failures:
            print(f"  FAIL {f}", file=sys.stderr)
        return 1
    print(f"rollout-loop-smoke: {len(_CHECKS)} contracts hold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
