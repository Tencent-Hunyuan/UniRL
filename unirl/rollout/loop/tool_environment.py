"""ToolEnvironment — the first concrete agent-loop :class:`Environment` (LIN-492).

Turns the policy's ``<tool_call>{...}</tool_call>`` into a tool execution and feeds the result
back as the next turn's observation, ending the episode when the model stops calling tools (it
gave a final answer) or ``max_turns`` is hit. See ``unirl/rollout/loop/README.md`` and the
real-world references it mirrors (relax ``DeepeyesEnv``, slime ``Geo3kEnv``): the loop stays
mechanical; the *decision* — parse → execute → done — lives here.

**Contract with the engine's (separately built) multi-turn conditioning.** The observation is the
**raw** tool-result text (a :class:`Texts` primitive, no ``<tool_response>`` markup). The engine
adapter maps an observation Part to a ``{"role": "tool", "content": ...}`` chat message and the
tokenizer's chat template adds the ``<tool_response>`` wrapper — so we never double-wrap. Tool
schemas for the prompt come from :meth:`ToolEnvironment.tool_schemas`.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple
from uuid import uuid4

from unirl.rollout.loop.tools.tool import StatefulTool, Tool
from unirl.types.primitives import Texts
from unirl.types.sample import Primitive, Sample, _part_with_field

logger = logging.getLogger(__name__)

# ``<tool_call>{...}</tool_call>`` — the Hermes/Qwen convention (matches relax/slime/areal).
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
# Opener for a tool call whose ``</tool_call>`` was trimmed by a ``stop`` string mid-generation.
_TOOL_CALL_OPEN_RE = re.compile(r"<tool_call>\s*(\{)", re.DOTALL)


def _balanced_json(text: str, start: int) -> Optional[str]:
    """Extract a brace-balanced JSON object beginning at ``start`` (the index of its ``{``).

    String-aware so braces inside quotes don't unbalance it — recovers a tool call whose closing
    ``</tool_call>`` was eaten by a ``stop`` string during generation.
    """
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
        elif not in_string:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


def parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Parse the last ``<tool_call>{...}</tool_call>`` in ``text`` into ``{"name", "arguments"}``.

    Returns ``None`` when there is no parseable tool call — the environment reads that as a final
    answer (the trajectory is done). Tolerant of a stop-trimmed closing tag (balanced-brace
    fallback) and of ``arguments`` arriving as a JSON-encoded string.
    """
    raw_json: Optional[str] = None
    matches = list(_TOOL_CALL_RE.finditer(text))
    if matches:
        raw_json = matches[-1].group(1).strip()
    else:
        opener = _TOOL_CALL_OPEN_RE.search(text)
        if opener is not None:
            raw_json = _balanced_json(text, opener.start(1))
    if raw_json is None:
        return None

    try:
        payload = json.loads(raw_json)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None

    name = payload.get("name") or (payload.get("function") or {}).get("name")
    arguments = payload.get("arguments")
    if arguments is None:
        arguments = (payload.get("function") or {}).get("arguments")
    if arguments is None:
        arguments = {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (json.JSONDecodeError, ValueError):
            return None
    if not name or not isinstance(arguments, dict):
        return None
    return {"name": name, "arguments": arguments}


class ToolEnvironment:
    """Agentic :class:`Environment`: parse tool calls, run tools, observe results, stop on a final answer.

    Drives :class:`~unirl.rollout.loop.agent_loop.AgentLoop` over a frontier of one-or-more samples
    (the GRPO group / continuations). :meth:`step` parses each frontier sample's text, executes any
    tool call, and returns a row-aligned observation. The batch's ``done`` is True once **no** sample
    emits a tool call (all gave a final answer) or ``max_turns`` is reached.
    """

    def __init__(self, tools: Sequence[Tool], max_turns: int = 6) -> None:
        if not tools:
            raise ValueError("ToolEnvironment requires at least one tool")
        self._tools: Dict[str, Tool] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"duplicate tool name: {tool.name!r}")
            self._tools[tool.name] = tool
        # Tools that hold per-trajectory session state (LIN-533); the stateless path skips all
        # session plumbing whenever this list is empty (zero regression for calculator/search).
        self._stateful_tools: List[StatefulTool] = [t for t in self._tools.values() if isinstance(t, StatefulTool)]
        self.max_turns = max_turns

    def tool_schemas(self) -> List[Dict[str, Any]]:
        """The tools' JSON schemas, for ``apply_chat_template(tools=...)`` prompt injection."""
        return [tool.json_schema() for tool in self._tools.values()]

    def reset(self, request: Sample) -> Sample:
        """Per-episode setup.

        Stateless tools: the request is returned **unchanged** (re-entrant, LIN-522 — the turn
        count is derived from the sample in :meth:`step`, not held on the instance, so one env
        instance serves many concurrent trajectories on a worker).

        Stateful tools (LIN-533): mint a per-trajectory ``session_id`` (``uuid4``) for each
        :class:`~unirl.rollout.loop.tools.tool.StatefulTool`, call ``session_start`` (cheap — the
        handle opens lazily in :meth:`step`), and stamp the ids into the root Part's *control* bag
        under ``"tool_sessions"`` so :meth:`step`/:meth:`close` recover them position-independently
        across the fork/observe chain. ``uuid4`` avoids collisions between the ``n`` GRPO siblings
        that share a root ``sample_id``.
        """
        if not self._stateful_tools:
            return request
        root = request.parts[0]
        context = (root.metadata or [None])[0] or {}
        sessions: Dict[str, str] = {}
        for tool in self._stateful_tools:
            sid = uuid4().hex
            tool.session_start(sid, context)
            sessions[tool.name] = sid
        control = dict(root.control or {})
        control["tool_sessions"] = sessions
        # ``_part_with_field`` swaps only ``control`` — preserving the encoded prompt
        # (``primitives``/``segment``/``metadata``), unlike a ``Part.input`` rebuild.
        return Sample.request(_part_with_field(root, "control", control))

    def step(self, sample: Sample) -> Tuple[Optional[Primitive], bool, dict]:
        """Consume the frontier action; return ``(observation, done, info)``.

        Reads ``sample.parts[-1].primitives['text'].texts`` (one text per frontier sample), executes any tool
        call per row, and returns a row-aligned :class:`Texts` observation. ``done`` once no row
        called a tool, or at ``max_turns``.

        Stateless / re-entrant: the turn number is derived from the sample
        (``len(sample.gen_parts())`` — one gen Part per turn so far), not a mutable
        instance counter, so concurrent trajectories sharing one env don't clobber it.
        """
        turn = len(sample.gen_parts())
        frontier = sample.parts[-1].primitives.get("text")
        if not isinstance(frontier, Texts):
            raise TypeError(f"ToolEnvironment.step expects a Texts frontier primitive; got {type(frontier).__name__}")
        texts = frontier.texts

        sessions = (sample.parts[0].control or {}).get("tool_sessions", {})
        calls = [parse_tool_call(t) for t in texts]
        results: List[Optional[str]] = [self._run(c, sessions) if c is not None else None for c in calls]
        per_sample_done = [c is None for c in calls]
        any_call = any(c is not None for c in calls)

        info = {
            "turn": turn,
            "tool_calls": calls,
            "results": results,
            "per_sample_done": per_sample_done,
        }

        if (not any_call) or (turn >= self.max_turns):
            return None, True, info

        # Row-aligned observation: the tool result for rows that called a tool; "" placeholder for
        # rows that already produced a final answer while siblings continue (the n>1 heterogeneous
        # case — a known follow-up that belongs to the loop/Sample layer, not the environment).
        observation = Texts(texts=[r if r is not None else "" for r in results])
        return observation, False, info

    def _run(self, call: Dict[str, Any], sessions: Dict[str, str]) -> str:
        """Dispatch one parsed call to its tool; surface any failure to the model as text.

        Stateful tools (LIN-533) go through ``execute_session`` with the ``session_id`` recovered
        from the root control bag; stateless tools keep the pure ``execute`` path.
        """
        name = call["name"]
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: unknown tool {name!r}. Available tools: {sorted(self._tools)}."
        args = call.get("arguments") or {}
        try:
            if isinstance(tool, StatefulTool):
                sid = sessions.get(name)
                if sid is None:
                    return f"Error: no active session for stateful tool {name!r}."
                return tool.execute_session(sid, args)
            return tool.execute(args)
        except Exception as exc:  # noqa: BLE001 — tool errors are fed back to the model, not raised
            return f"Error: {exc}"

    def close(self, sample: Sample) -> None:
        """Guaranteed teardown (LIN-533): end every open tool session for this trajectory.

        The engine calls this from ``_run_one``'s ``finally`` on every path — success, crash, and
        abort — on the trajectory's own drain thread. Swallows per-session errors so teardown can
        never destabilize the drain (``_run_one`` must not raise). A no-op for stateless tools /
        sessionless trajectories.
        """
        if not self._stateful_tools:
            return
        sessions = (sample.parts[0].control or {}).get("tool_sessions", {}) if sample.parts else {}
        if not sessions:
            return
        self._end_sessions(sessions)

    def _end_sessions(self, sessions: Dict[str, str]) -> None:
        """Run ``session_end`` for each open session; swallow + log failures (idempotent)."""
        for name, sid in sessions.items():
            tool = self._tools.get(name)
            if isinstance(tool, StatefulTool):
                try:
                    tool.session_end(sid)
                except Exception:  # noqa: BLE001 — teardown must not raise into the drain loop
                    logger.warning("session_end failed for tool %r session %s", name, sid, exc_info=True)


__all__ = ["ToolEnvironment", "parse_tool_call"]
