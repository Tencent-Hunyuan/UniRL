"""Typed structured tool calls for agent trajectories.

The canonical on-disk/interchange shape for tool calls is the OpenAI messages
convention (assistant message ``tool_calls`` entries; ``tool`` role results
paired by ``tool_call_id``). This module is the typed internal counterpart:
``ToolCalls`` rides on a :class:`~unirl.types.sample.Part` under the
``"tool_calls"`` primitive key and surfaces on the part's assistant
:class:`~unirl.types.sample.Turn`, so an agent's calls are typed data rather
than regex over rendered text. (``Turn`` is a read-only view over a ``Sample``;
:func:`unirl.types.agent_trace.turns_from_messages` parses a manifest history
into turns — there is no messages → ``Sample`` constructor.)

``arguments`` is ALWAYS a dict internally. OpenAI serializes
``function.arguments`` as a JSON-encoded string while HF chat templates expect
a dict (transformers docs warn the string form mis-renders); ingest normalizes
via :meth:`ToolCall.from_message_entry`, and :meth:`ToolCall.to_message_entry`
re-emits the dict form that ``apply_chat_template`` consumes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from unirl.distributed.tensor.batch import Batch, concat_field


@dataclass(frozen=True)
class ToolCall:
    """One structured function call: ``id`` pairs it with its tool-role result."""

    name: str
    # Excluded from __hash__ (dicts are unhashable); equality still compares it,
    # so equal calls keep equal hashes and set/dict dedup of parallel calls works.
    arguments: Dict[str, Any] = field(default_factory=dict, hash=False)
    id: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(f"ToolCall.name must be a non-empty string, got {self.name!r}.")
        if not isinstance(self.arguments, dict):
            raise TypeError(
                f"ToolCall.arguments must be a dict (normalize JSON strings at ingest), "
                f"got {type(self.arguments).__name__}."
            )
        if self.id is not None and not isinstance(self.id, str):
            raise TypeError(f"ToolCall.id must be a string or None, got {type(self.id).__name__}.")

    @classmethod
    def from_message_entry(cls, entry: Dict[str, Any], *, context: str = "ToolCall") -> "ToolCall":
        """Parse one OpenAI-style ``tool_calls`` entry, normalizing arguments to a dict.

        Accepts both the nested ``{"id", "type": "function", "function": {"name",
        "arguments"}}`` shape and a flat ``{"id", "name", "arguments"}`` shape;
        ``arguments`` may be a dict or a JSON-encoded string (the OpenAI wire form).
        """
        if not isinstance(entry, dict):
            raise TypeError(f"{context}: tool_calls entries must be dicts, got {type(entry).__name__}.")
        if "function" in entry and not isinstance(entry["function"], dict):
            raise TypeError(
                f"{context}: tool_calls entry 'function' must be a dict, got {type(entry['function']).__name__}."
            )
        fn = entry.get("function") if isinstance(entry.get("function"), dict) else entry
        name = fn.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{context}: tool_calls entry missing function name: {entry!r}.")
        arguments = fn.get("arguments", {})
        if isinstance(arguments, str):
            text = arguments.strip() or "{}"
            try:
                arguments = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{context}: tool call {name!r} arguments is not valid JSON: {text!r}.") from exc
        if not isinstance(arguments, dict):
            raise TypeError(
                f"{context}: tool call {name!r} arguments must decode to an object, got {type(arguments).__name__}."
            )
        call_id = entry.get("id")
        return cls(name=name, arguments=arguments, id=str(call_id) if call_id is not None else None)

    def to_message_entry(self) -> Dict[str, Any]:
        """The OpenAI-style ``tool_calls`` entry, with dict ``arguments`` (HF template form)."""
        entry: Dict[str, Any] = {"type": "function", "function": {"name": self.name, "arguments": self.arguments}}
        if self.id is not None:
            entry["id"] = self.id
        return entry


@dataclass
class ToolCalls(Batch):
    """Batch-aligned sparse structured tool calls (mirror of :class:`MediaRefs`).

    ``rows[i]`` is the ordered list of calls the assistant made for sample ``i``
    (empty row = no call; multiple entries = parallel calls). Sparsity stays
    inside each row so CONCAT selection, slicing, and transport remain
    row-aligned under the rectangular :class:`Batch` contract.

    Builder contract (shared by every ``Part.primitives`` key): same-position
    parts that get concatenated must agree on key presence. A builder whose
    batches can mix call-free and calling samples attaches ``ToolCalls`` with
    empty rows unconditionally rather than only when calls exist.
    """

    rows: List[List[ToolCall]] = concat_field(default_factory=list)

    def __post_init__(self) -> None:
        normalized: List[List[ToolCall]] = []
        for row, calls in enumerate(self.rows):
            if not isinstance(calls, (list, tuple)):
                raise TypeError(f"ToolCalls.rows[{row}] must be a list of ToolCall values.")
            values = list(calls)
            invalid = [type(call).__name__ for call in values if not isinstance(call, ToolCall)]
            if invalid:
                raise TypeError(f"ToolCalls.rows[{row}] contains non-ToolCall values: {invalid}.")
            normalized.append(values)
        self.rows = normalized

    @classmethod
    def from_rows(cls, rows: List[List[ToolCall]]) -> "ToolCalls":
        return cls(rows=[list(row) for row in rows])

    def __len__(self) -> int:
        return len(self.rows)


__all__ = ["ToolCall", "ToolCalls"]
