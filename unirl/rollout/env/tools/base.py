"""Tool — the abstract interface a ``ToolEnvironment`` dispatches to."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Union

from unirl.types.primitives import PrimitiveValue

ToolResult = Union[str, PrimitiveValue]


class Tool(ABC):
    """A single callable tool: a name, a JSON function-schema, and an executor."""

    name: str

    @abstractmethod
    def json_schema(self) -> Dict[str, Any]:
        """The OpenAI function-tool schema — the shape ``apply_chat_template(tools=[...])`` consumes."""
        ...

    @abstractmethod
    def execute(self, arguments: Dict[str, Any]) -> ToolResult:
        """Run the tool and return text or one batch-aligned multimodal observation."""
        ...


class StatefulTool(Tool):
    """A :class:`Tool` that holds **per-trajectory state** behind a session handle."""

    def session_start(self, session_id: str, context: Dict[str, Any]) -> None:
        """Open a session. Default no-op — a light tool may allocate lazily in ``execute_session``."""

    @abstractmethod
    def execute_session(self, session_id: str, arguments: Dict[str, Any]) -> ToolResult:
        """Run the session tool and return text or one multimodal observation."""
        ...

    def session_end(self, session_id: str) -> None:
        """Tear down a session. Default no-op. Idempotent, no-op on unknown ids, never raises."""

    def execute(self, arguments: Dict[str, Any]) -> ToolResult:  # pragma: no cover - guard
        """Stateless entrypoint — a programming error for a session-scoped tool; use :meth:`execute_session`."""
        raise NotImplementedError("StatefulTool is session-scoped; call execute_session(session_id, ...)")


__all__ = ["Tool", "ToolResult", "StatefulTool"]
