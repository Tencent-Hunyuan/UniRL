"""Environment — the world side of an agentic rollout turn (LIN-492)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol, Tuple

from unirl.types.primitives import PrimitiveValue
from unirl.types.sample import Sample


class Environment(Protocol):
    """The world side of a harness turn; concrete environments are separate."""

    max_turns: int

    def tool_schemas(self) -> List[Dict[str, Any]]:
        """The tools' JSON schemas, for ``apply_chat_template(tools=...)`` prompt injection."""
        ...

    def reset(self, request: Sample) -> Sample:
        """Optional per-episode setup; return the (possibly augmented) request Sample."""
        ...

    def step(self, sample: Sample) -> Tuple[Optional[PrimitiveValue], bool, dict]:
        """Consume the latest action; return ``(observation, done, info)``."""
        ...

    def close(self, sample: Sample) -> None:
        """Optional guaranteed teardown (LIN-533), called from the harness's ``finally`` on every"""
        ...


__all__ = ["Environment"]
