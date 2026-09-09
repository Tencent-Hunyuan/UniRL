"""The harness boundary: what a task-internal control flow may see and return."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Literal, Mapping, Protocol

if TYPE_CHECKING:
    from unirl.types.sample import Sample


@dataclass(frozen=True)
class HarnessOutcome:
    """What one task run produced."""

    sample: "Sample"
    status: Literal["completed", "suspended", "failed"]


@dataclass(frozen=True)
class HarnessContext:
    """The runtime surface a harness runs against: named engines + a stop probe."""

    engines: Mapping[str, Callable[["Sample"], "Sample"]] = field(default_factory=dict)
    suspend: Callable[[], bool] = lambda: False

    def generate(self, engine: str, sample: "Sample") -> "Sample":
        """One blocking model call on the named engine."""
        try:
            gen = self.engines[engine]
        except KeyError:
            raise KeyError(
                f"harness asked for engine {engine!r}; this runtime provides {sorted(self.engines)}"
            ) from None
        return gen(sample)

    def suspend_requested(self) -> bool:
        """True once the runtime wants a cooperative stop (quiesce / weight sync)."""
        return self.suspend()


class RolloutHarness(Protocol):
    """One task's internal control flow, hosted by a rollout-worker runtime."""

    def run(self, request: "Sample", context: HarnessContext) -> HarnessOutcome: ...


__all__ = ["HarnessContext", "HarnessOutcome", "RolloutHarness"]
