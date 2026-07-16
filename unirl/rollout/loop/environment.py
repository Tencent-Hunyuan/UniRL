"""Environment — the world side of an agent-loop turn (LIN-492).

See ``unirl/rollout/loop/README.md``. A structural ``Protocol`` seam only; concrete
environments (tool / critic) are designed separately. The loop treats it as optional.
"""

from __future__ import annotations

from typing import Optional, Protocol, Tuple

from unirl.types.sample import Primitive, Sample


class Environment(Protocol):
    """The world side of a turn. Optional for the loop; concrete environments are separate."""

    def reset(self, request: Sample) -> Sample:
        """Optional per-episode setup; return the (possibly augmented) request Sample."""
        ...

    def step(self, sample: Sample) -> Tuple[Optional[Primitive], bool, dict]:
        """Consume the latest action; return ``(observation, done, info)``.

        ``observation is None`` appends nothing this turn; ``done`` ends the episode.
        Must be re-entrant/thread-safe: one env instance serves many concurrent
        trajectories on a worker (the agentic drain calls it from one thread per
        trajectory), so per-trajectory state is derived from the sample, not
        held on the instance — or is lock-guarded.
        """
        ...

    def close(self, sample: Sample) -> None:
        """Optional guaranteed teardown (LIN-533), called from the engine's ``finally`` on every
        path — success, crash, and abort. The engine invokes it via ``getattr(env, "close", None)``,
        so an env holding no per-trajectory resource need not implement it (default: no-op). Stateful
        envs use it to release handles exactly once — tool sessions
        (:meth:`~unirl.rollout.loop.tool_environment.ToolEnvironment.close`) or ALFWorld episodes/
        pooled templates. Must be idempotent and **must not raise**.
        """
        ...


__all__ = ["Environment"]
