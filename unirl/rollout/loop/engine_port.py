"""RolloutEnginePort — the generation seam the agent loop calls (LIN-492).

See ``unirl/rollout/loop/README.md``. A structural ``Protocol`` for a single-turn
engine; ``SyncRolloutEngine`` is its nominal runtime counterpart.
"""

from __future__ import annotations

from typing import Protocol

from unirl.types.sample import Sample


class RolloutEnginePort(Protocol):
    """What the loop calls to generate one model turn."""

    def generate(self, sample: Sample) -> Sample:
        """Fill the request Sample's frontier gen Part and return it.

        One turn is always one ``Sample`` (never a trajectory list) — that
        list belongs to the coordinator contract.
        """
        ...


__all__ = ["RolloutEnginePort"]
