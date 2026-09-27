"""Abstract planner for adaptive (dynamic) Ulysses sequence parallelism."""

from __future__ import annotations

import abc
from collections.abc import Sequence


class SequenceParallelPlanner(abc.ABC):
    """Chooses a per-microbatch Ulysses degree from runtime sequence lengths."""

    @abc.abstractmethod
    def plan(self, seq_lens: Sequence[int], sp_max: int) -> list[int]:
        """Per-microbatch Ulysses degree (each a divisor of ``sp_max``), aligned to ``seq_lens``."""
        raise NotImplementedError


__all__ = ["SequenceParallelPlanner"]
