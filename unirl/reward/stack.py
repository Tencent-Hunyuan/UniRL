"""Worker-local micro-batch scheduler: score each generated micro while the next one generates."""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.distributed.group.remote import Remote
from unirl.types.sample import Part, Sample


class RewardStack(Remote):
    """Generate one DP shard in micro-batches and overlap each micro's scoring with the next micro's generation."""

    def __init__(
        self,
        *,
        rollout: Any,
        reward: Any,
        micro_batch_size: int,
        overlap: bool = False,
    ) -> None:
        super().__init__()
        cls = type(self).__name__
        if int(micro_batch_size) < 1:
            raise ValueError(f"{cls}.micro_batch_size must be >= 1; got {micro_batch_size}")
        if reward is None:
            raise ValueError(f"{cls} requires a reward sibling; build it only for recipes with a `reward:` block.")
        # Siblings resolve to the live local roles, so both calls below stay in-process.
        self.rollout = rollout
        self.reward = reward
        self.micro_batch_size = int(micro_batch_size)
        self.overlap = bool(overlap)
        self._rows = 0
        self._micros = 0
        self._generate_s = 0.0
        self._score_s = 0.0
        self._wall_s = 0.0

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def rollout_and_score(self, sample: Sample) -> Sample:
        """Fill this shard's frontier Part micro by micro and return it with rewards attached."""
        gen = sample.parts[-1]
        total = int(gen.batch_size)
        bounds = self._bounds(gen)
        self._rows, self._micros, self._generate_s, self._score_s = total, len(bounds), 0.0, 0.0
        started = time.perf_counter()
        try:
            if len(bounds) == 1:
                return self._score(self._generate(sample, gen, 0, total))
            parts = self._overlapped(sample, gen, bounds) if self.overlap else self._serial(sample, gen, bounds)
            return sample.replace_frontier(Part.concat(parts))
        finally:
            self._wall_s = time.perf_counter() - started

    def _bounds(self, gen: Part) -> List[Tuple[int, int]]:
        """Micro slices of at least micro_batch_size rows, each extended to the end of the group it would split."""
        total = int(gen.batch_size)
        groups = gen.group_ids
        bounds: List[Tuple[int, int]] = []
        start = 0
        while start < total:
            end = min(start + self.micro_batch_size, total)
            while end < total and groups[end] == groups[end - 1]:
                end += 1
            bounds.append((start, end))
            start = end
        return bounds

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def timing(self) -> Dict[str, float]:
        """Shape and generate/score/wall seconds of this rank's last rollout_and_score call."""
        return {
            "rows": float(self._rows),
            "micros": float(self._micros),
            "generate_s": self._generate_s,
            "score_s": self._score_s,
            "wall_s": self._wall_s,
        }

    def _serial(self, sample: Sample, gen: Part, bounds: Sequence[Tuple[int, int]]) -> List[Part]:
        """Generate then score each micro in turn — the default path; see the reward README."""
        return [self._score(self._generate(sample, gen, start, end)).parts[-1] for start, end in bounds]

    def _overlapped(self, sample: Sample, gen: Part, bounds: Sequence[Tuple[int, int]]) -> List[Part]:
        """Keep one scoring call in flight while the next micro generates; a scorer raise surfaces here."""
        parts: List[Part] = []
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="reward-stack") as pool:
            inflight: Optional[Future] = None
            for start, end in bounds:
                generated = self._generate(sample, gen, start, end)
                if inflight is not None:
                    parts.append(inflight.result())
                inflight = pool.submit(self._score_part, generated)
            if inflight is not None:
                parts.append(inflight.result())
        return parts

    def _generate(self, sample: Sample, gen: Part, start: int, end: int) -> Sample:
        micro = sample if start == 0 and end == int(gen.batch_size) else sample.replace_frontier(gen.slice(start, end))
        t0 = time.perf_counter()
        generated = self.rollout.generate(micro)
        self._generate_s += time.perf_counter() - t0
        return generated

    def _score(self, generated: Sample) -> Sample:
        t0 = time.perf_counter()
        scored = self.reward.score_and_attach(generated)
        self._score_s += time.perf_counter() - t0
        return scored

    def _score_part(self, generated: Sample) -> Part:
        return self._score(generated).parts[-1]


__all__ = ["RewardStack"]
