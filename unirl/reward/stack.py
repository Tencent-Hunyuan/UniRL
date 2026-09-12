"""Worker-local micro-batch scheduler: score each generated micro while the next one generates."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, List, Optional, Sequence, Tuple

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
        overlap: bool = True,
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

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def rollout_and_score(self, sample: Sample) -> Sample:
        """Fill this shard's frontier Part micro by micro and return it with rewards attached."""
        gen = sample.parts[-1]
        total = int(gen.batch_size)
        if total <= self.micro_batch_size:
            return self._score(self.rollout.generate(sample))
        bounds = [
            (start, min(start + self.micro_batch_size, total)) for start in range(0, total, self.micro_batch_size)
        ]
        parts = self._overlapped(sample, gen, bounds) if self.overlap else self._serial(sample, gen, bounds)
        return sample.replace_frontier(Part.concat(parts))

    def _serial(self, sample: Sample, gen: Part, bounds: Sequence[Tuple[int, int]]) -> List[Part]:
        """Generate then score each micro in turn — the control arm for the overlapped path."""
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
        return self.rollout.generate(sample.replace_frontier(gen.slice(start, end)))

    def _score(self, generated: Sample) -> Sample:
        return self.reward.score_and_attach(generated)

    def _score_part(self, generated: Sample) -> Part:
        return self._score(generated).parts[-1]


__all__ = ["RewardStack"]
