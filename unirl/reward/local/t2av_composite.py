"""T2AV composite reward — weighted blend of video + audio scorers."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List

from unirl.reward.base import BaseRewardComponentSpec, PromptVideoReward, RewardBackend
from unirl.types.reward import RewardRequest, RewardResponse


def _require_prompt_video_term(weights: Dict[str, float], scorers: Dict[str, RewardBackend]) -> None:
    coverage = {
        name: isinstance(scorer, PromptVideoReward) and scorer.covers_prompt_video() for name, scorer in scorers.items()
    }
    if any(weights[name] > 0.0 and coverage[name] for name in scorers):
        return
    details = ", ".join(f"{name}(weight={weights[name]}, covers_prompt_video={coverage[name]})" for name in weights)
    raise ValueError(
        "T2AVCompositeScorer: no positive-weight inner scorer relates the prompt to the video. "
        f"Current mix: {details}. "
        "Add videopickscore / videoalign, or set scorers.imagebind.config.mode to "
        "'text_video' or 'all'. clap and imagebind mode='audio_video' do not cover this."
    )


class T2AVCompositeScorer(RewardBackend):
    """Weighted blend of Hydra-instantiated inner reward scorers for T2AV."""

    input_kind = "video"

    def __init__(self, *, config: "T2AVCompositeSpec", base_device: str = "auto") -> None:
        super().__init__(model_name="t2av_composite")
        del base_device
        if not config.weights:
            raise ValueError("T2AVCompositeScorer requires a non-empty `weights` dict (scorer_name -> weight).")
        if not config.scorers:
            raise ValueError("T2AVCompositeScorer requires a non-empty `scorers` mapping (name -> RewardBackend).")

        weights = dict(config.weights)
        if any(not isinstance(weight, (int, float)) or not math.isfinite(weight) for weight in weights.values()):
            raise ValueError("T2AVCompositeScorer: weights must be finite numbers.")
        self.weights = {name: float(weight) for name, weight in weights.items()}

        for name, scorer in config.scorers.items():
            if not isinstance(scorer, RewardBackend):
                raise TypeError(
                    f"T2AVCompositeScorer: scorers[{name!r}] is {type(scorer).__name__}, not a "
                    "RewardBackend — each entry needs its own `_target_` for Hydra to instantiate."
                )
            input_kind = scorer.preferred_input_kind
            if input_kind != self.input_kind:
                raise ValueError(
                    f"T2AVCompositeScorer: scorers[{name!r}] consumes {input_kind!r}, expected {self.input_kind!r}."
                )

        weight_names = set(self.weights)
        scorer_names = set(config.scorers)
        if weight_names != scorer_names:
            missing = sorted(weight_names - scorer_names)
            extra = sorted(scorer_names - weight_names)
            raise ValueError(
                "T2AVCompositeScorer: `weights` and `scorers` must use the same keys "
                f"(weights missing scorers={missing}, scorers missing weights={extra})."
            )

        self._scorers: Dict[str, RewardBackend] = dict(config.scorers)
        _require_prompt_video_term(self.weights, self._scorers)

    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        start = time.time()
        bs = request.batch_size
        try:
            import torch

            component_rewards: Dict[str, List[float]] = {}
            total = torch.zeros(bs, dtype=torch.float32)
            for name, scorer in self._scorers.items():
                resp = scorer.compute_rewards(request)
                comp = torch.tensor(resp.rewards, dtype=torch.float32)
                if comp.numel() != bs:
                    raise RuntimeError(
                        f"T2AVCompositeScorer: inner scorer {name!r} returned {comp.numel()} rewards "
                        f"for a batch of {bs}."
                    )
                if resp.successes and not all(resp.successes):
                    error = next((error for error in (resp.errors or []) if error), "unknown inner error")
                    raise RuntimeError(f"T2AVCompositeScorer: inner scorer {name!r} failed: {error}")
                component_rewards[name] = comp.tolist()
                total = total + self.weights[name] * comp

            return RewardResponse(
                rewards=total.tolist(),
                component_rewards=component_rewards,
                successes=[True] * bs,
                errors=[None] * bs,
                compute_time=time.time() - start,
            )
        except Exception as e:
            return RewardResponse(
                rewards=[0.0] * bs,
                successes=[False] * bs,
                errors=[str(e)] * bs,
                compute_time=time.time() - start,
            )

    def covers_prompt_video(self) -> bool:
        return True

    def is_available(self) -> bool:
        return all(s.is_available() for s in self._scorers.values())

    def offload(self) -> None:
        for s in self._scorers.values():
            s.offload()

    def onload(self) -> None:
        for s in self._scorers.values():
            s.onload()

    def dispose(self) -> None:
        for s in self._scorers.values():
            s.dispose()


@dataclass
class T2AVCompositeSpec(BaseRewardComponentSpec):
    """Typed config for the T2AV composite reward."""

    weights: Dict[str, float] = field(default_factory=dict)
    scorers: Dict[str, RewardBackend] = field(default_factory=dict)


__all__ = ["T2AVCompositeScorer", "T2AVCompositeSpec"]
