"""Composite reward — aggregates a panel of inner reward backends, with optional score-only monitors."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

from unirl.reward.base import BaseRewardComponentSpec, PromptVideoReward, RewardBackend
from unirl.reward.panel import aggregate_panel, monitor_key, validate_panel
from unirl.types.reward import RewardRequest, RewardResponse


class CompositeRewardScorer(RewardBackend):
    """Aggregate Hydra-instantiated inner reward backends of one modality; monitors are scored but not aggregated."""

    def __init__(self, *, config: "CompositeRewardSpec", base_device: str = "auto") -> None:
        super().__init__(model_name="composite")
        del base_device
        weights = {} if config.weights is None else dict(config.weights)
        validate_panel(
            "CompositeRewardScorer",
            training=config.scorers,
            monitors=config.monitors,
            weighted=weights,
            aggregation_method=config.aggregation_method,
        )
        unknown = sorted(set(weights) - set(config.scorers))
        if unknown:
            raise ValueError(f"CompositeRewardScorer: `weights` names {unknown}, which are not in `scorers`.")
        if any(not isinstance(weight, (int, float)) or not math.isfinite(weight) for weight in weights.values()):
            raise ValueError("CompositeRewardScorer: weights must be finite numbers.")
        self.aggregation_method = config.aggregation_method
        self.weights = {name: float(weights.get(name, 1.0)) for name in config.scorers}

        self._backends: Dict[str, RewardBackend] = {
            **config.scorers,
            **{monitor_key(name): monitor for name, monitor in config.monitors.items()},
        }
        for key, backend in self._backends.items():
            if not isinstance(backend, RewardBackend):
                raise TypeError(
                    f"CompositeRewardScorer: {key!r} is {type(backend).__name__}, not a "
                    "RewardBackend — each entry needs its own `_target_` for Hydra to instantiate."
                )
        kinds = {key: backend.preferred_input_kind for key, backend in self._backends.items()}
        if len(set(kinds.values())) != 1:
            raise ValueError(f"CompositeRewardScorer: inner backends must consume one input kind; got {kinds}.")
        self.input_kind = next(iter(kinds.values()))

        coverage = {
            name: scorer.covers_prompt_video()
            for name, scorer in config.scorers.items()
            if isinstance(scorer, PromptVideoReward)
        }
        if self.input_kind == "video" and coverage:
            if not any(self.weights[name] > 0.0 and covers for name, covers in coverage.items()):
                details = ", ".join(
                    f"{name}(weight={self.weights[name]}, covers_prompt_video={covers})"
                    for name, covers in coverage.items()
                )
                raise ValueError(
                    "CompositeRewardScorer: no positive-weight training scorer relates the prompt to the video. "
                    f"Current mix: {details}. "
                    "Add videopickscore / videoalign, or set scorers.imagebind.config.mode to "
                    "'text_video' or 'all'. clap and imagebind mode='audio_video' do not cover this, "
                    "and a monitor does not count."
                )

    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        start = time.time()
        bs = request.batch_size
        try:
            import torch

            components: Dict[str, "torch.Tensor"] = {}
            for key, backend in self._backends.items():
                resp = backend.compute_rewards(request)
                comp = torch.tensor(resp.rewards, dtype=torch.float32)  # [B]
                if comp.numel() != bs:
                    raise RuntimeError(
                        f"CompositeRewardScorer: inner backend {key!r} returned {comp.numel()} rewards "
                        f"for a batch of {bs}."
                    )
                if resp.successes and not all(resp.successes):
                    error = next((error for error in resp.errors if error), "unknown inner error")
                    raise RuntimeError(f"CompositeRewardScorer: inner backend {key!r} failed: {error}")
                if not torch.isfinite(comp).all():
                    raise RuntimeError(f"CompositeRewardScorer: inner backend {key!r} returned non-finite rewards.")
                components[key] = comp

            weights = list(self.weights.values())
            rewards = [
                float(aggregate_panel(row, weights, self.aggregation_method))
                for row in zip(*(components[name] for name in self.weights))
            ]

            return RewardResponse(
                rewards=rewards,
                component_rewards={key: comp.tolist() for key, comp in components.items()},
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

    def load(self) -> None:
        for s in self._backends.values():
            s.load()

    def offload(self) -> None:
        for s in self._backends.values():
            s.offload()

    def onload(self) -> None:
        for s in self._backends.values():
            s.onload()

    def dispose(self) -> None:
        for s in self._backends.values():
            s.dispose()


@dataclass
class CompositeRewardSpec(BaseRewardComponentSpec):
    """Typed config for the composite reward."""

    scorers: Dict[str, RewardBackend] = field(default_factory=dict)
    weights: Optional[Dict[str, float]] = None
    monitors: Dict[str, RewardBackend] = field(default_factory=dict)
    aggregation_method: str = "weighted_sum"


__all__ = ["CompositeRewardScorer", "CompositeRewardSpec"]
