"""Content-primary TTS reward with explicit quality gates and penalties."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, fields as dataclass_fields
from typing import Any, Dict, List

from unirl.reward.base import BaseRewardComponentSpec, RewardBackend
from unirl.types.reward import RewardRequest, RewardResponse

from .registry import resolve_builtin_reward_scorer_class, resolve_builtin_reward_spec_class


class TTSCompositeScorer(RewardBackend):
    """Weighted reward where content dominates and quality can veto."""

    input_kind = "audio"

    def __init__(self, *, config: "TTSCompositeSpec", base_device: str) -> None:
        super().__init__(model_name="tts_composite", batch_size=config.batch_size)
        self.weights: Dict[str, float] = dict(config.weights or {})
        if not self.weights:
            raise ValueError("TTSCompositeScorer requires non-empty weights.")
        if any(float(weight) < 0 for weight in self.weights.values()):
            raise ValueError("TTSCompositeScorer weights must be non-negative.")
        weight_sum = float(sum(self.weights.values()))
        content_fraction = float(self.weights.get(config.content_component, 0.0)) / weight_sum
        if content_fraction < config.min_content_weight_fraction:
            raise ValueError(
                f"Content component {config.content_component!r} contributes {content_fraction:.3f}, "
                f"below min_content_weight_fraction={config.min_content_weight_fraction:.3f}."
            )
        self._config = config
        self._scorers: Dict[str, RewardBackend] = {}
        for name in self.weights:
            inner_cls = resolve_builtin_reward_scorer_class(name)
            inner_spec_cls = resolve_builtin_reward_spec_class(name)
            allowed = {item.name for item in dataclass_fields(inner_spec_cls)}
            component_config = dict(config.component_configs.get(name, {}))
            unknown = sorted(set(component_config) - allowed)
            if unknown:
                raise ValueError(f"Unknown config keys for {name}: {unknown}.")
            if "device" in allowed:
                component_config.setdefault("device", config.device)
            if "batch_size" in allowed:
                component_config.setdefault("batch_size", config.batch_size)
            inner_spec = inner_spec_cls(**component_config)
            self._scorers[name] = inner_cls(config=inner_spec, base_device=base_device)
        self.training_eligible = all(
            bool(getattr(scorer, "training_eligible", True)) for scorer in self._scorers.values()
        )

    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        start = time.time()
        bs = request.batch_size
        try:
            import torch

            component_rewards: Dict[str, List[float]] = {}
            component_details: Dict[str, List[Dict[str, Any]]] = {}
            total = torch.zeros(bs, dtype=torch.float32)
            successes = [True] * bs
            sample_errors: List[List[str]] = [[] for _ in range(bs)]
            for name, scorer in self._scorers.items():
                resp = scorer.compute_rewards(request)
                comp = torch.tensor(list(resp.rewards), dtype=torch.float32)
                if comp.numel() != bs:
                    raise RuntimeError(
                        f"TTSCompositeScorer: inner scorer {name!r} returned {comp.numel()} for batch {bs}."
                    )
                component_rewards[name] = comp.tolist()
                component_details[name] = list(resp.details or [{} for _ in range(bs)])
                for metric_name, values in dict(resp.component_rewards or {}).items():
                    if len(values) == bs:
                        component_rewards[f"{name}/{metric_name}"] = [float(value) for value in values]
                for index in range(bs):
                    ok = index < len(resp.successes) and bool(resp.successes[index])
                    successes[index] = successes[index] and ok
                    if not ok:
                        error = resp.errors[index] if index < len(resp.errors) else "unknown component failure"
                        sample_errors[index].append(f"{name}: {error}")
                total = total + float(self.weights[name]) * comp
            total = total / float(sum(self.weights.values()))

            gate_pass = torch.ones(bs, dtype=torch.bool)
            for name, threshold in self._config.quality_gates.items():
                if name not in component_rewards:
                    raise ValueError(f"quality_gates references inactive component {name!r}.")
                passed = torch.tensor(component_rewards[name]) >= float(threshold)
                component_rewards[f"gate/{name}"] = passed.float().tolist()
                gate_pass &= passed

            penalty_multiplier = torch.ones(bs, dtype=torch.float32)
            for name, threshold in self._config.penalty_thresholds.items():
                if name not in component_rewards:
                    raise ValueError(f"penalty_thresholds references inactive component {name!r}.")
                factor = float(self._config.penalty_factors.get(name, 1.0))
                below = torch.tensor(component_rewards[name]) < float(threshold)
                penalty_multiplier = torch.where(
                    below,
                    penalty_multiplier * factor,
                    penalty_multiplier,
                )
                component_rewards[f"penalty/{name}"] = torch.where(
                    below,
                    torch.full((bs,), factor),
                    torch.ones(bs),
                ).tolist()
            total = torch.where(gate_pass, total * penalty_multiplier, torch.zeros_like(total))
            # A failed model/reference is not a legitimate zero reward. Keep the
            # numeric placeholder for shape stability but mark it failed so
            # RewardService refuses to attach it to a training sample.
            total = torch.where(torch.tensor(successes), total, torch.zeros_like(total))
            details = [
                {
                    "status": "available" if successes[index] else "failed",
                    "components": {
                        name: component_details[name][index] if index < len(component_details[name]) else {}
                        for name in self._scorers
                    },
                    "quality_gate_passed": bool(gate_pass[index]),
                    "penalty_multiplier": float(penalty_multiplier[index]),
                }
                for index in range(bs)
            ]
            return RewardResponse(
                rewards=total.tolist(),
                component_rewards=component_rewards,
                details=details,
                successes=successes,
                errors=["; ".join(items) if items else None for items in sample_errors],
                compute_time=time.time() - start,
            )
        except Exception as e:
            return RewardResponse(
                rewards=[0.0] * bs,
                successes=[False] * bs,
                errors=[str(e)] * bs,
                compute_time=time.time() - start,
            )

    @property
    def preferred_input_kind(self) -> str:
        return self.input_kind

    def is_available(self) -> bool:
        return all(s.is_available() for s in self._scorers.values())

    def offload(self) -> None:
        for s in self._scorers.values():
            if hasattr(s, "offload"):
                s.offload()

    def onload(self) -> None:
        for s in self._scorers.values():
            if hasattr(s, "onload"):
                s.onload()

    def dispose(self) -> None:
        for s in self._scorers.values():
            if hasattr(s, "dispose"):
                s.dispose()


@dataclass
class TTSCompositeSpec(BaseRewardComponentSpec):
    batch_size: int = 4
    device: str = "auto"
    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "tts_wer": 0.65,
            "tts_speaker_sim": 0.15,
            "tts_utmos": 0.15,
            "tts_stability": 0.05,
        }
    )
    component_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    content_component: str = "tts_wer"
    min_content_weight_fraction: float = 0.50
    quality_gates: Dict[str, float] = field(
        default_factory=lambda: {
            "tts_stability": 1.0,
            "tts_utmos": 0.30,
        }
    )
    penalty_thresholds: Dict[str, float] = field(default_factory=lambda: {"tts_speaker_sim": 0.40})
    penalty_factors: Dict[str, float] = field(default_factory=lambda: {"tts_speaker_sim": 0.50})

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_content_weight_fraction <= 1.0:
            raise ValueError("min_content_weight_fraction must be in [0, 1].")
        for name, factor in self.penalty_factors.items():
            if not 0.0 <= float(factor) <= 1.0:
                raise ValueError(f"penalty factor for {name!r} must be in [0, 1].")
