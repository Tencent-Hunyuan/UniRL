"""T2AV composite reward — weighted blend of video + audio scorers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from unirl.reward.base import BaseRewardComponentSpec, RewardBackend
from unirl.types.reward import RewardRequest, RewardResponse

from .registry import resolve_builtin_reward_scorer_class, resolve_builtin_reward_spec_class


class T2AVCompositeScorer(RewardBackend):
    """Weighted blend of inner reward scorers for T2AV (video + audio)."""

    input_kind = "video"

    def __init__(self, *, config: "T2AVCompositeSpec", base_device: str) -> None:
        super().__init__(model_name="t2av_composite", batch_size=config.batch_size)
        # A zero-weight component must be operationally disabled, not merely
        # multiplied by zero after loading and inference. This lets recipes
        # switch off expensive modalities without retaining their model or
        # allowing a failed/NaN scorer to poison the composite.
        self.weights: Dict[str, float] = {
            str(name): float(weight) for name, weight in dict(config.weights or {}).items() if float(weight) != 0.0
        }
        if not self.weights:
            raise ValueError("T2AVCompositeScorer requires at least one non-zero weight.")

        self._scorers: Dict[str, RewardBackend] = {}
        for name in self.weights:
            inner_cls = resolve_builtin_reward_scorer_class(name)
            inner_spec_cls = resolve_builtin_reward_spec_class(name)
            inner_spec = inner_spec_cls()
            import dataclasses

            # Propagate only fields BOTH the composite and the inner spec declare.
            shared = (
                "device",
                "batch_size",
                "frame_selection",
                "num_score_frames",
                "frame_aggregation",
                "topk_frames",
                "all_frame_mean_weight",
            )
            overrides = {f: getattr(config, f) for f in shared if hasattr(inner_spec, f) and hasattr(config, f)}
            if name == "clap" and config.clap_model_id and hasattr(inner_spec, "model_id"):
                overrides["model_id"] = config.clap_model_id
            if name == "imagebind":
                if config.imagebind_model_path and hasattr(inner_spec, "model_path"):
                    overrides["model_path"] = config.imagebind_model_path
                if config.imagebind_mode and hasattr(inner_spec, "mode"):
                    overrides["mode"] = config.imagebind_mode
            if overrides:
                inner_spec = dataclasses.replace(inner_spec, **overrides)
            self._scorers[name] = inner_cls(config=inner_spec, base_device=base_device)

    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        start = time.time()
        bs = request.batch_size
        try:
            import torch

            component_rewards: Dict[str, List[float]] = {}
            total = torch.zeros(bs, dtype=torch.float32)
            for name, scorer in self._scorers.items():
                resp = scorer.compute_rewards(request)
                if len(resp.successes) != bs or len(resp.errors) != bs:
                    raise RuntimeError(
                        f"T2AVCompositeScorer: inner scorer {name!r} returned "
                        f"{len(resp.successes)} success flags and {len(resp.errors)} errors for a batch of {bs}."
                    )
                failed = [(i, error) for i, (ok, error) in enumerate(zip(resp.successes, resp.errors)) if not ok]
                if failed:
                    raise RuntimeError(
                        f"T2AVCompositeScorer: inner scorer {name!r} failed "
                        f"{len(failed)} of {bs} samples; first few: {failed[:3]}"
                    )
                comp = torch.tensor(list(resp.rewards), dtype=torch.float32)
                if comp.numel() != bs:
                    raise RuntimeError(
                        f"T2AVCompositeScorer: inner scorer {name!r} returned {comp.numel()} rewards "
                        f"for a batch of {bs}."
                    )
                if not torch.isfinite(comp).all():
                    bad = (~torch.isfinite(comp)).nonzero(as_tuple=False).flatten().tolist()
                    raise RuntimeError(
                        f"T2AVCompositeScorer: inner scorer {name!r} returned non-finite rewards at indices {bad[:8]}"
                    )
                component_rewards[name] = comp.tolist()
                total = total + float(self.weights[name]) * comp

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

    @property
    def preferred_input_kind(self) -> str:
        return self.input_kind

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

    batch_size: int = 8
    device: str = "auto"
    # Forwarded to inner scorers that declare it (videopickscore). "first"
    # keeps the historical behaviour; "middle" avoids scoring a blank opening
    # frame on clips that fade or reveal in.
    frame_selection: str = "first"
    num_score_frames: int = 1
    frame_aggregation: str = "mean"
    topk_frames: int = 3
    all_frame_mean_weight: float = 0.25
    clap_model_id: Optional[str] = None
    imagebind_model_path: Optional[str] = None
    imagebind_mode: str = "audio_video"
    weights: Dict[str, float] = field(default_factory=lambda: {"videopickscore": 0.5, "clap": 0.5})
