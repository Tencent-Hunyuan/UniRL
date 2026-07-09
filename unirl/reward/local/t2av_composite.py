"""T2AV composite reward — weighted blend of video + audio scorers.

For LTX-2.3 text-to-audio-video, a single track carries both the decoded video
(``request.generated["video"]``) and the jointly-generated audio
(``request.generated["audio"]``, injected by the reward service from
``track.decoded_audio``). This composite runs several inner scorers over the
SAME request and returns their weighted sum, exposing each as a component.

Mirrors ``VideoRewardScorer``'s composite pattern: inner scorers are resolved
from the built-in registry by name, so any registered scorer (e.g.
``videopickscore`` for video quality, ``clap`` for audio-text alignment,
``imagebind`` for audio-video alignment) can be blended.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List

from unirl.reward.base import BaseRewardComponentSpec, RewardBackend
from unirl.types.reward import RewardRequest, RewardResponse

from .registry import resolve_builtin_reward_scorer_class, resolve_builtin_reward_spec_class


class T2AVCompositeScorer(RewardBackend):
    """Weighted blend of inner reward scorers for T2AV (video + audio).

    ``input_kind = "video"``: the track routes through the video path; audio is
    the parallel side-channel the reward service injects. Each inner scorer
    reads whichever modality it needs off the shared request.
    """

    input_kind = "video"

    def __init__(self, *, config: "T2AVCompositeSpec", base_device: str) -> None:
        super().__init__(model_name="t2av_composite", batch_size=config.batch_size)
        self.weights: Dict[str, float] = dict(config.weights or {})
        if not self.weights:
            raise ValueError("T2AVCompositeScorer requires a non-empty `weights` dict (scorer_name -> weight).")

        self._scorers: Dict[str, RewardBackend] = {}
        for name in self.weights:
            inner_cls = resolve_builtin_reward_scorer_class(name)
            inner_spec_cls = resolve_builtin_reward_spec_class(name)
            inner_spec = inner_spec_cls()
            # Propagate batch_size/device to inner specs that accept them.
            import dataclasses

            overrides = {f: getattr(config, f) for f in ("device", "batch_size") if hasattr(inner_spec, f)}
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
                comp = torch.tensor(list(resp.rewards), dtype=torch.float32)
                if comp.numel() != bs:
                    raise RuntimeError(
                        f"T2AVCompositeScorer: inner scorer {name!r} returned {comp.numel()} rewards "
                        f"for a batch of {bs}."
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
    """Typed config for the T2AV composite reward.

    ``weights`` maps inner scorer canonical names (e.g. ``videopickscore``,
    ``clap``, ``imagebind``) to blend weights. Each named scorer is built from
    its default Spec with ``device``/``batch_size`` propagated.
    """

    batch_size: int = 8
    device: str = "auto"
    weights: Dict[str, float] = field(default_factory=lambda: {"videopickscore": 0.5, "clap": 0.5})
