"""Per-domain reward dispatch: route each item to its domain's inner scorer."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from unirl.distributed.tensor.batch import Batch
from unirl.reward.base import BaseRewardComponentSpec, RewardBackend
from unirl.types.reward import RewardRequest, RewardResponse

DOMAIN_KEY = "domain"


def _select_rows(value: Any, indices: torch.Tensor) -> Any:
    """Re-index one request field along the batch dim."""
    if value is None:
        return None
    if isinstance(value, Batch):
        return value.select(indices)
    raise TypeError(
        f"PerDomainRewardScorer: cannot row-select request field of type {type(value).__name__}; "
        "expected a Batch primitive (Texts/Images/Videos/Audios)."
    )


class PerDomainRewardScorer(RewardBackend):
    """Dispatch each request row to the inner scorer of its metadata domain."""

    input_kind = "image"

    def __init__(self, *, config: "PerDomainSpec", base_device: str) -> None:
        del base_device
        super().__init__(model_name="per_domain", batch_size=config.batch_size)
        if not config.scorers:
            raise ValueError("PerDomainRewardScorer requires a non-empty `scorers` mapping (domain -> RewardBackend).")
        for domain, scorer in config.scorers.items():
            if not isinstance(scorer, RewardBackend):
                raise TypeError(
                    f"PerDomainRewardScorer: scorers[{domain!r}] is {type(scorer).__name__}, not a "
                    "RewardBackend — each entry needs its own `_target_` for Hydra to instantiate."
                )
        self._scorers: Dict[str, RewardBackend] = dict(config.scorers)

    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        start = time.time()
        bs = request.batch_size

        groups: Dict[str, List[int]] = {}
        unmapped: List[int] = []
        metadata = [None] * bs if request.metadata is None else request.metadata
        for i in range(bs):
            md = metadata[i] if i < len(metadata) else None
            domain = md.get(DOMAIN_KEY) if isinstance(md, dict) else None
            if domain is None or str(domain) not in self._scorers:
                unmapped.append(i)
            else:
                groups.setdefault(str(domain), []).append(i)

        rewards = [0.0] * bs
        successes = [True] * bs
        errors: List[Optional[str]] = [None] * bs
        component_rewards: Dict[str, List[float]] = {d: [float("nan")] * bs for d in self._scorers}

        for domain, indices in groups.items():
            idx = torch.tensor(indices, dtype=torch.long)
            sub_request = RewardRequest(
                generated={k: _select_rows(v, idx) for k, v in request.generated.items()},
                conditioning={k: _select_rows(v, idx) for k, v in request.conditioning.items()},
                original_prompt=_select_rows(request.original_prompt, idx),
                generation_prompt=_select_rows(request.generation_prompt, idx),
                metadata=None if request.metadata is None else [request.metadata[i] for i in indices],
                sample_ids=None if request.sample_ids is None else [request.sample_ids[i] for i in indices],
                group_ids=None if request.group_ids is None else [request.group_ids[i] for i in indices],
                audio_sample_rate=request.audio_sample_rate,
            )
            resp = self._scorers[domain].compute_rewards(sub_request)
            if len(resp.rewards) != len(indices):
                raise RuntimeError(
                    f"PerDomainRewardScorer: scorer for domain {domain!r} returned "
                    f"{len(resp.rewards)} rewards for {len(indices)} items."
                )
            for local_i, global_i in enumerate(indices):
                score = float(resp.rewards[local_i])
                rewards[global_i] = score
                component_rewards[domain][global_i] = score
                successes[global_i] = bool(resp.successes[local_i])
                errors[global_i] = resp.errors[local_i]

        for i in unmapped:
            successes[i] = False
            md = metadata[i] if i < len(metadata) else None
            tag = md.get(DOMAIN_KEY) if isinstance(md, dict) else None
            errors[i] = (
                f"PerDomainRewardScorer: metadata[{DOMAIN_KEY!r}]={tag!r} has no configured scorer "
                f"(configured: {sorted(self._scorers)})."
            )

        return RewardResponse(
            rewards=rewards,
            component_rewards=component_rewards,
            successes=successes,
            errors=errors,
            compute_time=time.time() - start,
        )

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
class PerDomainSpec(BaseRewardComponentSpec):
    """Typed config for :class:`PerDomainRewardScorer`."""

    batch_size: int = 8
    device: str = "auto"
    scorers: Dict[str, RewardBackend] = field(default_factory=dict)


__all__ = ["PerDomainRewardScorer", "PerDomainSpec"]
