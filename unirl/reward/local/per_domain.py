"""Per-domain reward dispatch: route each item to its domain's inner scorer.

Multi-domain training (e.g. DiffusionOPD's per-teacher datasets) needs a
different scorer per prompt domain — OCR reads quoted target text, GenEval
wants ``include``/``exclude`` specs, PickScore is generic — and a scorer must
never see items outside its domain. :class:`PerDomainRewardScorer` groups the
request rows by ``metadata["domain"]`` (the tag ``MultiDomainRLDataSource``
stamps), scores each group with the mapped inner backend, and scatters the
results back in place.

The inner scorers arrive fully constructed (Hydra recursively instantiates
each ``_target_`` in ``PerDomainSpec.scorers``), so any :class:`RewardBackend`
works — local or :class:`unirl.reward.remote.RemoteRewardBackend` — without
this file knowing registry names.

Monitoring conventions: per-domain scores surface as
``component_rewards["<domain>"]`` with ``NaN`` on rows outside the domain, so
wandb draws one curve per domain that only ticks on its own iterations.

Failure conventions: rows with a missing/unmapped domain are reported as
per-item failures (the reward service decides whether that is fatal); an inner
scorer that *raises* propagates — this combinator never swallows a traceback
into a per-item error string.
"""

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
    """Re-index one request field along the batch dim.

    Every primitive container is a :class:`Batch` (``Texts`` / ``Images`` /
    ``Videos`` / ``Audios``), so ``Batch.select`` handles CONCAT and packed
    varlen fields uniformly. Anything else is refused loudly — a silently
    unsliced field would misalign rows inside the inner scorer.
    """
    if value is None:
        return None
    if isinstance(value, Batch):
        return value.select(indices)
    raise TypeError(
        f"PerDomainRewardScorer: cannot row-select request field of type {type(value).__name__}; "
        "expected a Batch primitive (Texts/Images/Videos/Audios)."
    )


def _select_request(request: RewardRequest, indices: List[int]) -> RewardRequest:
    """A fresh ``RewardRequest`` holding only the given rows."""
    idx = torch.tensor(indices, dtype=torch.long)

    def _pick(x: Optional[List[Any]]) -> Optional[List[Any]]:
        return None if x is None else [x[i] for i in indices]

    return RewardRequest(
        primitives={k: _select_rows(v, idx) for k, v in request.primitives.items()},
        generated={k: _select_rows(v, idx) for k, v in request.generated.items()},
        metadata=_pick(request.metadata),
        prompt_ids=_pick(request.prompt_ids),
        sample_ids=_pick(request.sample_ids),
        group_ids=_pick(request.group_ids),
        reward_types=list(request.reward_types),
        return_components=request.return_components,
        audio_sample_rate=request.audio_sample_rate,
    )


class PerDomainRewardScorer(RewardBackend):
    """Dispatch each request row to the inner scorer of its metadata domain."""

    input_kind = "image"

    def __init__(self, *, config: "PerDomainSpec", base_device: str) -> None:
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
        metadata = request.metadata or [None] * bs
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
            resp = self._scorers[domain].compute_rewards(_select_request(request, indices))
            if len(resp.rewards) != len(indices):
                raise RuntimeError(
                    f"PerDomainRewardScorer: scorer for domain {domain!r} returned "
                    f"{len(resp.rewards)} rewards for {len(indices)} items."
                )
            inner_successes = resp.successes or [True] * len(indices)
            inner_errors = resp.errors or [None] * len(indices)
            for local_i, global_i in enumerate(indices):
                score = float(resp.rewards[local_i])
                rewards[global_i] = score
                component_rewards[domain][global_i] = score
                successes[global_i] = bool(inner_successes[local_i])
                errors[global_i] = inner_errors[local_i]

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
class PerDomainSpec(BaseRewardComponentSpec):
    """Typed config for :class:`PerDomainRewardScorer`.

    ``scorers`` maps a domain tag (the value the data source writes to
    ``metadata["domain"]``) to a fully built :class:`RewardBackend`; in yaml
    each entry carries its own ``_target_`` and Hydra instantiates it
    recursively before this spec is constructed. Example::

        config:
          _target_: unirl.reward.local.per_domain.PerDomainSpec
          scorers:
            pickscore:
              _target_: unirl.reward.local.pickscore.PickScoreRewardScorer
              base_device: cuda
              config: {_target_: unirl.reward.local.pickscore.PickScoreSpec}
            geneval:
              _target_: unirl.reward.remote.RemoteRewardBackend
              base_device: cpu
              config:
                _target_: unirl.reward.remote.RemoteRewardSpec
                base_url: ${oc.env:REWARD_SERVICE_URL}
                required_rewards: [geneval]
                reward_weights: {geneval: 1.0}
    """

    batch_size: int = 8
    device: str = "auto"
    scorers: Dict[str, RewardBackend] = field(default_factory=dict)


__all__ = ["PerDomainRewardScorer", "PerDomainSpec"]
