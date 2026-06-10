"""Reward service: score one ``RolloutTrack`` against ``(RolloutReq, decoded)``.

Holds exactly one :class:`~unirl.reward.base.RewardBackend` — a local
in-process scorer or the remote RewardService HTTP client. Builds a
:class:`RewardRequest` from the track, scores it, and attaches the rewards back
to a copy of the track under DP-sharded distributed dispatch.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch

from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.distributed.group.remote import Remote
from unirl.types.reward import RewardRequest, RewardResponse
from unirl.types.rollout_req import PrimitiveValue, RolloutReq
from unirl.types.rollout_resp import RolloutTrack, _track_with_field
from unirl.types.sampling import get_ar_params, get_diffusion_params

from .base import RewardBackend

logger = logging.getLogger(__name__)

_KIND_TO_KEY = {"image": "image", "video": "video", "text": "text"}


def _normalize_prompt_metadata(
    *,
    prompt_metadata: Optional[List[Optional[Dict[str, Any]]]],
    sample_count: int,
    prompt_ids: Optional[List[str]] = None,
    samples_per_prompt: Optional[int] = None,
) -> Optional[List[Optional[Dict[str, Any]]]]:
    """Normalize prompt metadata to sample-aligned layout."""
    if not isinstance(prompt_metadata, list) or not prompt_metadata:
        return None

    if sample_count <= 0:
        return None

    if len(prompt_metadata) == sample_count:
        return list(prompt_metadata)

    if isinstance(prompt_ids, list) and len(prompt_ids) == sample_count:
        ordered_prompt_ids: List[str] = []
        seen: set[str] = set()
        for raw_prompt_id in prompt_ids:
            prompt_id = str(raw_prompt_id).strip()
            if not prompt_id or prompt_id in seen:
                continue
            seen.add(prompt_id)
            ordered_prompt_ids.append(prompt_id)
        if len(prompt_metadata) == len(ordered_prompt_ids):
            metadata_by_prompt_id = {
                prompt_id: prompt_metadata[idx] for idx, prompt_id in enumerate(ordered_prompt_ids)
            }
            return [metadata_by_prompt_id.get(str(raw_prompt_id).strip()) for raw_prompt_id in prompt_ids]

    raise ValueError(
        "Prompt metadata must already be sample-aligned or expand via explicit prompt_ids. "
        f"Got sample_count={sample_count}, metadata={len(prompt_metadata)}, "
        f"prompt_ids={len(prompt_ids) if isinstance(prompt_ids, list) else None}, "
        f"samples_per_prompt={samples_per_prompt}."
    )


def _build_request_for_track(
    *,
    reward_input_kind: str,
    samples_per_prompt: int,
    track: RolloutTrack,
    req_primitives: Dict[str, PrimitiveValue],
    prompt_ids: List[str],
    sample_ids: List[str],
    group_ids: List[str],
    prompt_metadata: Optional[List[Optional[Dict[str, Any]]]] = None,
) -> RewardRequest:
    """Assemble a ``RewardRequest`` from one track + its request-side primitives."""
    decoded = track.decoded
    if decoded is None:
        raise ValueError("Reward request assembly requires non-None track.decoded.")

    gen_key = _KIND_TO_KEY.get(reward_input_kind)
    if gen_key is None:
        raise ValueError(f"Unknown reward_input_kind={reward_input_kind!r}. Expected one of {sorted(_KIND_TO_KEY)}.")

    normalized_metadata = _normalize_prompt_metadata(
        prompt_metadata=prompt_metadata,
        sample_count=len(sample_ids),
        prompt_ids=prompt_ids,
        samples_per_prompt=samples_per_prompt,
    )

    return RewardRequest(
        primitives=dict(req_primitives),
        generated={gen_key: decoded},
        prompt_ids=list(prompt_ids),
        sample_ids=list(sample_ids),
        group_ids=list(group_ids),
        metadata=(
            normalized_metadata
            if normalized_metadata is not None and any(m is not None for m in normalized_metadata)
            else None
        ),
    )


class RewardService(Remote):
    """Actor-side reward entry: one backend, scores one track in place."""

    def __init__(
        self,
        backend: RewardBackend,
        truncated_reward: str = "zero",
        overlong_buffer_len: int = 4096,
        overlong_penalty_factor: float = 1.0,
    ) -> None:
        super().__init__()
        self.backend = backend
        # How to score AR generations that hit max_new_tokens (sglang finish=="length"):
        #   "zero" — force reward 0 on truncated traces (anti-ramble; the default).
        #   "keep" — keep the raw score on the partial text (= verl dapo reward manager
        #            with overlong_buffer.enable=False: no zeroing, no penalty).
        #   "soft" — verl DAPO overlong reward shaping (overlong_buffer.enable=True): a
        #            graded NEGATIVE penalty over the last `overlong_buffer_len` tokens
        #            before max_new_tokens — never a hard zero. Mirrors
        #            verl.workers.reward_manager.dapo: reward += min(-exceed/buf*factor, 0).
        self.truncated_reward = str(truncated_reward)
        self.overlong_buffer_len = int(overlong_buffer_len)
        self.overlong_penalty_factor = float(overlong_penalty_factor)
        if self.truncated_reward not in ("zero", "keep", "soft"):
            raise ValueError(f"truncated_reward must be zero|keep|soft, got {self.truncated_reward!r}")
        logger.info(
            "RewardService initialized with backend=%s, truncated_reward=%s",
            backend.get_model_name() or type(backend).__name__,
            self.truncated_reward,
        )

    @property
    def preferred_input_kind(self) -> str:
        """The decoded media kind the backend consumes (image/video/text)."""
        kind = str(getattr(self.backend, "preferred_input_kind", "") or "").strip().lower()
        if kind not in {"image", "video", "text"}:
            raise ValueError(
                f"Reward backend must expose preferred_input_kind as 'image', 'video', or 'text'. Got {kind!r}."
            )
        return kind

    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        return self.backend.compute_rewards(request)

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def score_and_attach(self, *, req: RolloutReq, track: RolloutTrack) -> RolloutTrack:
        """Score one track's decoded media and return a copy with rewards attached.

        Copies ``req.primitives`` (input context) into the reward request and
        pairs with ``track.decoded`` (generated output). For PE-joint tracks
        where the request has fewer samples than the track (N×M expansion),
        each primitive — and the per-prompt metadata — is replicated by the
        expansion factor.

        Returns a new :class:`RolloutTrack` with ``rewards`` and
        ``component_rewards`` populated; the input track is left unchanged so
        the result flows back through Handle dispatch (pytree_merge across DP
        shards) without relying on worker-local mutation.

        Fail-fast on per-sample failure flags so partial/corrupt rewards cannot
        silently enter advantage computation.
        """
        if track.rewards is not None:
            raise RuntimeError("Actor-side reward compute does not accept precomputed rewards on the track.")

        sample_ids = list(track.sample_ids)
        req_primitives: Dict[str, PrimitiveValue] = dict(req.primitives)

        # Determine the request-side batch size from any primitive.
        req_batch = 0
        for v in req_primitives.values():
            if v is not None:
                req_batch = len(v)
                break

        expanded_metadata: Optional[List[Optional[Dict[str, Any]]]] = None
        if req_batch > 0 and req_batch != len(sample_ids):
            # PE-joint expansion: req has P prompts, track has P*N*M samples.
            if len(sample_ids) % req_batch != 0:
                raise RuntimeError(
                    f"RewardService.score_and_attach: req batch {req_batch} != track.sample_ids "
                    f"count {len(sample_ids)} and not an integer multiple — sample alignment broken."
                )
            factor = len(sample_ids) // req_batch
            ar_params = get_ar_params(req.sampling_params)
            diff_params = get_diffusion_params(req.sampling_params)
            n = int(ar_params.samples_per_prompt) if ar_params is not None else 1
            m = int(diff_params.samples_per_prompt) if diff_params is not None else 1
            if factor != n * m:
                raise RuntimeError(
                    f"RewardService.score_and_attach: implicit expansion factor {factor} "
                    f"(track={len(sample_ids)} / req={req_batch}) does not match sampling_params "
                    f"N*M={n * m} (N={n}, M={m}). Sample alignment is ambiguous."
                )
            req_primitives = {k: v.repeat_interleave(factor) for k, v in req_primitives.items()}
            # Keep metadata aligned with primitives (one entry per sample).
            if req.metadata:
                expanded_metadata = [m for m in req.metadata for _ in range(factor)]

        final_metadata = (
            expanded_metadata if expanded_metadata is not None else (list(req.metadata) if req.metadata else None)
        )

        request = _build_request_for_track(
            reward_input_kind=self.preferred_input_kind,
            samples_per_prompt=max(1, len(sample_ids)),
            track=track,
            req_primitives=req_primitives,
            prompt_ids=[str(sid) for sid in sample_ids],
            sample_ids=sample_ids,
            group_ids=list(track.group_ids),
            prompt_metadata=final_metadata,
        )
        reward_response = self.compute_rewards(request)

        failed = [(i, e) for i, (ok, e) in enumerate(zip(reward_response.successes, reward_response.errors)) if not ok]
        if failed:
            raise RuntimeError(
                f"Reward computation flagged {len(failed)} of {len(reward_response.successes)} "
                f"sample(s) as failure. First few: {failed[:3]}"
            )

        rewards = torch.tensor(reward_response.rewards, dtype=torch.float32)

        # Length-based reward shaping for AR generations that hit max_new_tokens
        # (sglang finish == "length"). A non-terminating trace whose text happens to
        # contain a matching answer (e.g. a mid-reasoning \boxed{}) can teach the
        # model to ramble up to the token cap — a real failure mode at long
        # max_new_tokens. `truncated_reward` (see __init__) picks the policy:
        #   "zero" — force reward 0 on truncated traces (anti-ramble).
        #   "keep" — leave the raw score (= verl dapo, overlong disabled). No-op here.
        #   "soft" — verl DAPO graded overlong penalty (never a hard zero).
        # seg_lengths and rewards are shard-aligned (one entry per sample).
        ar_params = get_ar_params(req.sampling_params)
        if self.truncated_reward != "keep" and ar_params is not None and track.segment is not None:
            seg_lengths = getattr(track.segment, "lengths", None)
            if seg_lengths is not None and seg_lengths.numel() == rewards.numel():
                seg_lengths = seg_lengths.to(rewards.device).float()
                max_len = float(int(ar_params.max_new_tokens))
                if self.truncated_reward == "zero":
                    truncated = seg_lengths >= max_len
                    rewards = torch.where(truncated, torch.zeros_like(rewards), rewards)
                else:  # "soft": verl overlong shaping — graded negative penalty over the
                    # last overlong_buffer_len tokens before max_len, clamped to <= 0.
                    buf = float(self.overlong_buffer_len)
                    exceed = seg_lengths - (max_len - buf)
                    penalty = torch.clamp(-exceed / buf * self.overlong_penalty_factor, max=0.0)
                    rewards = rewards + penalty
            elif seg_lengths is not None:
                # Mismatched counts are expected when the AR segment is not 1:1 with
                # rewards (e.g. composed PE: N AR segments vs N*M rewards), so we skip
                # rather than crash. But in a pure-AR run a mismatch means the shaping
                # silently did nothing — log it so the skip is discoverable.
                logger.debug(
                    "RewardService: skipped AR truncation shaping (seg_lengths=%d != rewards=%d).",
                    seg_lengths.numel(),
                    rewards.numel(),
                )

        component_rewards = {
            str(name): torch.tensor(list(values or []), dtype=torch.float32)
            for name, values in dict(reward_response.component_rewards or {}).items()
        }
        track = _track_with_field(track, "rewards", rewards)
        return _track_with_field(track, "component_rewards", component_rewards)

    def is_available(self) -> bool:
        return self.backend.is_available()

    def offload(self) -> None:
        self.backend.offload()

    def onload(self) -> None:
        self.backend.onload()

    def dispose(self) -> None:
        self.backend.dispose()


__all__ = [
    "RewardService",
]
