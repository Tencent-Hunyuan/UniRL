"""Reward service: score a response :class:`~unirl.types.sample.Sample` in place.

Holds exactly one :class:`~unirl.reward.base.RewardBackend` — a local in-process
scorer or the remote RewardService HTTP client. Builds a :class:`RewardRequest`
from the Sample's frontier Part (the generated output) plus its conditioning (the
input context), scores it, and returns a copy of the Sample with the rewards
attached to the frontier Part, under DP-sharded distributed dispatch.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch

from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.distributed.group.remote import Remote
from unirl.types.primitives import Images, Texts, primitive_modality_key
from unirl.types.reward import RewardRequest, RewardResponse
from unirl.types.sample import Primitive, Sample, _part_with_field
from unirl.types.sampling import ARSamplingParams
from unirl.types.segments import SegmentStatus

from .base import DifferentiableReward, RewardBackend

logger = logging.getLogger(__name__)


def _codec_statistics(tokens: torch.Tensor) -> Dict[str, Any]:
    values = [int(value) for value in tokens.detach().cpu().view(-1).tolist()]
    count = len(values)
    if count == 0:
        return {
            "num_codec_tokens": 0,
            "codec_unique_ratio": 0.0,
            "codec_max_run_fraction": 1.0,
            "codec_repetition_fraction": 1.0,
        }
    max_run = run = 1
    for previous, current in zip(values, values[1:]):
        run = run + 1 if current == previous else 1
        max_run = max(max_run, run)
    ngram_size = min(4, count)
    ngrams = [tuple(values[i : i + ngram_size]) for i in range(count - ngram_size + 1)]
    repeated = len(ngrams) - len(set(ngrams))
    return {
        "num_codec_tokens": count,
        "codec_unique_ratio": len(set(values)) / count,
        "codec_max_run_fraction": max_run / count,
        "codec_repetition_fraction": repeated / len(ngrams) if ngrams else 0.0,
        "last_codec_token_id": values[-1],
    }


def _frontier_rollout_metadata(frontier, index: int) -> Dict[str, Any]:
    rollout: Dict[str, Any] = {}
    status = frontier.status
    if status is None and frontier.segment is not None:
        status = getattr(frontier.segment, "status", None)
    if status is not None and index < int(status.numel()):
        status_value = int(status[index].item())
        rollout["segment_status"] = SegmentStatus(status_value).name.lower()
        rollout["has_eos"] = status_value == int(SegmentStatus.COMPLETED)
        rollout["decode_failure"] = status_value == int(SegmentStatus.ABORTED)

    segment = frontier.segment
    if segment is not None:
        tokens = getattr(segment, "tokens", None)
        cu = getattr(segment, "cu_seqlens", None)
        if tokens is not None and cu is not None and index + 1 < int(cu.numel()):
            start, end = int(cu[index].item()), int(cu[index + 1].item())
            sample_tokens = tokens[start:end]
            rollout.update(_codec_statistics(sample_tokens))
            behavior = frontier.conditions.get("behavior_sampling") if isinstance(frontier.conditions, dict) else None
            eos_id = behavior.get("eos_token_id") if isinstance(behavior, dict) else None
            if eos_id is not None:
                rollout["has_eos"] = bool(sample_tokens.numel() and int(sample_tokens[-1].item()) == int(eos_id))

    sampling_params = frontier.sampling_params
    if isinstance(sampling_params, ARSamplingParams):
        rollout["max_new_tokens"] = int(sampling_params.max_new_tokens)
    audio_meta = frontier.primitive_metadata.get("audio", {})
    if "decode_failure" in audio_meta:
        rollout["decode_failure"] = bool(audio_meta["decode_failure"])
    return rollout


def _build_reward_request(sample: Sample, preferred_input_kind: str) -> RewardRequest:
    """Assemble a :class:`RewardRequest` from a response ``Sample``.

    The frontier (last) Part is the generated output being scored; the input
    context is :meth:`Sample.conditioning` — each ancestor primitive keyed by its
    modality slot with the NEAREST ancestor winning, so a PE/recaption image
    scores against the rewrite and an it2i edit against the instruction (plain T2I
    has a single ancestor, so the choice is moot). Prompt metadata is the root's,
    aligned to the frontier (:meth:`Sample.root_metadata`). Everything is already
    row-aligned to the frontier, so there is no request/track expansion to
    reconcile.
    """
    frontier = sample.parts[-1]
    primitives: Dict[str, Primitive] = {}
    for prim in sample.conditioning():
        primitives[primitive_modality_key(prim)] = prim  # nearest ancestor wins (last)

    if preferred_input_kind not in frontier.primitives:
        raise ValueError(
            f"Reward backend consumes {preferred_input_kind!r} but the frontier Part generated "
            f"{sorted(frontier.primitives)!r}; check the recipe's reward/model pairing."
        )

    root_metadata = sample.root_metadata(-1)
    frontier_metadata: List[Dict[str, Any]] = list(frontier.metadata or [])
    metadata: List[Dict[str, Any]] = []
    for index in range(frontier.batch_size):
        merged = dict(root_metadata[index] or {}) if index < len(root_metadata) else {}
        if index < len(frontier_metadata):
            merged.update(frontier_metadata[index] or {})
        rollout = dict(merged.get("rollout") or {})
        rollout.update(_frontier_rollout_metadata(frontier, index))
        merged["rollout"] = rollout
        metadata.append(merged)
    generated = dict(frontier.primitives)
    audio_sample_rate: Optional[int] = None
    audio_metadata = frontier.primitive_metadata.get("audio", {})
    if "audio" in generated and audio_metadata.get("sample_rate") is not None:
        audio_sample_rate = int(audio_metadata["sample_rate"])
    return RewardRequest(
        primitives=primitives,
        generated=generated,
        audio_sample_rate=audio_sample_rate,
        prompt_ids=[str(sid) for sid in frontier.sample_ids],
        sample_ids=list(frontier.sample_ids),
        group_ids=list(frontier.group_ids),
        metadata=metadata,
    )


class RewardService(Remote):
    """Actor-side reward entry: one backend, scores a Sample's frontier Part in place."""

    def __init__(
        self,
        backend: RewardBackend,
        truncated_reward: str = "zero",
        overlong_buffer_len: int = 4096,
        overlong_penalty_factor: float = 1.0,
    ) -> None:
        super().__init__()
        if not bool(getattr(backend, "training_eligible", True)):
            raise ValueError(
                f"{type(backend).__name__} is diagnostic-only (dry_run/unavailable) and cannot be "
                "registered with the training RewardService."
            )
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
        if kind not in {"image", "video", "text", "audio"}:
            raise ValueError(
                f"Reward backend must expose preferred_input_kind as 'image', 'video', 'text', or 'audio'. Got {kind!r}."
            )
        return kind

    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        return self.backend.compute_rewards(request)

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def score_differentiable(self, *, images: Images, prompts: Texts) -> torch.Tensor:
        """ReFL scoring: score grad-carrying ``images`` (pixels ``[B, C, H, W]`` in
        ``[0, 1]``) against ``prompts`` and return a ``[B]`` reward tensor with
        ``grad_fn`` intact.

        Deliberately bypasses :meth:`score_and_attach` / ``RewardRequest.images``
        (those go through ``tensor_frame_to_pil`` + ``torch.tensor(...)``, which
        detach). Under ``enable_grad()`` the framework marks ``images.pixels`` as a
        grad leaf and chains the returned reward's grad back to it. The backend must
        satisfy the :class:`~unirl.reward.base.DifferentiableReward` Protocol.
        """
        if not isinstance(self.backend, DifferentiableReward):
            raise TypeError(
                f"RewardService.score_differentiable: backend "
                f"{type(self.backend).__name__} is not a DifferentiableReward — ReFL "
                f"needs a differentiable in-process reward (e.g. pickscore/clip/hpsv2)."
            )
        return self.backend.compute_rewards_differentiable(images.pixels, list(prompts.texts))

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def score_and_attach(self, sample: Sample) -> Sample:
        """Score the frontier (last) Part's generated media and return the updated Sample.

        The frontier is the generated output; its :meth:`Sample.conditioning` is the
        input context and :meth:`Sample.root_metadata` the per-sample spec — both
        already row-aligned to the frontier, so there is no request/track expansion
        to reconcile. DP_SCATTER shards the whole Sample by prompt-tree
        (:meth:`Sample.slice`), keeping each shard's conditioning and frontier
        co-resident.

        Returns a new :class:`~unirl.types.sample.Sample` with ``rewards`` and
        ``component_rewards`` on the frontier Part; the other parts are untouched
        (the trainer credit-assigns upward via :meth:`Sample.propagate_rewards`).
        Fail-fast on per-sample failure flags so partial/corrupt rewards cannot
        silently enter advantage computation.
        """
        frontier = sample.parts[-1]
        if frontier.rewards is not None:
            raise RuntimeError("Actor-side reward compute does not accept precomputed rewards on the frontier Part.")
        if not frontier.primitives:
            raise ValueError("RewardService.score_and_attach: frontier Part has no generated primitives to score.")

        request = _build_reward_request(sample, self.preferred_input_kind)
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
        # Only applies when the SCORED frontier is itself an AR generation, where
        # its segment lengths are 1:1 with the rewards.
        sp = frontier.sampling_params
        if self.truncated_reward != "keep" and isinstance(sp, ARSamplingParams) and frontier.segment is not None:
            seg_lengths = getattr(frontier.segment, "lengths", None)
            if seg_lengths is not None and seg_lengths.numel() == rewards.numel():
                seg_lengths = seg_lengths.to(rewards.device).float()
                max_len = float(int(sp.max_new_tokens))
                if self.truncated_reward == "zero":
                    truncated = seg_lengths >= max_len
                    rewards = torch.where(truncated, torch.zeros_like(rewards), rewards)
                else:  # "soft": verl overlong shaping — graded negative penalty over the
                    # last overlong_buffer_len tokens before max_len, clamped to <= 0.
                    buf = float(self.overlong_buffer_len)
                    exceed = seg_lengths - (max_len - buf)
                    penalty = torch.clamp(-exceed / buf * self.overlong_penalty_factor, max=0.0)
                    rewards = rewards + penalty

        component_rewards = {
            str(name): torch.tensor(list(values or []), dtype=torch.float32)
            for name, values in dict(reward_response.component_rewards or {}).items()
        }
        scored = _part_with_field(frontier, "rewards", rewards)
        scored = _part_with_field(scored, "component_rewards", component_rewards)
        return sample.with_parts([*sample.parts[:-1], scored])

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
