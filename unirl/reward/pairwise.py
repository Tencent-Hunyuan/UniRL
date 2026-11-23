"""Disjoint within-prompt comparisons; see the reward README for grouping and judge contracts."""

from __future__ import annotations

import math
import os
import random
import time
from abc import ABC, abstractmethod

import requests

from unirl.reward.base import PromptVideoReward, RewardBackend
from unirl.reward.remote import _encode_image_b64
from unirl.types.reward import DEFAULT_PROMPT_SOURCE, PromptSource, RewardRequest, RewardResponse
from unirl.types.sample_id import parent_id


class PairwiseJudge(ABC):
    """Consume adjacent candidate pairs and return one left-candidate label in {0, 0.5, 1} per pair."""

    input_kind = "image"
    prompt_source: PromptSource = DEFAULT_PROMPT_SOURCE

    @abstractmethod
    def score_pairs(self, request: RewardRequest) -> list[float]:
        """Compare rows (0, 1), (2, 3), ... jointly; invalid or failed judgments must raise."""

    def load(self) -> None:
        pass

    def offload(self) -> None:
        pass

    def onload(self) -> None:
        pass

    def dispose(self) -> None:
        pass


class PairwiseRewardBackend(RewardBackend):
    """Assign complementary comparison labels to complete groups, preserving the input row order."""

    requires_complete_groups = True
    supports_truncation_penalty = False

    def __init__(self, judge: PairwiseJudge, group_size: int, batch_size: int = 8, seed: int = 0) -> None:
        if isinstance(group_size, bool) or not isinstance(group_size, int) or group_size < 2 or group_size % 2:
            raise ValueError("PairwiseRewardBackend.group_size must be an even integer >= 2.")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("PairwiseRewardBackend.batch_size must be a positive integer (pairs per call).")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("PairwiseRewardBackend.seed must be a nonnegative integer.")
        super().__init__(model_name="pairwise", batch_size=batch_size, prompt_source=judge.prompt_source)
        self.judge = judge
        self.input_kind = judge.input_kind
        self.group_size = group_size
        self.seed = seed

    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        started = time.perf_counter()
        ids = request.sample_ids
        if ids is None or len(ids) != request.batch_size or len(set(ids)) != len(ids):
            raise ValueError("Pairwise rewards require unique, row-aligned sample_ids.")
        prompts = self.prompts(request)
        if len(prompts) != len(ids):
            raise ValueError("Pairwise rewards require a prompt for every candidate.")
        groups: dict[str, list[int]] = {}
        for row, sid in enumerate(ids):
            group = parent_id(sid)
            if group is None:
                raise ValueError(f"Pairwise candidate {sid!r} has no parent prompt lineage.")
            if request.group_ids is not None and request.group_ids[row] != group:
                raise ValueError(f"Pairwise group_ids disagree with sample_ids at row {row}.")
            groups.setdefault(group, []).append(row)
        pairs: list[tuple[int, int]] = []
        for group, rows in groups.items():
            if len(rows) != self.group_size:
                raise ValueError(
                    f"Pairwise prompt {group!r} has {len(rows)} candidates; expected complete group of {self.group_size}."
                )
            if len({prompts[row] for row in rows}) != 1:
                raise ValueError(f"Pairwise prompt {group!r} contains different scoring prompts.")
            rows.sort(key=lambda row: ids[row])
            random.Random(f"{self.seed}:{group}").shuffle(rows)
            pairs.extend(zip(rows[::2], rows[1::2]))
        rewards = [0.5] * len(ids)
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            labels = self.judge.score_pairs(request.select([row for pair in batch for row in pair]))
            if len(labels) != len(batch) or any(label not in (0.0, 0.5, 1.0) for label in labels):
                raise ValueError("Pairwise judge must return exactly one label in {0, 0.5, 1} per pair.")
            for (left, right), label in zip(batch, labels):
                rewards[left], rewards[right] = float(label), 1.0 - float(label)
        return RewardResponse(
            rewards=rewards,
            has_pairwise_labels=True,
            successes=[True] * len(rewards),
            errors=[None] * len(rewards),
            compute_time=time.perf_counter() - started,
        )

    def load(self) -> None:
        self.judge.load()

    def covers_prompt_video(self) -> bool:
        return not isinstance(self.judge, PromptVideoReward) or self.judge.covers_prompt_video()

    def offload(self) -> None:
        self.judge.offload()

    def onload(self) -> None:
        self.judge.onload()

    def dispose(self) -> None:
        self.judge.dispose()


class ScalarComparisonJudge(PairwiseJudge):
    """Compare scores from an independent pointwise reward backend."""

    def __init__(self, backend: RewardBackend) -> None:
        if not isinstance(backend, RewardBackend) or backend.requires_complete_groups:
            raise ValueError("ScalarComparisonJudge requires an independent per-candidate scalar backend.")
        if isinstance(backend.batch_size, bool) or not isinstance(backend.batch_size, int) or backend.batch_size < 1:
            raise ValueError("Scalar comparison backend.batch_size must be a positive integer.")
        self.backend = backend
        self.input_kind = backend.preferred_input_kind
        self.prompt_source = backend.prompt_source

    def score_pairs(self, request: RewardRequest) -> list[float]:
        if request.batch_size % 2:
            raise ValueError("ScalarComparisonJudge requires adjacent candidate pairs.")
        scores: list[float] = []
        limit = self.backend.batch_size
        for start in range(0, request.batch_size, limit):
            rows = list(range(start, min(start + limit, request.batch_size)))
            response = self.backend.compute_rewards(request.select(rows))
            if (
                len(response.rewards) != len(rows)
                or len(response.successes) != len(rows)
                or not all(response.successes)
                or not all(math.isfinite(value) for value in response.rewards)
            ):
                raise ValueError("Scalar comparison received missing, failed, or non-finite rewards.")
            scores.extend(response.rewards)
        return [0.5 if left == right else float(left > right) for left, right in zip(scores[::2], scores[1::2])]

    def load(self) -> None:
        self.backend.load()

    def covers_prompt_video(self) -> bool:
        return not isinstance(self.backend, PromptVideoReward) or self.backend.covers_prompt_video()

    def offload(self) -> None:
        self.backend.offload()

    def onload(self) -> None:
        self.backend.onload()

    def dispose(self) -> None:
        self.backend.dispose()


class PairwiseImageJudge(PairwiseJudge):
    """Compare two generated images jointly through an OpenAI-compatible multimodal chat endpoint."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        timeout: float = 60.0,
        api_key_env: str = "JUDGE_API_KEY",
        prompt_source: PromptSource = DEFAULT_PROMPT_SOURCE,
    ) -> None:
        if (
            not endpoint
            or not model
            or isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise ValueError("PairwiseImageJudge requires endpoint, model, and a finite positive timeout.")
        self.endpoint = endpoint
        self.model = model
        self.timeout = float(timeout)
        self.api_key_env = api_key_env
        self.prompt_source = prompt_source
        self._session = requests.Session()

    def dispose(self) -> None:
        self._session.close()

    def score_pairs(self, request: RewardRequest) -> list[float]:
        images = request.images
        prompt = request.original_prompt if self.prompt_source == "original" else request.generation_prompt
        if images is None or prompt is None or len(images) % 2:
            raise ValueError("PairwiseImageJudge requires image pairs and their prompts.")
        if request.conditioning:
            raise ValueError(
                "PairwiseImageJudge currently supports text-to-image comparisons without conditioning media."
            )
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get(self.api_key_env, "")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        labels = []
        for left in range(0, len(images), 2):
            content = [
                {
                    "type": "text",
                    "text": (
                        "Compare image A and image B for the following prompt. Consider prompt alignment, visual quality, "
                        "and artifacts. Reply with exactly A if A is better, B if B is better, or TIE if equally good.\n"
                        f"Prompt: {prompt.texts[left]}"
                    ),
                }
            ]
            for label, image in zip(("A", "B"), images[left : left + 2]):
                url = "data:image/png;base64," + _encode_image_b64(image, image_format="PNG", quality=100)
                content.extend(
                    [{"type": "text", "text": f"Image {label}:"}, {"type": "image_url", "image_url": {"url": url}}]
                )
            with self._session.post(
                self.endpoint,
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": content}],
                    "temperature": 0.0,
                    "max_tokens": 16,
                },
                headers=headers,
                timeout=self.timeout,
            ) as response:
                response.raise_for_status()
                verdict = response.json()["choices"][0]["message"]["content"].strip().upper()
            if verdict not in {"A", "B", "TIE"}:
                raise ValueError("Pairwise image judge must reply with exactly A, B, or TIE.")
            labels.append({"A": 1.0, "B": 0.0, "TIE": 0.5}[verdict])
        return labels


__all__ = ["PairwiseImageJudge", "PairwiseJudge", "PairwiseRewardBackend", "ScalarComparisonJudge"]
