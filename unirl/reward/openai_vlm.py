"""Driver-side asynchronous reward using an OpenAI-compatible vision judge."""

from __future__ import annotations

import base64
import io
import json
import math
import queue
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import requests
from PIL import Image

from unirl.config.require import require
from unirl.distributed.tensor import hydrate, map_tree
from unirl.reward.service import attach_reward_response, build_reward_request
from unirl.types.reward import RewardResponse
from unirl.types.sample import Sample

_DEFAULT_SYSTEM_PROMPT = """You are a strict image quality judge. Evaluate the generated image against the user's
text prompt. Consider prompt alignment, object and attribute correctness, composition, visual quality, and visible
artifacts. Return JSON only, exactly in this shape: {"score": <number from 0.0 to 1.0>, "reason": "<short reason>"}."""


@dataclass
class OpenAIVLMRewardSpec:
    """Configuration for the driver-side OpenAI-compatible vision judge."""

    base_url: str = "http://28.7.193.213:8000/v1"
    model: str = "zehan"
    batch_size: int = 1
    flush_timeout_ms: float = 10.0
    max_concurrency: int = 16
    timeout: float = 300.0
    max_retries: int = 3
    retry_delay: float = 1.0
    temperature: float = 0.0
    max_tokens: int = 128
    image_format: str = "JPEG"
    image_quality: int = 90
    score_key: str = "score"
    score_min: float = 0.0
    score_max: float = 1.0
    component_name: str = "zehan_qwen25vl_judge"
    system_prompt: str = _DEFAULT_SYSTEM_PROMPT
    api_key: str = ""

    def __post_init__(self) -> None:
        require(bool(self.base_url.strip()), "OpenAIVLMRewardSpec.base_url must be non-empty")
        require(bool(self.model.strip()), "OpenAIVLMRewardSpec.model must be non-empty")
        require(self.batch_size >= 1, f"batch_size must be >= 1, got {self.batch_size}")
        require(self.flush_timeout_ms >= 0, f"flush_timeout_ms must be >= 0, got {self.flush_timeout_ms}")
        require(self.max_concurrency >= 1, f"max_concurrency must be >= 1, got {self.max_concurrency}")
        require(self.timeout > 0, f"timeout must be positive, got {self.timeout}")
        require(self.max_retries >= 1, f"max_retries must be >= 1, got {self.max_retries}")
        require(self.score_max > self.score_min, "score_max must be greater than score_min")
        require(bool(self.component_name.strip()), "component_name must be non-empty")


def _encode_image_data_url(image: Image.Image, *, image_format: str, quality: int) -> str:
    image_format = image_format.upper()
    image = image.convert("RGB")
    buffer = io.BytesIO()
    kwargs: dict[str, Any] = {"format": image_format}
    if image_format == "JPEG":
        kwargs["quality"] = int(quality)
    image.save(buffer, **kwargs)
    mime = "image/jpeg" if image_format == "JPEG" else f"image/{image_format.lower()}"
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _message_text(raw: Any) -> str:
    choices = raw.get("choices") if isinstance(raw, dict) else None
    if not isinstance(choices, list) or not choices:
        raise ValueError(f"judge response has no choices: {raw!r}")
    message = choices[0].get("message", {})
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"]
        if texts:
            return "\n".join(texts)
    raise ValueError(f"judge response has no textual message content: {raw!r}")


def _parse_score(text: str, spec: OpenAIVLMRewardSpec) -> float:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)

    parsed: Any = None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if match is not None:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                parsed = None

    value: Any = parsed.get(spec.score_key) if isinstance(parsed, dict) else None
    if value is None:
        pattern = rf"[\"']?{re.escape(spec.score_key)}[\"']?\s*[:=]\s*(-?\d+(?:\.\d+)?)"
        match = re.search(pattern, stripped, flags=re.IGNORECASE)
        if match is not None:
            value = match.group(1)
    if value is None and re.fullmatch(r"-?\d+(?:\.\d+)?", stripped):
        value = stripped

    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"judge output does not contain numeric {spec.score_key!r}: {text!r}")
    try:
        score = float(value)
    except ValueError as exc:
        raise ValueError(f"judge output contains invalid score {value!r}: {text!r}") from exc
    if not math.isfinite(score):
        raise ValueError(f"judge returned non-finite score {score!r}")
    if not spec.score_min <= score <= spec.score_max:
        raise ValueError(
            f"judge score {score} outside configured range [{spec.score_min}, {spec.score_max}]: {text!r}"
        )
    return score


class OpenAIVLMRewardScorer:
    """Synchronous scoring core shared by the async driver batcher and eval."""

    preferred_input_kind = "image"

    def __init__(self, config: OpenAIVLMRewardSpec) -> None:
        self.config = config
        self._thread_local = threading.local()
        self._sessions: List[requests.Session] = []
        self._sessions_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=config.max_concurrency,
            thread_name_prefix="driver-vlm-http",
        )

    def _session(self) -> requests.Session:
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            session.trust_env = False
            if self.config.api_key:
                session.headers["Authorization"] = f"Bearer {self.config.api_key}"
            self._thread_local.session = session
            with self._sessions_lock:
                self._sessions.append(session)
        return session

    @property
    def _chat_url(self) -> str:
        return f"{self.config.base_url.rstrip('/')}/chat/completions"

    def is_available(self) -> bool:
        try:
            response = self._session().get(
                f"{self.config.base_url.rstrip('/')}/models",
                timeout=min(5.0, self.config.timeout),
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    def _score_one(self, prompt: str, image: Image.Image) -> float:
        image_url = _encode_image_data_url(
            image,
            image_format=self.config.image_format,
            quality=self.config.image_quality,
        )
        payload = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "messages": [
                {"role": "system", "content": self.config.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Target text prompt:\n{prompt}\n\nJudge the generated image."},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
        }
        last_error: Optional[BaseException] = None
        for attempt in range(self.config.max_retries):
            try:
                response = self._session().post(self._chat_url, json=payload, timeout=self.config.timeout)
                response.raise_for_status()
                return _parse_score(_message_text(response.json()), self.config)
            except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
                last_error = exc
                if attempt + 1 < self.config.max_retries:
                    time.sleep(self.config.retry_delay)
        raise RuntimeError(
            f"OpenAI VLM reward failed after {self.config.max_retries} attempt(s) at {self._chat_url}"
        ) from last_error

    def score_samples(self, samples: Sequence[Sample]) -> List[Sample]:
        requests_by_sample = [build_reward_request(sample, self.preferred_input_kind) for sample in samples]
        pending: List[tuple[int, Future[float]]] = []
        sample_sizes: List[int] = []
        for sample_index, request in enumerate(requests_by_sample):
            generated = map_tree(request.generated["image"], hydrate)
            images = [image.to_pil() for image in generated.to_list()]
            prompts = request.prompts
            if len(images) != len(prompts):
                raise RuntimeError(
                    f"driver reward image/prompt size mismatch for sample {sample_index}: "
                    f"{len(images)} != {len(prompts)}"
                )
            sample_sizes.append(len(images))
            pending.extend(
                (sample_index, self._executor.submit(self._score_one, prompt, image))
                for prompt, image in zip(prompts, images)
            )

        scores: List[List[float]] = [[] for _ in samples]
        for sample_index, future in pending:
            scores[sample_index].append(future.result())

        scored_samples = []
        for sample, values, expected in zip(samples, scores, sample_sizes):
            if len(values) != expected:
                raise RuntimeError(f"driver reward returned {len(values)} scores for expected batch {expected}")
            response = RewardResponse(
                rewards=values,
                component_rewards={self.config.component_name: list(values)},
                successes=[True] * expected,
                errors=[None] * expected,
            )
            scored_samples.append(attach_reward_response(sample, response, truncated_reward="keep"))
        return scored_samples

    def score_and_attach(self, sample: Sample) -> Sample:
        return self.score_samples([sample])[0]

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)
        with self._sessions_lock:
            sessions, self._sessions = self._sessions, []
        for session in sessions:
            session.close()


@dataclass
class _RewardJob:
    sample: Sample
    future: Future[Sample]


class AsyncRewardCall:
    """Release a rollout lane while its driver-side reward remains in flight."""

    def __init__(self, rollout_call: Any, batcher: "OpenAIVLMRewardBatcher") -> None:
        self._rollout_call = rollout_call
        self._batcher = batcher
        self._reward_future: Optional[Future[Sample]] = None
        self._lock = threading.Lock()

    def _start_if_ready(self, *, block: bool) -> bool:
        with self._lock:
            if self._reward_future is not None:
                return True
            if not block and not self._rollout_call.ready():
                return False
            sample = self._rollout_call.result()
            parts = getattr(sample, "parts", None)
            status = parts[-1].harness_status if parts else None
            if status == "suspended":
                # Quiesce owns suspended-trajectory carry/relaunch. Scoring it
                # here would fail on a partial frontier and destroy that state.
                self._reward_future = Future()
                self._reward_future.set_result(sample)
            else:
                self._reward_future = self._batcher.submit(sample)
            return True

    def ready(self) -> bool:
        # Ready releases RolloutPool's engine capacity. result() still waits for
        # reward, so collect sees only fully scored Samples.
        return self._start_if_ready(block=False)

    def result(self) -> Sample:
        self._start_if_ready(block=True)
        assert self._reward_future is not None
        return self._reward_future.result()


class OpenAIVLMRewardBatcher:
    """Driver queue that microbatches completed rollout Samples for VLM scoring."""

    def __init__(self, config: OpenAIVLMRewardSpec) -> None:
        self.config = config
        self.scorer = OpenAIVLMRewardScorer(config)
        self._queue: queue.Queue[Optional[_RewardJob]] = queue.Queue()
        self._closed = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name="driver-vlm-reward", daemon=True)
        self._thread.start()

    def chain(self, rollout_call: Any) -> AsyncRewardCall:
        return AsyncRewardCall(rollout_call, self)

    def submit(self, sample: Sample) -> Future[Sample]:
        with self._lock:
            if self._closed:
                raise RuntimeError("OpenAIVLMRewardBatcher is closed")
            future: Future[Sample] = Future()
            self._queue.put(_RewardJob(sample, future))
            return future

    def score_and_attach(self, sample: Sample) -> Sample:
        return self.submit(sample).result()

    def is_available(self) -> bool:
        return self.scorer.is_available()

    def _run(self) -> None:
        stop_after_batch = False
        while True:
            first = self._queue.get()
            if first is None:
                return
            jobs = [first]
            deadline = time.monotonic() + self.config.flush_timeout_ms / 1000.0
            while len(jobs) < self.config.batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = self._queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if item is None:
                    stop_after_batch = True
                    break
                jobs.append(item)

            try:
                results = self.scorer.score_samples([job.sample for job in jobs])
                for job, result in zip(jobs, results):
                    job.future.set_result(result)
            except BaseException as exc:
                for job in jobs:
                    job.future.set_exception(exc)
            if stop_after_batch:
                return

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._queue.put(None)
        self._thread.join()
        self.scorer.close()


__all__ = [
    "AsyncRewardCall",
    "OpenAIVLMRewardBatcher",
    "OpenAIVLMRewardScorer",
    "OpenAIVLMRewardSpec",
]
