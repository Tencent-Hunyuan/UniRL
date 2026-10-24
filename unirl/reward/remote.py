"""Remote reward backend: an HTTP client for the RewardService server."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import math
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import requests as http_requests
import torch
from PIL import Image

from unirl.config.require import require
from unirl.reward.base import PromptRewardComponentSpec, RewardBackend
from unirl.types.reward import RewardRequest, RewardResponse
from unirl.utils.media import tensor_frame_to_pil

logger = logging.getLogger(__name__)


def _encode_image_b64(
    image: Union[Image.Image, torch.Tensor],
    *,
    image_format: str,
    quality: int,
) -> str:
    """Encode an image to a base64 string for the RewardService wire format."""
    if isinstance(image, torch.Tensor):
        image = tensor_frame_to_pil(image)
    if image.mode != "RGB":
        image = image.convert("RGB")

    buf = io.BytesIO()
    save_kwargs: dict = {"format": image_format}
    if image_format.upper() == "JPEG":
        save_kwargs["quality"] = quality
    image.save(buf, **save_kwargs)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _encode_video_b64(video: torch.Tensor, *, fps: int) -> str:
    """Encode a video tensor ``(C, T, H, W)`` to a base64 mp4 string."""
    from diffusers.utils import export_to_video

    v = video.detach().cpu()
    if v.dim() == 5:
        v = v.squeeze(0)
    if v.dim() != 4:
        raise ValueError(f"Expected 4D (C, T, H, W) video tensor, got shape {tuple(v.shape)}.")

    if v.is_floating_point():
        v = (v.clamp(0.0, 1.0) * 255).byte()
    frames = [Image.fromarray(v[:, t, :, :].permute(1, 2, 0).numpy()) for t in range(v.shape[1])]

    tmp = tempfile.NamedTemporaryFile(prefix="reward_svc_", suffix=".mp4", delete=False)
    tmp.close()
    try:
        export_to_video(frames, tmp.name, fps=fps)
        with open(tmp.name, "rb") as f:
            video_bytes = f.read()
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    return base64.b64encode(video_bytes).decode("ascii")


def _wire_identity(
    request: RewardRequest,
    index: int,
    *,
    required_rewards: List[str],
    expected_scorer_version: Optional[str],
    media_fingerprint: str,
    prompt: str,
    metadata: Any,
) -> Dict[str, Any]:
    digest = hashlib.sha256()
    digest.update(media_fingerprint.encode("ascii"))
    digest.update(b"\0")
    digest.update(prompt.encode("utf-8"))
    digest.update(b"\0")
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8"))
    payload_fingerprint = digest.hexdigest()

    sample_id = request.sample_ids[index] if request.sample_ids and index < len(request.sample_ids) else None
    group_id = request.group_ids[index] if request.group_ids and index < len(request.group_ids) else None
    policy_version = metadata.get("policy_version") if isinstance(metadata, dict) else None
    request_id = str(sample_id or uuid.uuid4())
    digest_input = json.dumps(
        {
            "protocol": "1",
            "request_id": request_id,
            "required_rewards": required_rewards,
            "policy_version": policy_version,
            "scorer_version": expected_scorer_version,
            "payload_fingerprint": payload_fingerprint,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    raw_rank = os.environ.get("RANK")
    source_rank: Optional[int] = None
    if raw_rank is not None:
        try:
            source_rank = int(raw_rank)
        except ValueError:
            source_rank = None
    return {
        "request_id": request_id,
        "sample_id": sample_id,
        "group_id": group_id,
        "source_rank": source_rank,
        "policy_version": policy_version if isinstance(policy_version, int) else None,
        "scorer_version": expected_scorer_version,
        "idempotency_key": hashlib.sha256(digest_input.encode("utf-8")).hexdigest(),
    }


class RemoteRewardBackend(RewardBackend):
    """HTTP client backend for the remote RewardService ``POST /score`` endpoint."""

    def __init__(self, *, config: "RemoteRewardSpec", base_device: str) -> None:
        del base_device
        super().__init__(
            model_name="reward_service",
            batch_size=config.batch_size,
            timeout=config.timeout,
            prompt_source=config.prompt_source,
        )
        self.base_url = config.base_url.rstrip("/")
        self.required_rewards = list(config.required_rewards)
        self.reward_weights = {} if config.reward_weights is None else dict(config.reward_weights)
        self.max_retries = config.max_retries
        self.retry_delay = config.retry_delay
        self.request_batch_size = config.request_batch_size
        self.require_identity_echo = config.require_identity_echo
        self.expected_scorer_version = config.expected_scorer_version
        self.sub_metric_reduce = config.sub_metric_reduce
        self.image_format = config.image_format
        self.image_quality = config.image_quality
        self.raise_on_failure = config.raise_on_failure
        self.aggregation_method = config.aggregation_method
        self.video_fps = config.video_fps
        self.input_kind = config.input_kind

        self._session = http_requests.Session()
        self._session.trust_env = False

    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        """Convert a UniRL request, call the remote service, and parse the response."""
        start = time.time()
        bs = request.batch_size
        try:
            payload = (
                self._build_video_score_payload(request) if request.is_video else self._build_score_payload(request)
            )
            raw = self._post_score_requests(payload["requests"])
            return self._parse_score_response(raw, bs, time.time() - start)
        except Exception:
            if self.raise_on_failure:
                raise
            logger.exception("RemoteRewardBackend.compute_rewards failed (degraded mode)")
            return RewardResponse(
                rewards=[0.0] * bs,
                successes=[False] * bs,
                errors=["RemoteRewardBackend failure (see logs)"] * bs,
                compute_time=time.time() - start,
            )

    def dispose(self) -> None:
        """Close the HTTP session."""
        self._session.close()

    def _build_score_payload(self, request: RewardRequest) -> Dict[str, Any]:
        """Convert a UniRL ``RewardRequest`` into the RewardService ``ScoreRequest``."""
        images = [] if request.images is None else request.images
        prompts = self.prompts(request)
        metadata_list = request.metadata
        wire_requests: List[Dict[str, Any]] = []

        prim_image = request.conditioning.get("image")
        condition_images = (
            None if prim_image is None else [tensor_frame_to_pil(image.pixels) for image in prim_image.to_list()]
        )

        for idx in range(len(images)):
            prompt = prompts[idx] if idx < len(prompts) else ""
            image_b64 = _encode_image_b64(
                images[idx],
                image_format=self.image_format,
                quality=self.image_quality,
            )
            sample_metadata = metadata_list[idx] if metadata_list is not None and idx < len(metadata_list) else None

            if condition_images is not None and idx < len(condition_images):
                condition_b64 = _encode_image_b64(
                    condition_images[idx],
                    image_format=self.image_format,
                    quality=self.image_quality,
                )
                history = [
                    {"text": prompt, "image_b64": condition_b64},
                    {"text": prompt, "image_b64": image_b64},
                ]
                media_fingerprint = hashlib.sha256(f"{condition_b64}:{image_b64}".encode("ascii")).hexdigest()
            else:
                history = [{"text": prompt, "image_b64": image_b64}]
                media_fingerprint = hashlib.sha256(image_b64.encode("ascii")).hexdigest()

            identity = _wire_identity(
                request,
                idx,
                required_rewards=self.required_rewards,
                expected_scorer_version=self.expected_scorer_version,
                media_fingerprint=media_fingerprint,
                prompt=prompt,
                metadata=sample_metadata,
            )
            wire_requests.append(
                {
                    "history": history,
                    "required_rewards": list(self.required_rewards),
                    "metadata": sample_metadata,
                    **identity,
                }
            )

        return {"protocol_version": "1", "requests": wire_requests}

    def _build_video_score_payload(self, request: RewardRequest) -> Dict[str, Any]:
        """Convert a video ``RewardRequest`` into the RewardService ``ScoreRequest``."""
        videos = [] if request.videos is None else request.videos
        prompts = self.prompts(request)
        metadata_list = request.metadata
        wire_requests: List[Dict[str, Any]] = []

        for idx in range(len(videos)):
            prompt = prompts[idx] if idx < len(prompts) else ""
            video_b64 = _encode_video_b64(videos[idx], fps=self.video_fps)
            sample_metadata = metadata_list[idx] if metadata_list is not None and idx < len(metadata_list) else None
            identity = _wire_identity(
                request,
                idx,
                required_rewards=self.required_rewards,
                expected_scorer_version=self.expected_scorer_version,
                media_fingerprint=hashlib.sha256(video_b64.encode("ascii")).hexdigest(),
                prompt=prompt,
                metadata=sample_metadata,
            )
            wire_requests.append(
                {
                    "history": [{"text": prompt, "video_b64": video_b64}],
                    "required_rewards": list(self.required_rewards),
                    "metadata": sample_metadata,
                    **identity,
                }
            )

        return {"protocol_version": "1", "requests": wire_requests}

    def _post_score_requests(self, wire_requests: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not wire_requests:
            return {"protocol_version": "1", "results": [], "errors": []}
        request_batch_size = len(wire_requests) if self.request_batch_size is None else self.request_batch_size
        merged_results: List[Dict[str, Dict[str, float]]] = []
        merged_errors: List[Dict[str, str]] = []

        for start in range(0, len(wire_requests), request_batch_size):
            chunk = wire_requests[start : start + request_batch_size]
            raw = self._post_score({"protocol_version": "1", "requests": chunk})
            response_version = raw.get("protocol_version")
            if response_version is not None and response_version != "1":
                raise ValueError(f"RewardService protocol_version={response_version!r}, expected '1'")
            raw_results = raw.get("results")
            results = [] if raw_results is None else list(raw_results)
            raw_errors = raw.get("errors")
            errors = [] if raw_errors is None else list(raw_errors)
            raw_identities = raw.get("identities")
            identities = [] if raw_identities is None else list(raw_identities)
            if len(results) > len(chunk) or len(errors) > len(chunk):
                raise ValueError(
                    f"RewardService returned more rows than requested for chunk {start}: "
                    f"results={len(results)} errors={len(errors)} requested={len(chunk)}"
                )
            results.extend({} for _ in range(len(chunk) - len(results)))
            errors.extend({} for _ in range(len(chunk) - len(errors)))

            if identities:
                if len(identities) != len(chunk):
                    raise ValueError(
                        f"RewardService identity count {len(identities)} != requested chunk size {len(chunk)}"
                    )
                for expected, actual in zip(chunk, identities, strict=True):
                    self._validate_identity_echo(expected, actual)
            elif self.require_identity_echo:
                raise ValueError("RewardService response omitted required item identities")

            merged_results.extend(results)
            merged_errors.extend(errors)

        return {
            "protocol_version": "1",
            "results": merged_results,
            "errors": merged_errors,
        }

    def _validate_identity_echo(self, expected: Dict[str, Any], actual: Dict[str, Any]) -> None:
        for key in ("request_id", "sample_id", "group_id", "source_rank", "policy_version", "idempotency_key"):
            if actual.get(key) != expected.get(key):
                raise ValueError(
                    f"RewardService identity mismatch for {key}: expected {expected.get(key)!r}, "
                    f"got {actual.get(key)!r}"
                )
        if self.expected_scorer_version is not None and actual.get("scorer_version") != self.expected_scorer_version:
            raise ValueError(
                f"RewardService scorer_version={actual.get('scorer_version')!r} "
                f"!= expected {self.expected_scorer_version!r}"
            )

    def _post_score(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST to ``/score`` with retry logic."""
        url = f"{self.base_url}/score"
        last_exc: Optional[BaseException] = None

        for attempt in range(self.max_retries):
            try:
                resp = self._session.post(url, json=payload, timeout=self.timeout)
                resp.raise_for_status()
                return resp.json()
            except http_requests.exceptions.Timeout as e:
                last_exc = e
                logger.warning(
                    "RemoteRewardBackend: request timed out (attempt %d/%d)",
                    attempt + 1,
                    self.max_retries,
                )
            except http_requests.exceptions.RequestException as e:
                response = e.response
                if response is not None and 400 <= response.status_code < 500 and response.status_code != 429:
                    raise
                last_exc = e
                logger.warning(
                    "RemoteRewardBackend: %s (attempt %d/%d)",
                    e,
                    attempt + 1,
                    self.max_retries,
                )

            if attempt < self.max_retries - 1:
                time.sleep(self.retry_delay)

        raise RuntimeError(f"RemoteRewardBackend: failed after {self.max_retries} retries calling {url}") from last_exc

    def _parse_score_response(
        self,
        raw: Dict[str, Any],
        batch_size: int,
        compute_time: float,
    ) -> RewardResponse:
        """Convert the RewardService ``ScoreResponse`` JSON into a UniRL ``RewardResponse``."""
        results: List[Dict[str, Dict[str, float]]] = raw["results"]
        errors_list: List[Dict[str, str]] = raw["errors"]

        component_rewards: Dict[str, List[float]] = {name: [] for name in self.required_rewards}
        aggregated_rewards: List[float] = []
        successes: List[bool] = []
        sample_errors: List[Optional[str]] = []

        for i in range(batch_size):
            sample_result = results[i] if i < len(results) else {}
            sample_errors_dict = errors_list[i] if i < len(errors_list) else {}

            scores: List[float] = []
            weights: List[float] = []
            error_parts: List[str] = []

            for reward_name in self.required_rewards:
                if reward_name in sample_result:
                    sub_metrics = sample_result[reward_name]
                    non_finite = self._first_non_finite(sub_metrics)
                    if non_finite is not None:
                        # Reject non-finite scores before they poison group normalization.
                        metric_name, bad_value = non_finite
                        component_rewards[reward_name].append(0.0)
                        error_parts.append(
                            f"{reward_name}: non-finite value {bad_value!r} for sub-metric {metric_name!r}"
                        )
                        continue
                    score = self._reduce_sub_metrics(sub_metrics)
                    component_rewards[reward_name].append(score)
                    scores.append(score)
                    weights.append(self.reward_weights.get(reward_name, 1.0))
                else:
                    component_rewards[reward_name].append(0.0)
                    if reward_name in sample_errors_dict:
                        error_parts.append(f"{reward_name}: {sample_errors_dict[reward_name]}")
                    else:
                        error_parts.append(f"{reward_name}: missing from server response without error")

            if scores:
                if self.aggregation_method == "weighted_sum":
                    total_w = sum(weights)
                    agg = sum(s * w for s, w in zip(scores, weights)) / total_w if total_w > 0 else 0.0
                elif self.aggregation_method == "mean":
                    agg = sum(scores) / len(scores)
                elif self.aggregation_method == "min":
                    agg = min(scores)
                else:
                    agg = max(scores)
                aggregated_rewards.append(agg)
                successes.append(len(error_parts) == 0)
            else:
                aggregated_rewards.append(0.0)
                successes.append(False)

            sample_errors.append("; ".join(error_parts) if error_parts else None)

        return RewardResponse(
            rewards=aggregated_rewards,
            component_rewards=component_rewards,
            successes=successes,
            errors=sample_errors,
            compute_time=compute_time,
        )

    @staticmethod
    def _first_non_finite(sub_metrics: Dict[str, float]) -> Optional[Tuple[str, Any]]:
        """Return the first ``(name, value)`` whose value is not a finite number."""
        for name, value in sub_metrics.items():
            if (
                value is None
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                return name, value
        return None

    def _reduce_sub_metrics(self, sub_metrics: Dict[str, float]) -> float:
        """Collapse a reward's sub-metric dict into a single float."""
        if not sub_metrics:
            raise ValueError("RewardService returned an empty sub-metric mapping")
        values = [float(v) for v in sub_metrics.values()]
        if self.sub_metric_reduce == "first":
            return values[0]
        if self.sub_metric_reduce == "mean":
            return sum(values) / len(values)
        return max(values)


@dataclass
class RemoteRewardSpec(PromptRewardComponentSpec):
    """Typed config for the remote RewardService backend."""

    base_url: str = ""
    required_rewards: Tuple[str, ...] = ()
    reward_weights: Optional[Dict[str, float]] = None
    batch_size: int = 8
    timeout: float = 300.0
    max_retries: int = 3
    retry_delay: float = 1.0
    # Transport chunking is independent from scorer/model micro-batching.
    # None preserves the legacy one-POST-per-DP-shard behavior.
    request_batch_size: Optional[int] = None
    require_identity_echo: bool = False
    # Managed/direct single-scorer servers only; the multi-reward gateway rejects a pin.
    expected_scorer_version: Optional[str] = None
    sub_metric_reduce: str = "first"
    aggregation_method: str = "weighted_sum"
    image_format: str = "JPEG"
    image_quality: int = 95
    video_fps: int = 8
    input_kind: str = "image"
    raise_on_failure: bool = True

    def __post_init__(self) -> None:
        require(
            bool(str(self.base_url).strip()),
            "RemoteRewardSpec.base_url must be non-empty",
        )
        require(
            len(self.required_rewards) > 0,
            "RemoteRewardSpec.required_rewards must be non-empty",
        )
        require(
            self.max_retries >= 1,
            f"RemoteRewardSpec.max_retries must be >= 1; got {self.max_retries!r}",
        )
        require(
            self.retry_delay >= 0,
            f"RemoteRewardSpec.retry_delay must be >= 0; got {self.retry_delay!r}",
        )
        require(
            self.request_batch_size is None or self.request_batch_size >= 1,
            f"RemoteRewardSpec.request_batch_size must be None or >= 1; got {self.request_batch_size!r}",
        )
        require(
            self.sub_metric_reduce in {"first", "mean", "max"},
            f"RemoteRewardSpec.sub_metric_reduce must be one of first/mean/max; got {self.sub_metric_reduce!r}",
        )
        require(
            self.aggregation_method in {"weighted_sum", "mean", "min", "max"},
            f"RemoteRewardSpec.aggregation_method must be one of "
            f"weighted_sum/mean/min/max; got {self.aggregation_method!r}",
        )
        require(
            self.input_kind in {"image", "video"},
            f"RemoteRewardSpec.input_kind must be 'image' or 'video'; got {self.input_kind!r}",
        )


__all__ = [
    "RemoteRewardBackend",
    "RemoteRewardSpec",
]
