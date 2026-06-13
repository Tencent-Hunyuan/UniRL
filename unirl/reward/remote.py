"""Remote reward backend: an HTTP client for the RewardService server.

Bridges the UniRL reward interface (flat images + prompts) with the
RewardService wire format (history turns + required_rewards). One client
handles *all* requested reward models in a single HTTP round trip because the
server multiplexes them via ``required_rewards``.

Configured as the backend on :class:`~unirl.reward.service.RewardService`::

    reward:
      _target_: unirl.reward.service.RewardService
      backend:
        _target_: unirl.reward.remote.RemoteRewardBackend
        base_device: cpu
        config:
          _target_: unirl.reward.remote.RemoteRewardSpec
          base_url: http://reward-server:8080
          required_rewards: [hpsv2, clip]
          reward_weights: {hpsv2: 0.6, clip: 0.4}
"""

from __future__ import annotations

import base64
import io
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import requests as http_requests
import torch
from PIL import Image

from unirl.config.require import require
from unirl.reward.base import BaseRewardComponentSpec, RewardBackend
from unirl.types.reward import RewardRequest, RewardResponse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pil_from_tensor(tensor: torch.Tensor) -> Image.Image:
    """Convert a CHW float or uint8 tensor to a PIL RGB image.

    Float tensors are clamped to ``[0, 1]`` (the producer-side contract on
    ``RolloutSamples.decoded_images``); ``to_pil_image`` then handles the
    uint8 conversion. Always moves to CPU before conversion.
    """
    from torchvision.transforms.functional import to_pil_image

    tensor = tensor.detach().cpu()
    if tensor.is_floating_point():
        tensor = tensor.clamp(0.0, 1.0)
    return to_pil_image(tensor)


def _encode_image_b64(
    image: Union[Image.Image, torch.Tensor],
    image_format: str = "JPEG",
    quality: int = 95,
) -> str:
    """Encode an image to a base64 string for the RewardService wire format."""
    if isinstance(image, torch.Tensor):
        image = _pil_from_tensor(image)
    if image.mode != "RGB":
        image = image.convert("RGB")

    buf = io.BytesIO()
    save_kwargs: dict = {"format": image_format}
    if image_format.upper() == "JPEG":
        save_kwargs["quality"] = quality
    image.save(buf, **save_kwargs)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _encode_video_b64(
    video: torch.Tensor,
    fps: int = 8,
) -> str:
    """Encode a video tensor ``(C, T, H, W)`` to a base64 mp4 string.

    Uses ``diffusers.utils.export_to_video`` for frame encoding, then reads
    the bytes and base64-encodes them for HTTP transmission.
    """
    import tempfile

    from diffusers.utils import export_to_video
    from PIL import Image as _PIL_Image

    v = video.detach().cpu()
    if v.dim() == 5:
        v = v.squeeze(0)
    if v.dim() != 4:
        raise ValueError(f"Expected 4D (C, T, H, W) video tensor, got shape {tuple(v.shape)}.")

    # Channel-first to list of PIL frames
    if v.is_floating_point():
        v = v.clamp(0.0, 1.0)
    frames = []
    for t in range(v.shape[1]):
        frame = v[:, t, :, :]  # (C, H, W)
        if frame.is_floating_point():
            frame = (frame * 255).byte()
        frame_np = frame.permute(1, 2, 0).numpy()
        frames.append(_PIL_Image.fromarray(frame_np))

    tmp = tempfile.NamedTemporaryFile(prefix="reward_svc_", suffix=".mp4", delete=False)
    tmp.close()
    import os

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


# ---------------------------------------------------------------------------
# RemoteRewardBackend
# ---------------------------------------------------------------------------


class RemoteRewardBackend(RewardBackend):
    """HTTP client backend for the remote RewardService ``POST /score`` endpoint.

    Converts UniRL's ``RewardRequest`` (flat images + prompts) into
    the RewardService wire format (list of per-sample history-turn requests),
    calls the service, and converts the nested response back into a flat
    ``RewardResponse``.

    One instance handles all ``required_rewards`` in a single HTTP call,
    because the RewardService server multiplexes multiple reward models
    via the ``required_rewards`` field per request.

    Constructed by ``_target_`` with a :class:`RemoteRewardSpec` config;
    ``base_device`` is accepted for backend-interface uniformity but ignored
    (the backend is HTTP-only).
    """

    _REDUCE_STRATEGIES = {"first", "mean", "max"}
    _AGGREGATION_METHODS = {"weighted_sum", "mean", "min", "max"}

    def __init__(self, *, config: "RemoteRewardSpec", base_device: str) -> None:
        del base_device  # HTTP backend, no device dependency
        super().__init__(
            model_name="reward_service",
            batch_size=config.batch_size,
            timeout=config.timeout,
        )
        self.base_url = config.base_url.rstrip("/")
        self.required_rewards = list(config.required_rewards)
        self.reward_weights = dict(config.reward_weights or {})
        self.max_retries = config.max_retries
        self.retry_delay = config.retry_delay
        self.sub_metric_reduce = config.sub_metric_reduce
        self.image_format = config.image_format
        self.image_quality = config.image_quality
        self.raise_on_failure = config.raise_on_failure
        self.aggregation_method = config.aggregation_method
        self.video_fps = config.video_fps
        # Instance attr overrides the RewardBackend.input_kind class default;
        # RewardService reads it via preferred_input_kind to route image vs video.
        self.input_kind = config.input_kind

        self._remote_rewards_validated = False

        # Disable proxy env vars — reward services are typically on an internal
        # network where corporate HTTP proxies (squid etc.) would return 503.
        self._session = http_requests.Session()
        self._session.trust_env = False

    # ------------------------------------------------------------------
    # Public interface (RewardBackend)
    # ------------------------------------------------------------------

    def compute_rewards(self, request: RewardRequest) -> RewardResponse:
        """Convert a UniRL request, call the remote service, and
        convert the response back.

        Video requests are rejected explicitly: this executor only knows the
        image-history wire format, and silently falling through would yield
        all-zero rewards that corrupt downstream advantage computation.

        By default (``raise_on_failure=True``) HTTP errors and malformed
        responses are propagated as exceptions so that corrupted zero-rewards
        never silently enter the training loop.  Set ``raise_on_failure=False``
        for a degraded mode that returns zeroed rewards on failure.

        Returns:
            A ``RewardResponse`` with ``rewards`` (one float per sample —
            the weighted aggregation of all required_rewards), and
            ``component_rewards`` keyed by reward name.
        """
        start = time.time()
        if request.is_video:
            return self._compute_video_rewards(request, start)

        bs = request.batch_size
        try:
            payload = self._build_score_payload(request)
            raw = self._post_score(payload)
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

    def is_available(self) -> bool:
        """Ping ``/health``; ``True`` iff the server is reachable.

        On the first successful ping also runs roster validation
        (see ``_validate_required_rewards_once``); transport errors and
        non-200 still return ``False``.
        """
        try:
            resp = self._session.get(
                f"{self.base_url}/health",
                timeout=5.0,
            )
        except http_requests.exceptions.RequestException:
            return False
        if resp.status_code != 200:
            return False
        self._validate_required_rewards_once(resp)
        return True

    def _validate_required_rewards_once(self, health_resp: http_requests.Response) -> None:
        """One-shot: ``raise ValueError`` if any required reward is not in the
        ``/health`` roster (catches component-name typos at startup, not at
        first ``/score`` call); log the full roster at INFO on success.

        Expected ``/health`` body shape:
        ``{"status": "ok", "rewards": {<name>: [<readiness>, ...], ...}}``.
        """
        if self._remote_rewards_validated:
            return

        try:
            body = health_resp.json()
        except ValueError as e:
            raise ValueError(f"RemoteRewardBackend: /health at {self.base_url} returned non-JSON body.") from e

        if not isinstance(body, dict) or not isinstance(body.get("rewards"), dict):
            raise ValueError(
                f"RemoteRewardBackend: /health at {self.base_url} returned unexpected shape: "
                f"{body!r}. Expected {{'status': 'ok', 'rewards': {{<name>: [...]}}}}."
            )

        available = sorted(body["rewards"].keys())
        available_set = set(available)
        missing = [name for name in self.required_rewards if name not in available_set]
        if missing:
            raise ValueError(
                f"RemoteRewardBackend: required_rewards={missing} not served by "
                f"{self.base_url}; server reports available={available}. "
                f"Check REWARD_COMPONENTS for typos "
                f"(e.g. 'unifiedreward' vs 'unified_reward')."
            )

        logger.info(
            "RemoteRewardBackend: %s serves rewards=%s",
            self.base_url,
            available,
        )
        self._remote_rewards_validated = True

    def dispose(self) -> None:
        """Close the HTTP session."""
        self._session.close()

    # ------------------------------------------------------------------
    # Request conversion: UniRL → RewardService wire format
    # ------------------------------------------------------------------

    def _build_score_payload(self, request: RewardRequest) -> Dict[str, Any]:
        """Convert a UniRL ``RewardRequest`` into the RewardService
        ``ScoreRequest`` JSON payload.

        Each sample ``(images[i], prompts[i])`` becomes one entry in the
        wire-format ``requests`` list, with
        ``history = [{"text": prompt, "image_b64": ...}]`` and
        ``required_rewards`` set to ``self.required_rewards``.

        When the request carries condition images in ``primitives["image"]``
        (e.g. image-editing pipelines), a two-turn history is built:
            history[0] = {"text": prompt, "image_b64": condition_image}
            history[1] = {"text": prompt, "image_b64": generated_image}
        This matches the EditReward scorer's input convention.

        Per-sample metadata from ``request.metadata`` is forwarded when present.
        """
        images = request.images or []
        prompts = request.prompts
        metadata_list = request.metadata
        wire_requests: List[Dict[str, Any]] = []

        # Check for condition images in primitives (source images for editing)
        condition_images = self._get_condition_images(request)

        for idx in range(len(images)):
            prompt = prompts[idx] if idx < len(prompts) else ""
            image_b64 = _encode_image_b64(
                images[idx],
                image_format=self.image_format,
                quality=self.image_quality,
            )
            sample_metadata = None
            if metadata_list is not None and idx < len(metadata_list):
                sample_metadata = metadata_list[idx]

            if condition_images is not None and idx < len(condition_images):
                # Two-turn history: condition (source) image + generated (edited) image
                condition_b64 = _encode_image_b64(
                    condition_images[idx],
                    image_format=self.image_format,
                    quality=self.image_quality,
                )
                history = [
                    {"text": prompt, "image_b64": condition_b64},
                    {"text": prompt, "image_b64": image_b64},
                ]
            else:
                # Single-turn history: generated image only (T2I)
                history = [{"text": prompt, "image_b64": image_b64}]

            wire_requests.append(
                {
                    "history": history,
                    "required_rewards": list(self.required_rewards),
                    "metadata": sample_metadata,
                }
            )

        return {"requests": wire_requests}

    def _get_condition_images(self, request: RewardRequest) -> Optional[List[Union[Image.Image, torch.Tensor]]]:
        """Extract per-sample condition images from request primitives.

        Returns None if no condition images are present (pure T2I case).
        """
        prim_image = request.primitives.get("image")
        if prim_image is None:
            return None
        # Images batch has .pixels [B, C, H, W]
        from unirl.utils.media import tensor_frame_to_pil

        return [tensor_frame_to_pil(img) for img in prim_image.pixels.unbind(0)]

    # ------------------------------------------------------------------
    # Video reward support
    # ------------------------------------------------------------------

    def _compute_video_rewards(self, request: RewardRequest, start: float) -> RewardResponse:
        """Send video tensors to the remote service and parse the response."""
        bs = request.batch_size
        try:
            payload = self._build_video_score_payload(request)
            raw = self._post_score(payload)
            return self._parse_score_response(raw, bs, time.time() - start)
        except Exception:
            if self.raise_on_failure:
                raise
            logger.exception("RemoteRewardBackend._compute_video_rewards failed (degraded mode)")
            return RewardResponse(
                rewards=[0.0] * bs,
                successes=[False] * bs,
                errors=["RemoteRewardBackend video failure (see logs)"] * bs,
                compute_time=time.time() - start,
            )

    def _build_video_score_payload(self, request: RewardRequest) -> Dict[str, Any]:
        """Convert a video ``RewardRequest`` into the RewardService ``ScoreRequest``
        JSON payload.

        Mirrors :meth:`_build_score_payload`: each sample ``(videos[i],
        prompts[i])`` becomes one wire request whose single history turn carries
        ``{"text": prompt, "video_b64": ...}``. The server's ``HistoryTurn``
        schema requires a history list — a flat ``{"video_b64", "prompt"}`` body
        is rejected with HTTP 422 — so video and image payloads share the same
        history-turn shape.

        Per-sample metadata from ``request.metadata`` is forwarded when present.
        """
        videos = request.videos or []
        prompts = request.prompts
        metadata_list = request.metadata
        wire_requests: List[Dict[str, Any]] = []

        for idx in range(len(videos)):
            prompt = prompts[idx] if idx < len(prompts) else ""
            video_b64 = _encode_video_b64(videos[idx], fps=self.video_fps)
            sample_metadata = None
            if metadata_list is not None and idx < len(metadata_list):
                sample_metadata = metadata_list[idx]
            wire_requests.append(
                {
                    "history": [{"text": prompt, "video_b64": video_b64}],
                    "required_rewards": list(self.required_rewards),
                    "metadata": sample_metadata,
                }
            )

        return {"requests": wire_requests}

    # ------------------------------------------------------------------
    # HTTP call with retries
    # ------------------------------------------------------------------

    def _post_score(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST to ``/score`` with retry logic.

        Raises ``RuntimeError`` chained from the last underlying exception
        if all retries are exhausted.
        """
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

    # ------------------------------------------------------------------
    # Response conversion: RewardService wire format → UniRL
    # ------------------------------------------------------------------

    def _parse_score_response(
        self,
        raw: Dict[str, Any],
        batch_size: int,
        compute_time: float,
    ) -> RewardResponse:
        """Convert the RewardService ``ScoreResponse`` JSON into a UniRL
        ``RewardResponse``.

        For each sample *i*:

        1. Extract ``results[i][reward_name] → {sub_metric: float}``.
        2. Reduce sub-metrics to one float per reward via
           ``_reduce_sub_metrics``.
        3. Store per-reward scores in ``component_rewards``.
        4. Aggregate across rewards → ``rewards[i]`` using
           ``self.aggregation_method``.
        """
        results: List[Dict[str, Dict[str, float]]] = raw.get("results", [])
        errors_list: List[Dict[str, str]] = raw.get("errors", [])

        # Pad if server returned fewer entries than expected.
        while len(results) < batch_size:
            results.append({})
        while len(errors_list) < batch_size:
            errors_list.append({})

        component_rewards: Dict[str, List[float]] = {name: [] for name in self.required_rewards}
        aggregated_rewards: List[float] = []
        successes: List[bool] = []
        sample_errors: List[Optional[str]] = []

        for i in range(batch_size):
            sample_result = results[i]
            sample_errors_dict = errors_list[i]

            scores: List[float] = []
            weights: List[float] = []
            error_parts: List[str] = []

            # Validation contract: every reward this sample asked for must come
            # back. Today every sample asks for the same self.required_rewards
            # (set at executor construction). When per-sample required_rewards
            # arrives for multi-turn rollouts (the wire format already supports
            # it via wire_requests[i]["required_rewards"]), replace the loop
            # source with `request.required_rewards[i]` — the failure semantics
            # ("asked-for not returned") stay identical.
            for reward_name in self.required_rewards:
                if reward_name in sample_result:
                    sub_metrics = sample_result[reward_name]
                    # Any non-finite sub-metric fails the whole reward for this
                    # sample (a partially-broken output is suspect), even an axis
                    # the reduction below would not select.
                    non_finite = self._first_non_finite(sub_metrics)
                    if non_finite is not None:
                        # NaN/inf/null marks an unusable score: the scorer hit a
                        # per-item failure (e.g. OCR could not read the image), or a
                        # NaN serialized to JSON null on the wire. Flag the sample as
                        # failed instead of feeding a non-finite value into advantage
                        # normalization, where it would poison the whole group.
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
                        # Asked-for reward absent without explanation: server bug, not legitimate omission.
                        error_parts.append(f"{reward_name}: missing from server response without error")

            if scores:
                aggregated_rewards.append(self._aggregate_scores(scores, weights))
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

    def _aggregate_scores(self, scores: List[float], weights: List[float]) -> float:
        """Aggregate per-reward scores for one sample.

        Strategies:
            ``"weighted_sum"``: ``Σ(score_k * w_k) / Σ(w_k)``.
            ``"mean"``: arithmetic mean (ignores weights).
            ``"min"``: minimum score across rewards.
            ``"max"``: maximum score across rewards.
        """
        if not scores:
            return 0.0
        if self.aggregation_method == "weighted_sum":
            total_w = sum(weights)
            return sum(s * w for s, w in zip(scores, weights)) / total_w if total_w > 0 else 0.0
        if self.aggregation_method == "mean":
            return sum(scores) / len(scores)
        if self.aggregation_method == "min":
            return min(scores)
        # "max"
        return max(scores)

    @staticmethod
    def _first_non_finite(sub_metrics: Dict[str, float]) -> Optional[Tuple[str, Any]]:
        """Return the first ``(name, value)`` whose value is not a finite number.

        Catches ``None`` (a server-side NaN that serialized to JSON ``null``),
        ``NaN`` / ``inf`` floats, booleans, and non-numeric junk. Returns
        ``None`` when every value is a finite number. Used to convert an
        unusable reward into a sample failure before it reaches advantage
        computation.
        """
        for name, value in sub_metrics.items():
            # bool is an int subclass, so reject it explicitly before the numeric
            # check — a True/False reward is junk, not a 1.0/0.0 score.
            if (
                value is None
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                return name, value
        return None

    def _reduce_sub_metrics(self, sub_metrics: Dict[str, float]) -> float:
        """Collapse a reward's sub-metric dict into a single float.

        Strategies:
            ``"first"``: value of the first sub-metric
                (stable iteration order in Python 3.7+).
            ``"mean"``: arithmetic mean of all sub-metric values.
            ``"max"``: maximum sub-metric value.
        """
        if not sub_metrics:
            return 0.0
        values = list(sub_metrics.values())
        if self.sub_metric_reduce == "first":
            return float(values[0])
        if self.sub_metric_reduce == "mean":
            return float(sum(values) / len(values))
        # "max"
        return float(max(values))


@dataclass
class RemoteRewardSpec(BaseRewardComponentSpec):
    """Typed config for the remote RewardService backend.

    Registered as a polymorphic ``reward/component``; one instance multiplexes
    all ``required_rewards`` in a single HTTP round-trip to ``base_url``.
    """

    base_url: str = ""
    required_rewards: Tuple[str, ...] = ()
    reward_weights: Optional[Dict[str, float]] = None
    batch_size: int = 8
    # Per-attempt HTTP read timeout. Keep it above the RewardService server's
    # per-reward score_timeout_s (ServerCfg default 120s) so a server-side reward
    # timeout comes back as a structured errors[i][reward] instead of the client
    # timing out first and re-POSTing the whole batch (burning the retry budget).
    timeout: float = 300.0
    max_retries: int = 3
    retry_delay: float = 1.0
    sub_metric_reduce: str = "first"
    aggregation_method: str = "weighted_sum"
    image_format: str = "JPEG"
    image_quality: int = 95
    video_fps: int = 8
    # "image" (default) or "video": selects which decoded-media key
    # RewardService.score_and_attach populates, hence whether compute_rewards
    # builds an image or a video payload. A video reward (e.g. videoalign) is
    # configured as its own component with input_kind: video.
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
