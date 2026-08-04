"""Video HPSv3 scorer — per-frame HPSv3 with top-ratio aggregation.

Reproduces the reward used by Flash-GRPO's WAN2.1 T2V recipe
(``flow_grpo/rewards.py::video_hpsv3_remote``): score every frame of a
generated video with HPSv3, sort the per-frame scores descending, and
return the mean of the top 30%. Taking the best-scoring frames rather
than the whole-clip mean keeps the signal from being dominated by the
weakest frames, which is what upstream tuned against.

Upstream splits the work across the wire: its client extracts frames,
JPEG-compresses them, and POSTs each video's frames to a dedicated
HPSv3 server that only ever sees "a batch of images". Here both halves
live in one scorer, matching this repo's convention that a reward name
maps to a single self-contained scorer.

Implementation: :class:`HPSv3Scorer` is reused wholesale as an internal
component — it already owns checkpoint resolution, the
``max_batch_size`` chunking that bounds Qwen2-VL-7B activation memory,
and the inter-chunk ``empty_cache()``. Each extracted frame is wrapped
in a single-turn :class:`ScoreItem` and handed to it, so this module
adds only frame extraction and aggregation.

Cost warning: one forward per frame. An 81-frame clip at
``frame_stride=1`` is 81 Qwen2-VL-7B forwards, so a 64-video rollout
costs ~5.2k forwards. ``frame_stride`` subsamples frames to trade
fidelity for throughput; it defaults to 1 (exact upstream semantics).
Raise ``server.score_timeout_s`` well above its 120 s default before
enabling this reward.

Failure semantics: an undecodable clip scores ``float("nan")`` rather
than failing the batch. Scoring one clip at a time makes that isolation
free, so a single corrupt video does not discard a whole rollout's GPU
time (contrast ``videoalign``, which hands its entire batch to one
inferencer call and has no per-item boundary to catch at).

Deps: identical to the image ``hpsv3`` scorer, so deployments point
``runtime_env`` at ``envs/hpsv3.txt`` (which already carries
``opencv-python-headless``) and Ray reuses the same cached venv.
"""

from __future__ import annotations

import contextlib
import os
from typing import TYPE_CHECKING

import torch

from reward_service.logging_utils import get_logger
from reward_service.scorers._common import materialize_video
from reward_service.scorers.base import BaseScorer, ScoreItem
from reward_service.scorers.hpsv3_scorer import HPSv3Scorer
from reward_service.scorers.registry import register

if TYPE_CHECKING:
    from PIL import Image

logger = get_logger(__name__)

# Fraction of best-scoring frames averaged into the final reward.
# 0.3 is the upstream Flash-GRPO value.
DEFAULT_TOP_RATIO = 0.3


def top_ratio_mean(scores: list[float], top_ratio: float) -> float:
    """Return the mean of the highest-scoring ``top_ratio`` fraction of scores.

    Args:
        scores: Per-frame scores, in any order. Must not be empty.
        top_ratio: Fraction of frames to keep, in ``(0, 1]``.

    Returns:
        Mean of the ``max(1, int(len(scores) * top_ratio))`` largest scores.

    Raises:
        ValueError: If ``scores`` is empty or ``top_ratio`` is outside ``(0, 1]``.
    """
    if not scores:
        raise ValueError("scores must not be empty")
    if not 0.0 < top_ratio <= 1.0:
        raise ValueError(f"top_ratio must be in (0, 1], got {top_ratio}")
    # The max(1, ...) guard matters for short clips: upstream's bare
    # int(l * 0.3) is 0 — and raises ZeroDivisionError — for any clip with 3
    # or fewer frames, so this degrades to "the single best frame" exactly
    # where upstream would crash.
    keep = max(1, int(len(scores) * top_ratio))
    best = sorted(scores, reverse=True)[:keep]
    return sum(best) / len(best)


def extract_frames(video_path: str, frame_stride: int = 1) -> list["Image.Image"]:
    """Decode a video into RGB PIL frames, keeping every ``frame_stride``-th one.

    Frames are read sequentially rather than by random seek: the clips here
    are short and fully consumed, so sequential decode avoids per-seek
    keyframe rewinds.

    Args:
        video_path: Path to a video file readable by OpenCV.
        frame_stride: Keep frames at indices ``0, stride, 2*stride, ...``.
            ``1`` keeps every frame. Must be positive.

    Returns:
        RGB frames in presentation order.

    Raises:
        ValueError: If ``frame_stride`` is not positive, the file cannot be
            opened, or no frames could be decoded.
    """
    # Imported lazily so the module stays importable (for unit tests and
    # registry introspection) without the actor venv's cv2/PIL present.
    import cv2
    from PIL import Image

    if frame_stride <= 0:
        raise ValueError(f"frame_stride must be positive, got {frame_stride}")

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"could not open video: {video_path}")
    frames: list[Image.Image] = []
    try:
        index = 0
        while True:
            is_read, frame_bgr = capture.read()
            if not is_read:
                break
            if index % frame_stride == 0:
                frames.append(Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)))
            index += 1
    finally:
        capture.release()

    if not frames:
        raise ValueError(f"decoded 0 frames from video: {video_path}")
    return frames


class VideoHPSv3Scorer(BaseScorer):
    name = "videohpsv3"
    sub_metric_names = ("videohpsv3",)

    def __init__(
        self,
        weights_path: str | None = None,
        config_path: str | None = None,
        device: str = "cuda",
        max_batch_size: int = 4,
        frame_stride: int = 1,
        top_ratio: float = DEFAULT_TOP_RATIO,
    ) -> None:
        """Load HPSv3 and configure frame sampling / aggregation.

        Args:
            weights_path: Directory containing ``HPSv3.safetensors``, the file
                itself, or a Hugging Face repo id; ``None`` uses the hpsv3
                package's default HF download. Forwarded to
                :class:`HPSv3Scorer`.
            config_path: Optional HPSv3 config YAML override. Forwarded to
                :class:`HPSv3Scorer`.
            device: Target device (``"cuda"`` / ``"cpu"``).
            max_batch_size: Frames per inferencer forward. Bounds peak
                activation memory; frames of one clip are chunked by
                :class:`HPSv3Scorer`. Defaults to the image scorer's 4 —
                8 has been observed to OOM a 95 GB H20.
            frame_stride: Keep every Nth frame. ``1`` (default) reproduces
                upstream exactly; raise it to cut cost roughly linearly.
            top_ratio: Fraction of best frames averaged into the reward.
                Defaults to upstream's 0.3.

        Raises:
            ValueError: If ``frame_stride`` is not positive or ``top_ratio``
                is outside ``(0, 1]``.
        """
        if frame_stride <= 0:
            raise ValueError(f"frame_stride must be positive, got {frame_stride}")
        if not 0.0 < top_ratio <= 1.0:
            raise ValueError(f"top_ratio must be in (0, 1], got {top_ratio}")
        self._frame_stride = frame_stride
        self._top_ratio = top_ratio
        # Composition, not inheritance: the image scorer is a complete
        # frame-scoring engine (checkpoint resolution + batching + cache
        # reclaim) and this class only adds decode/aggregate around it.
        self._frame_scorer = HPSv3Scorer(
            weights_path=weights_path,
            config_path=config_path,
            device=device,
            max_batch_size=max_batch_size,
        )

    @torch.inference_mode()
    def score(self, items: list[ScoreItem]) -> list[dict[str, float]]:
        if not items:
            return []

        # One clip at a time: a single clip already far exceeds
        # max_batch_size, so cross-item batching adds no throughput while
        # making peak memory harder to reason about.
        clip_scores: list[dict[str, float]] = []
        for i, item in enumerate(items):
            if item.videos is None or item.videos[-1] is None:
                raise ValueError(
                    f"videohpsv3 requires a video on item[{i}]'s last turn; "
                    f"got videos={item.videos!r}"
                )
            clip_scores.append(
                {"videohpsv3": self._score_clip(item.videos[-1], item.history[-1][0])}
            )
        return clip_scores

    def _score_clip(self, source: bytes | str, prompt: str) -> float:
        """Score one clip: materialise, decode, score every frame, aggregate.

        Returns ``float("nan")`` if the clip cannot be decoded. Per
        :meth:`BaseScorer.score`'s contract that is the in-band "value
        unavailable" marker, and it isolates one corrupt clip instead of
        failing the whole reward bucket — cheap here because clips are
        already scored one at a time.
        """
        owned_tempfiles: list[str] = []
        try:
            video_path = materialize_video(source, owned_tempfiles, prefix="videohpsv3_")
            frames = extract_frames(video_path, self._frame_stride)
        except ValueError:
            logger.warning("videohpsv3: could not decode clip, scoring it NaN", exc_info=True)
            return float("nan")
        finally:
            # Frames are decoded into memory, so the tempfile is dead weight
            # past this point — release it before the (much longer) scoring.
            for path in owned_tempfiles:
                with contextlib.suppress(OSError):
                    os.unlink(path)

        frame_items = [ScoreItem(history=[(prompt, frame)]) for frame in frames]
        logger.debug("videohpsv3: scoring %d frames (stride=%d)", len(frames), self._frame_stride)
        scored_frames = self._frame_scorer.score(frame_items)
        return top_ratio_mean([s["hpsv3"] for s in scored_frames], self._top_ratio)

    def close(self) -> None:
        self._frame_scorer.close()


register("videohpsv3", VideoHPSv3Scorer)
