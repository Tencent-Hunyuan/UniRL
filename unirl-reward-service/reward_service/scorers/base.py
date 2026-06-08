"""Scorer abstract base class and shared types.

A scorer takes a list of items (each item = a history of (text, media)
turns) and returns one dict of sub-metrics per item. Scorers are plain
Python objects; the Ray actor layer wraps them for distribution and
GPU isolation.

Modalities:

* T2I scorers consume ``item.history[*][1]`` (PIL images).
* T2V scorers (e.g. ``videoalign``) consume ``item.videos[*]``: the
  parallel-to-history sequence of video sources, each entry either
  raw ``bytes`` (the server decoded ``video_b64``) or a ``str`` path
  on the server's local filesystem (the server resolved
  ``video_path``). A scorer that needs a video but receives ``None``
  raises; the gateway routes purely by ``required_rewards`` name and
  does not gate on modality, so requesting a video scorer without a
  video surfaces as that reward's error for the request.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class ScoreItem:
    """Input for one reward request, as seen by a scorer.

    Args:
        history: Ordered list of ``(text, image)`` turns. ``image`` may
            be ``None`` for video-only turns. T2I-style scorers
            typically only consume the last turn's image; dialogue-style
            scorers may walk the whole list.
        videos: Optional parallel sequence of video sources, one per
            history turn. Each entry is either ``bytes`` (raw mp4 bytes
            decoded from ``video_b64``), a ``str`` (a path on the
            server's local filesystem coming from ``video_path``), or
            ``None`` for turns without a video. The whole field is
            ``None`` when the request had no video at all (the common
            T2I case) — image scorers can ignore it entirely.
        metadata: Optional dict forwarded as-is from the request.
    """

    history: list[tuple[str, Image.Image | None]]
    videos: tuple[bytes | str | None, ...] | None = None
    metadata: dict | None = None


class BaseScorer(ABC):
    """Contract every reward scorer implements.

    Subclasses must:
      - implement `score(items)` returning one dict per item
      - set `sub_metric_names` (class attr or set in __init__) so the
        router can document what sub-metrics a reward emits
    """

    name: str = ""
    sub_metric_names: tuple[str, ...] = ()

    @abstractmethod
    def score(self, items: list[ScoreItem]) -> list[dict[str, float]]:
        """Return a list of {sub_metric: score} dicts, one per input item.

        An empty input list must yield an empty output list.
        Output length is always equal to input length.

        Failure semantics:

        * **Per-item failure** (e.g. the model could not read one image): put
          ``float("nan")`` in that item's sub-metric instead of a fabricated
          number. NaN is the in-band "value unavailable" marker — the consumer
          (UniRL) treats any non-finite reward as a sample failure
          and drops it via fail-fast, so a NaN never enters the training signal
          as a real score. Raising instead would fail the whole batch, because
          one ``score()`` call serves the entire reward bucket.
        * **Whole-reward / configuration failure** (model crashed, GPU OOM,
          required metadata not wired): ``raise``. The gateway captures it in
          ``errors[i][reward]`` for every item in the batch.
        """

    def close(self) -> None:
        """Release heavy resources (model, vLLM engine). Default is a no-op."""
