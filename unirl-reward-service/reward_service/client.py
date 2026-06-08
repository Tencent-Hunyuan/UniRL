"""Python client SDK.

Handles PIL.Image → base64 encoding and wraps the HTTP round-trip. This
is the entry point RL training loops should use; the server is kept
framework-agnostic.

Default image encoding is JPEG (q=95): ~5-10× smaller than PNG with
negligible effect on reward scores. Callers that need lossless pixels
should set ``image_format="PNG"``.

Video support: each ``RewardRequest`` may carry a parallel ``videos``
sequence (one per history turn). Each entry is one of:

* ``str`` — path on the *server's* local filesystem; sent as
  ``video_path`` (zero-copy, recommended for shared-FS deployments)
* ``bytes`` — already-loaded video file bytes; sent as base64-encoded
  ``video_b64`` (self-contained, recommended when client and server
  are on different machines)
* ``None`` — turn has no video (can mix image-only and video-bearing
  turns within the same request).
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from pathlib import Path

import requests
from PIL import Image

# Public type alias for clarity at call sites.
VideoSource = str | bytes | None


@dataclass
class RewardRequest:
    """Input to RewardClient.score. Mirrors the server-side schema but
    takes PIL.Image objects directly so callers don't touch base64.

    For T2V scorers, supply a parallel ``videos`` list (same length as
    ``history``). Each entry is a server-side path string, raw mp4
    bytes, or ``None`` when the turn has no video.
    """

    history: list[tuple[str, Image.Image | None]]
    required_rewards: list[str]
    metadata: dict | None = field(default=None)
    videos: list[VideoSource] | None = field(default=None)


def _encode_image(image: Image.Image, image_format: str = "JPEG", quality: int = 95) -> str:
    buf = io.BytesIO()
    save_kwargs: dict = {"format": image_format}
    if image_format.upper() == "JPEG":
        save_kwargs["quality"] = quality
    image.save(buf, **save_kwargs)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _encode_video_bytes(video_bytes: bytes) -> str:
    """Base64-encode raw video bytes for the wire."""
    return base64.b64encode(video_bytes).decode("ascii")


class RewardClient:
    def __init__(
        self,
        base_url: str,
        timeout: float = 60.0,
        image_format: str = "JPEG",
        image_quality: int = 95,
        trust_env: bool = False,
    ) -> None:
        """Create a reward service HTTP client.

        Args:
            base_url: Root URL of the service, e.g. ``http://host:8080``.
            timeout: Per-request timeout in seconds. Covers the full round
                trip including server-side model forward.
            image_format: Image encoding for the wire payload. JPEG is
                ~5–10× smaller than PNG for typical T2I sizes with
                negligible effect on scores; use PNG if callers need
                lossless pixels.
            image_quality: JPEG quality (1-95); ignored when using PNG.
            trust_env: Whether the underlying ``requests.Session`` should
                honour ``HTTP(S)_PROXY`` / ``NO_PROXY`` env vars. Default
                is ``False`` because reward services are usually reached
                over an internal network and corporate HTTP proxies
                (squid etc.) will return 503 for ``localhost:*`` targets.
                Set to ``True`` if you really do need to go through a
                proxy.
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.image_format = image_format
        self.image_quality = image_quality
        self.session = requests.Session()
        self.session.trust_env = trust_env

    def score(self, requests_: list[RewardRequest]) -> list[dict[str, dict[str, float]]]:
        payload = {"requests": [self._encode_request(r) for r in requests_]}
        resp = self.session.post(
            f"{self.base_url}/score", json=payload, timeout=self.timeout
        )
        resp.raise_for_status()
        return resp.json()["results"]

    def health(self) -> dict:
        resp = self.session.get(f"{self.base_url}/health", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def rewards(self) -> list[str]:
        resp = self.session.get(f"{self.base_url}/rewards", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()["rewards"]

    def _encode_request(self, req: RewardRequest) -> dict:
        if req.videos is not None and len(req.videos) != len(req.history):
            raise ValueError(
                f"videos length ({len(req.videos)}) must match history length ({len(req.history)})"
            )
        history_out: list[dict] = []
        for i, (text, img) in enumerate(req.history):
            turn: dict = {"text": text}
            if img is not None:
                turn["image_b64"] = _encode_image(img, self.image_format, self.image_quality)
            video = req.videos[i] if req.videos is not None else None
            if video is not None:
                self._attach_video(turn, video)
            if "image_b64" not in turn and "video_b64" not in turn and "video_path" not in turn:
                raise ValueError(
                    f"history turn {i} has neither image nor video — server will reject it"
                )
            history_out.append(turn)
        return {
            "history": history_out,
            "required_rewards": list(req.required_rewards),
            "metadata": req.metadata,
        }

    @staticmethod
    def _attach_video(turn: dict, video: VideoSource) -> None:
        """Attach a video source onto a wire-format turn dict.

        ``str`` becomes ``video_path`` (server reads from shared FS);
        ``bytes`` becomes base64-encoded ``video_b64``.  Validating
        path-vs-bytes choice up front gives a clearer error than
        letting Pydantic complain about unexpected types.
        """
        if isinstance(video, (str, Path)):
            turn["video_path"] = str(video)
        elif isinstance(video, (bytes, bytearray, memoryview)):
            turn["video_b64"] = _encode_video_bytes(bytes(video))
        else:
            raise TypeError(
                f"video must be str (path) or bytes, got {type(video).__name__}"
            )
