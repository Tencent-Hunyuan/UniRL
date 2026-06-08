"""HTTP request / response schemas.

The wire format carries images as base64-encoded bytes and videos as
either base64-encoded bytes (``video_b64``) or a path on the server's
local filesystem (``video_path``). Image-only T2I scorers consume the
images; video-aware scorers (e.g. ``videoalign``) consume the videos.
A turn must carry at least one media field — text-only turns are
rejected.

This module is the **single source of truth** for the RewardService wire
protocol. The training repo (UniRL) builds its HTTP payloads to
match these models but does not import this package at runtime; a contract
test there (``tests/reward/test_wire_contract.py``) validates its payloads
against ``ScoreRequest``, so the two sides cannot drift silently. Any change
to these models must be mirrored by that test.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class HistoryTurn(BaseModel):
    """One ``(text, media)`` pair in a conversation history.

    Exactly one of the video fields may be set per turn (mutual
    exclusion). Image and video may coexist if a future scorer wants
    a key-frame plus the full clip — current scorers only consume one
    of them.
    """

    text: str
    image_b64: str | None = Field(
        default=None,
        description="Base64-encoded image bytes (any PIL-readable format)",
    )
    video_b64: str | None = Field(
        default=None,
        description="Base64-encoded video file bytes (e.g. mp4); decoded server-side to a tempfile",
    )
    video_path: str | None = Field(
        default=None,
        description="Absolute path to a video file the server can read directly (shared-FS deployments)",
    )

    @model_validator(mode="after")
    def _check_media(self) -> "HistoryTurn":
        if self.video_b64 is not None and self.video_path is not None:
            raise ValueError("video_b64 and video_path are mutually exclusive")
        if self.image_b64 is None and self.video_b64 is None and self.video_path is None:
            raise ValueError(
                "HistoryTurn must include at least one of image_b64, video_b64, or video_path"
            )
        return self


class RewardRequest(BaseModel):
    """A single scoring request routed to one or more reward models."""

    history: list[HistoryTurn]
    required_rewards: list[str]
    metadata: dict[str, Any] | None = None


class ScoreRequest(BaseModel):
    """Top-level HTTP body: a batch of RewardRequest."""

    requests: list[RewardRequest]


class ScoreResponse(BaseModel):
    """Top-level HTTP response: one entry per request, each a nested dict.

    results[i][reward_name][sub_metric] -> float

    If a reward model fails for a given request, its entry is omitted
    and the reward name is listed in `errors[i][reward_name]` instead.
    """

    results: list[dict[str, dict[str, float]]]
    errors: list[dict[str, str]] = Field(default_factory=list)
