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

PROTOCOL_VERSION = "1"


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
    image_ipc: str | None = Field(
        default=None,
        description=(
            "CUDA-IPC handle blob for a float image tensor [C,H,W] in [0,1] "
            "(see tensor_ipc.encode_tensor). Negotiated data plane only: valid "
            "after a successful /handshake, same-device loopback deployments. "
            "Lossless and zero-copy; the sender keeps the tensor alive until "
            "the response arrives."
        ),
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
        if self.image_b64 is not None and self.image_ipc is not None:
            raise ValueError("image_b64 and image_ipc are mutually exclusive")
        if (
            self.image_b64 is None
            and self.image_ipc is None
            and self.video_b64 is None
            and self.video_path is None
        ):
            raise ValueError(
                "HistoryTurn must include at least one of image_b64, image_ipc, video_b64, or video_path"
            )
        return self


class RewardRequest(BaseModel):
    """A single scoring request routed to one or more reward models."""

    history: list[HistoryTurn]
    required_rewards: list[str]
    metadata: dict[str, Any] | None = None
    request_id: str | None = None
    sample_id: str | None = None
    group_id: str | None = None
    source_rank: int | None = None
    policy_version: int | None = None
    scorer_version: str | None = None
    idempotency_key: str | None = None

    def identity(self, *, actual_scorer_version: str | None = None) -> "RewardIdentity":
        return RewardIdentity(
            request_id=self.request_id,
            sample_id=self.sample_id,
            group_id=self.group_id,
            source_rank=self.source_rank,
            policy_version=self.policy_version,
            scorer_version=actual_scorer_version,
            idempotency_key=self.idempotency_key,
        )


class RewardIdentity(BaseModel):
    """Optional item identity echoed across chunked/retried score calls."""

    request_id: str | None = None
    sample_id: str | None = None
    group_id: str | None = None
    source_rank: int | None = None
    policy_version: int | None = None
    scorer_version: str | None = None
    idempotency_key: str | None = None


class ScoreRequest(BaseModel):
    """Top-level HTTP body: a batch of RewardRequest."""

    protocol_version: str = PROTOCOL_VERSION
    requests: list[RewardRequest]
    grad_mode: bool = Field(
        default=False,
        description=(
            "Differentiable scoring: the scorer forward runs under grad and the "
            "server retains the subgraph keyed by call_id until /backward or a "
            "lifecycle release. Requires image_ipc inputs and a scorer with "
            "supports_grad. grad_mode batches bypass the idempotency cache — "
            "a retained graph is stateful, not a pure function."
        ),
    )
    call_id: str | None = Field(
        default=None,
        description="Caller-chosen key for the retained subgraph; required when grad_mode",
    )


class HandshakeRequest(BaseModel):
    """Parent's IPC fingerprint + proposed data plane."""

    fingerprint: dict[str, str]
    proposed_transport: str = "cuda_ipc"


class HandshakeResponse(BaseModel):
    """Child's fingerprint and the transport it accepted."""

    fingerprint: dict[str, str]
    accepted_transport: str
    reason: str = ""


class BackwardRequest(BaseModel):
    """Second half of a grad_mode score: seed grads for the retained subgraph.

    ``grad_scores[i]`` is dL/dr for request i's scalar reward (the upstream
    grads are tiny — plain JSON floats). The response's image grads are
    full-size tensors and travel back as IPC handles.
    """

    call_id: str
    grad_scores: list[float]


class BackwardResponse(BaseModel):
    grad_ipc: list[str | None]
    protocol_version: str = PROTOCOL_VERSION


class ScoreResponse(BaseModel):
    """Top-level HTTP response: one entry per request, each a nested dict.

    results[i][reward_name][sub_metric] -> float

    If a reward model fails for a given request, its entry is omitted
    and the reward name is listed in `errors[i][reward_name]` instead.
    """

    results: list[dict[str, dict[str, float]]]
    errors: list[dict[str, str]] = Field(default_factory=list)
    identities: list[RewardIdentity] = Field(default_factory=list)
    protocol_version: str = PROTOCOL_VERSION
