"""Reward HTTP wire decoding shared by the full and direct servers."""

from __future__ import annotations

import base64
import io
from pathlib import Path

from fastapi import HTTPException
from PIL import Image

from reward_service.schemas import HistoryTurn, RewardRequest
from reward_service.scorers import ScoreItem


def decode_image(image_b64: str) -> Image.Image:
    try:
        raw = base64.b64decode(image_b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid image_b64: {exc}") from exc


def decode_ipc_image(blob: str) -> Image.Image:
    """Materialize an IPC-shared [C,H,W] float tensor as a PIL image.

    Classic scorers consume PIL (inherently 8-bit); the lossless float path
    is ``score_tensors``. This hop still skips the JPEG/PNG encode-decode
    round trip and is exact for tensors that came from 8-bit sources.
    """
    try:
        from reward_service.tensor_ipc import decode_tensor

        tensor = decode_tensor(blob)
        tensor = tensor.detach().clamp(0, 1).mul(255).round().to("cpu").byte()
        return Image.fromarray(tensor.permute(1, 2, 0).numpy())
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid image_ipc handle: {exc}") from exc


def resolve_video(turn: HistoryTurn) -> bytes | str | None:
    if turn.video_path is not None:
        path = Path(turn.video_path)
        if not path.is_file():
            raise HTTPException(
                status_code=400,
                detail=f"video_path does not exist or is not a regular file: {turn.video_path}",
            )
        return str(path)
    if turn.video_b64 is not None:
        try:
            return base64.b64decode(turn.video_b64)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"invalid video_b64: {exc}") from exc
    return None


def request_to_item(request: RewardRequest, *, allow_video: bool = True) -> ScoreItem:
    if not request.history:
        raise HTTPException(status_code=400, detail="history must not be empty")
    history: list[tuple[str, Image.Image | None]] = []
    videos: list[bytes | str | None] = []
    any_video = False
    for turn in request.history:
        if turn.image_b64 is not None:
            image = decode_image(turn.image_b64)
        elif turn.image_ipc is not None:
            image = decode_ipc_image(turn.image_ipc)
        else:
            image = None
        history.append((turn.text, image))
        video = resolve_video(turn)
        if video is not None and not allow_video:
            raise HTTPException(status_code=400, detail="this scorer server does not accept video inputs")
        any_video = any_video or video is not None
        videos.append(video)
    return ScoreItem(
        history=history,
        videos=tuple(videos) if any_video else None,
        metadata=request.metadata,
    )


__all__ = ["decode_image", "request_to_item", "resolve_video"]
