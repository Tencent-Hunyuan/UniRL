"""MediaPreview — per-rollout wandb-agnostic media payload.

Carries PIL images and raw 4D video tensors keyed to per-sample prompts /
rewards for wandb logging. Lives in its own module so the type survives
independently of the legacy ``RolloutSamples`` container (which used to
own it). Consumed via ``RolloutTrack.media_preview``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional

import torch

from unirl.distributed.tensor.batch import Batch, concat_field
from unirl.types.primitives import Images, Videos

if TYPE_CHECKING:
    from unirl.types.rollout_req import RolloutReq
    from unirl.types.rollout_resp import RolloutTrack


@dataclass
class MediaPreview(Batch):
    """Per-rollout wandb media preview payload that stays wandb-agnostic.

    ``images`` carries PIL images (one per sample; image models or
    middle-frame extraction for video models that want still previews).
    ``videos`` carries raw 4D ``(C, T, H, W)`` CPU tensors with values in
    ``[0, 1]`` — NOT pre-built ``wandb.Video`` objects. The wandb-side
    encoding (mp4 / fps / caption) is owned by
    :func:`unirl.utils.wandb_logger.UniRLWandBLogger.log_generated_media`
    so this dataclass + the ``utils/media.py`` helpers carry zero wandb
    dependency.

    Declaring the four parallel lists as ``concat_field`` lets
    ``Batch.concat`` auto-merge per-shard previews (lists extended) and
    ``Batch.slice(0, n)`` naturally cap the payload size.

    Three valid states (enforced in ``__post_init__``):

    - **image-only**: ``images`` populated, ``videos`` empty
    - **video-only**: ``videos`` populated, ``images`` empty
    - **image+video**: both populated; both must agree on per-sample
      length and the parallel ``prompts`` / ``rewards`` lists.

    Whichever side is non-empty defines the canonical batch size; every
    non-empty parallel list must agree with that size. ``__len__`` and
    ``batch_size`` mirror this — without the override, the default
    ``Batch.batch_size`` anchors on the first concat field
    (``images``), which would silently leave ``videos`` un-sliced when
    ``len(videos) > 0`` and ``len(images) == 0``.
    """

    images: List[Any] = concat_field(default_factory=list)
    videos: List[Any] = concat_field(default_factory=list)
    prompts: List[str] = concat_field(default_factory=list)
    rewards: List[float] = concat_field(default_factory=list)

    def __post_init__(self) -> None:
        n = len(self.images) if self.images else len(self.videos)
        for name, val in (
            ("images", self.images),
            ("videos", self.videos),
            ("prompts", self.prompts),
            ("rewards", self.rewards),
        ):
            if val and len(val) != n:
                raise ValueError(
                    f"MediaPreview: {name!r} has {len(val)} entries but the "
                    f"canonical batch size (from images / videos) is {n}. "
                    f"All non-empty parallel lists must agree."
                )

    @property
    def batch_size(self) -> int:
        return len(self.images) if self.images else len(self.videos)

    def __len__(self) -> int:
        return self.batch_size

    def is_empty(self) -> bool:
        return not self.images and not self.videos


def build_media_preview_for_track(
    *,
    req: "RolloutReq",
    track: "RolloutTrack",
    max_items: int,
) -> Optional[MediaPreview]:
    """Build a wandb-bound :class:`MediaPreview` from one track's decoded media.

    Two parallel modality paths, mirroring the legacy
    ``RolloutResponse.attach_media_preview``:

    - **Image path** (``isinstance(track.decoded, Images)``): unbinds
      ``Images.pixels`` along batch dim into per-sample 3D ``[C, H, W]``
      tensors and converts each to PIL via ``tensor_frame_to_pil`` (the
      wandb boundary). Slices to the first 3 channels first — drops
      alpha / model-specific 4th channel so wandb gets RGB.
    - **Video path** (``isinstance(track.decoded, Videos)``): reads
      per-sample 4D ``[C, T, H, W]`` CPU ``float32`` tensors via
      ``Videos.to_list()`` + ``permute(1, 0, 2, 3)``; keeps them raw,
      NOT pre-built ``wandb.Video`` (encoding is owned by
      ``UniRLWandBLogger.log_generated_media``).

    Returns ``None`` when the track's ``decoded`` is neither ``Images``
    nor ``Videos`` (e.g. text track) or when nothing is selected.
    """
    decoded = track.decoded
    if not isinstance(decoded, (Images, Videos)):
        return None
    limit = max(1, int(max_items))

    text_prim = req.primitives.get("text")
    prompt_texts: List[str] = (
        list(text_prim.texts) if text_prim is not None and getattr(text_prim, "texts", None) else []
    )
    rewards_flat: List[float] = []
    if track.rewards is not None and torch.is_tensor(track.rewards):
        rewards_flat = [float(v) for v in track.rewards.detach().cpu().reshape(-1).tolist()]

    images: List[Any] = []
    videos: List[Any] = []
    selected_indices: List[int] = []

    if isinstance(decoded, Images):
        from unirl.utils.media import tensor_frame_to_pil

        pixels = decoded.pixels
        if pixels is None:
            return None
        for idx in range(int(pixels.shape[0])):
            if len(selected_indices) >= limit:
                break
            img = pixels[idx]
            images.append(tensor_frame_to_pil(img[:3]))
            selected_indices.append(idx)
    else:
        per_sample = decoded.to_list()
        for idx, video in enumerate(per_sample):
            if len(selected_indices) >= limit:
                break
            frames = video.frames
            if frames.dim() != 4:
                continue
            videos.append(frames.permute(1, 0, 2, 3).contiguous().detach().cpu().to(dtype=torch.float32))
            selected_indices.append(idx)

    if not selected_indices:
        return None

    prompts_out = [str(prompt_texts[i]) if i < len(prompt_texts) else "" for i in selected_indices]
    reward_values = [float(rewards_flat[i]) if i < len(rewards_flat) else 0.0 for i in selected_indices]
    return MediaPreview(
        images=images,
        videos=videos,
        prompts=prompts_out,
        rewards=reward_values,
    )


__all__ = ["MediaPreview", "build_media_preview_for_track"]
