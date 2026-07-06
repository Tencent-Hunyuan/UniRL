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
    # Per-sample audio waveforms (mono [L] float32 CPU tensors) for muxing into
    # the mp4 upload. Parallel to ``videos`` — same length. ``None``/empty for
    # non-audio tracks.
    audios: List[Any] = concat_field(default_factory=list)
    # Source sample rate of the waveforms in ``audios``. Batch-shared (one rate
    # per preview). None when audios is empty.
    audio_sample_rate: Optional[int] = None
    prompts: List[str] = concat_field(default_factory=list)
    rewards: List[float] = concat_field(default_factory=list)

    def __post_init__(self) -> None:
        n = len(self.images) if self.images else len(self.videos)
        for name, val in (
            ("images", self.images),
            ("videos", self.videos),
            ("audios", self.audios),
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


def _ref_aligned_prefix_len(decoded: Any, min_items: int) -> int:
    """Smallest sample count >= ``min_items`` landing on a TensorRef ref boundary.

    ``decoded`` reaches the driver dehydrated: its tensor leaf (``Images.pixels``
    / ``Videos.frames``) is a ``TensorRef`` whose refs partition the batch by DP
    shard, and ``TensorRef`` only supports ref-boundary slicing. The cheapest
    preview prefix is the first shard boundary covering ``min_items`` samples, so
    media logging hydrates one shard instead of the full decoded batch.
    ``Videos`` ref sizes count frames (PACKED), so shard frame boundaries are
    mapped back to sample indices via the driver-side ``cu_seqlens``. Returns the
    full batch size when the leaf is already a real tensor (nothing to save) or a
    boundary cannot be mapped.
    """
    from unirl.distributed.tensor import TensorRef

    total = len(decoded)
    want = max(1, min(int(min_items), total))
    if isinstance(decoded, Images):
        meta = decoded.pixels
        if not isinstance(meta, TensorRef):
            return total
        rows = 0
        for size in meta.sizes:
            rows += int(size)
            if rows >= want:
                return rows
        return total
    meta = decoded.frames
    cu = decoded.cu_seqlens
    if not isinstance(meta, TensorRef) or cu is None:
        return total
    sample_at_frame = {int(v): i for i, v in enumerate(cu.tolist())}
    frames = 0
    for size in meta.sizes:
        frames += int(size)
        sample_idx = sample_at_frame.get(frames)
        if sample_idx is None:
            return total
        if sample_idx >= want:
            return sample_idx
    return total


def build_media_preview_for_track(
    *,
    req: "RolloutReq",
    track: "RolloutTrack",
    max_items: int,
    prompts: Optional[List[str]] = None,
) -> Optional[MediaPreview]:
    """Build a wandb-bound :class:`MediaPreview` from one track's decoded media.

    ``prompts`` (when given) is a per-sample caption list already aligned 1:1
    with this track's samples — pass it for multi-track recipes (PE / unified)
    whose ``req.primitives["text"]`` holds only the original prompts (shorter
    than the expanded track). When ``None`` the captions fall back to
    ``req.primitives["text"]``, which is correct for the single-track diffusion
    / AR path where ``_build_req`` already expands text 1:1 with samples.

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

    # ``decoded`` reaches the driver dehydrated (its tensor leaf is a
    # ``TensorRef`` proxy partitioned by DP shard). Slice to the smallest
    # ref-boundary prefix covering ``limit`` samples, then hydrate only that
    # shard so we pull one shard instead of the full decoded batch. Both steps
    # are no-ops when the leaf is already a real tensor (e.g. unit tests).
    from unirl.distributed.tensor import hydrate, map_tree

    prefix = _ref_aligned_prefix_len(decoded, limit)
    if 0 < prefix < len(decoded):
        decoded = decoded.slice(0, prefix)
    decoded = map_tree(decoded, hydrate)

    if prompts is not None:
        prompt_texts: List[str] = [str(p) for p in prompts]
    else:
        text_prim = req.primitives.get("text")
        prompt_texts = list(text_prim.texts) if text_prim is not None and getattr(text_prim, "texts", None) else []
    rewards_flat: List[float] = []
    if track.rewards is not None and torch.is_tensor(track.rewards):
        rewards_flat = [float(v) for v in track.rewards.detach().cpu().reshape(-1).tolist()]

    images: List[Any] = []
    videos: List[Any] = []
    selected_indices: List[int] = []

    if isinstance(decoded, Images):
        from unirl.utils.media import hstack_pils, tensor_frame_to_pil

        pixels = decoded.pixels
        if pixels is None:
            return None
        # it2i carries the per-sample input image in req.primitives["image"]; pair
        # it beside the output when it covers the (possibly shard-prefixed) batch.
        input_pixels = None
        image_prim = req.primitives.get("image")
        if isinstance(image_prim, Images) and image_prim.pixels is not None:
            input_pixels = image_prim.pixels
        show_edit_pairs = input_pixels is not None and int(input_pixels.shape[0]) >= int(pixels.shape[0])
        for idx in range(int(pixels.shape[0])):
            if len(selected_indices) >= limit:
                break
            out_pil = tensor_frame_to_pil(pixels[idx][:3])
            if show_edit_pairs:
                in_pil = tensor_frame_to_pil(input_pixels[idx][:3])
                images.append(hstack_pils(in_pil, out_pil))
            else:
                images.append(out_pil)
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

    # T2AV: extract per-sample audio waveforms for muxing into the mp4 upload.
    audios_out: List[Any] = []
    audio_sr: Optional[int] = None
    decoded_audio = getattr(track, "decoded_audio", None)
    if decoded_audio is not None and hasattr(decoded_audio, "to_list"):
        from unirl.distributed.tensor import hydrate, map_tree

        decoded_audio = map_tree(decoded_audio, hydrate)
        audio_list = decoded_audio.to_list()
        for idx in selected_indices:
            if idx < len(audio_list):
                audios_out.append(audio_list[idx].waveform.detach().cpu().float())
            else:
                audios_out.append(None)
        audio_sr = getattr(track, "audio_sample_rate", None)
        # Drop audio if none of the selected samples have it
        if all(a is None for a in audios_out):
            audios_out = []
            audio_sr = None

    prompts_out = [str(prompt_texts[i]) if i < len(prompt_texts) else "" for i in selected_indices]
    reward_values = [float(rewards_flat[i]) if i < len(rewards_flat) else 0.0 for i in selected_indices]
    return MediaPreview(
        images=images,
        videos=videos,
        audios=audios_out,
        audio_sample_rate=int(audio_sr) if audio_sr is not None else None,
        prompts=prompts_out,
        rewards=reward_values,
    )


__all__ = ["MediaPreview", "build_media_preview_for_track"]
