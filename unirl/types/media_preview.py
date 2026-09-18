"""MediaPreview — per-rollout wandb-agnostic media payload."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional

import torch

from unirl.distributed.tensor.batch import Batch, concat_field
from unirl.types.primitives import Audios, Images, Videos

if TYPE_CHECKING:
    from unirl.types.sample import Part


@dataclass
class MediaPreview(Batch):
    """Per-rollout wandb media preview payload that stays wandb-agnostic."""

    images: List[Any] = concat_field(default_factory=list)
    videos: List[Any] = concat_field(default_factory=list)
    audios: List[Any] = concat_field(default_factory=list)
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
    """Smallest sample count >= ``min_items`` landing on a TensorRef ref boundary."""
    from unirl.distributed.tensor import TensorRef

    total = len(decoded)
    want = max(1, min(int(min_items), total))
    if isinstance(decoded, Images):
        meta = decoded.packed_pixels
        cu = decoded.cu_seqlens
        if not isinstance(meta, TensorRef) or cu is None:
            return total
        sample_at_pixel = {int(v): i for i, v in enumerate(cu.tolist())}
        pixels = 0
        for size in meta.sizes:
            pixels += int(size)
            sample_idx = sample_at_pixel.get(pixels)
            if sample_idx is None:
                return total
            if sample_idx >= want:
                return sample_idx
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


def build_media_preview_for_part(
    *,
    part: "Part",
    max_items: int,
    prompts: Optional[List[str]] = None,
    input_image: Optional[Images] = None,
) -> Optional[MediaPreview]:
    """Build a wandb-bound :class:`MediaPreview` from one gen Part's decoded media."""
    decoded = part.primitives.get("image")
    if decoded is None:
        decoded = part.primitives.get("video")
    if not isinstance(decoded, (Images, Videos)):
        return None
    limit = max(1, int(max_items))

    from unirl.distributed.tensor import hydrate, map_tree

    prefix = _ref_aligned_prefix_len(decoded, limit)
    if 0 < prefix < len(decoded):
        decoded = decoded.slice(0, prefix)
    decoded = map_tree(decoded, hydrate)

    prompt_texts: List[str] = [str(p) for p in prompts] if prompts is not None else []
    rewards_flat: List[float] = []
    if part.rewards is not None and torch.is_tensor(part.rewards):
        rewards_flat = [float(v) for v in part.rewards.detach().cpu().reshape(-1).tolist()]

    images: List[Any] = []
    videos: List[Any] = []
    selected_indices: List[int] = []

    if isinstance(decoded, Images):
        from unirl.utils.media import hstack_pils, tensor_frame_to_pil

        per_sample = decoded.to_list()
        if not per_sample:
            return None
        input_pixels: Optional[List[Any]] = None
        if isinstance(input_image, Images):
            input_image = map_tree(input_image, hydrate)
            input_pixels = [image.pixels for image in input_image.to_list()]
        show_edit_pairs = input_pixels is not None and len(input_pixels) >= len(per_sample)
        for idx, image in enumerate(per_sample):
            if len(selected_indices) >= limit:
                break
            out_pil = tensor_frame_to_pil(image.pixels[:3])
            if show_edit_pairs:
                input_tensor = input_pixels[idx]
                in_pil = tensor_frame_to_pil(input_tensor[:3])
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

    audios_out: List[Any] = []
    audio_sr: Optional[int] = None
    decoded_audio = part.primitives.get("audio")
    if isinstance(decoded_audio, Audios):
        from unirl.distributed.tensor import hydrate, map_tree

        decoded_audio = map_tree(decoded_audio, hydrate)
        audio_list = decoded_audio.to_list()
        for idx in selected_indices:
            if idx < len(audio_list):
                audios_out.append(audio_list[idx].waveform.detach().cpu().float())
            else:
                audios_out.append(None)
        audio_sr = part.primitive_metadata.get("audio", {}).get("sample_rate")
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


__all__ = ["MediaPreview", "build_media_preview_for_part"]
