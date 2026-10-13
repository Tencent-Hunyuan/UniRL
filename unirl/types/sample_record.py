"""Bounded sample records aligned through Sample lineage; see trainer/README.md."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from unirl.distributed.tensor import hydrate
from unirl.types.media_preview import build_media_preview_for_part
from unirl.types.primitives import Images, Texts
from unirl.types.sample import Sample
from unirl.types.sample_id import ancestor_id
from unirl.types.sampling import ARSamplingParams

logger = logging.getLogger(__name__)


@dataclass
class MediaItem:
    """One driver-resident image, video [C, T, H, W], or audio [L] / [L, C]."""

    image: Optional[Any] = None
    video: Optional[Any] = None
    audio: Optional[Any] = None
    audio_sample_rate: Optional[int] = None


@dataclass
class SampleRecord:
    """One generated row with aligned prompt, context, outputs, and rewards."""

    phase: str
    step: int
    track: str
    sample_id: str
    group_id: str
    root_id: str
    prompt: Optional[str]
    context_texts: List[str]
    output_text: Optional[str]
    media: Optional[MediaItem]
    reward: Optional[float]
    component_rewards: Dict[str, float]
    metadata: Dict[str, Any]


def build_sample_records(
    sample: Sample, *, part_index: int, step: int, phase: str, max_items: int
) -> List[SampleRecord]:
    """Observe the first K rows of one gen Part without hydrating training payloads."""
    part = sample.parts[part_index]
    limit = min(max_items, part.batch_size)
    if limit <= 0:
        return []
    try:
        conditioning = sample.conditioning_at(part_index)
    except (ValueError, IndexError) as exc:
        logger.warning("Sample observation has malformed lineage: %s", exc)
        conditioning = []
    texts = [value.texts for value in conditioning if isinstance(value, Texts)]
    prompts = texts[0] if texts else None
    input_image = next((value for value in conditioning if isinstance(value, Images)), None)
    preview = part.media_preview
    if preview is None:
        preview = build_media_preview_for_part(part=part, max_items=limit, prompts=prompts, input_image=input_image)
    outputs = part.primitives.get("text")
    rewards = hydrate(part.rewards[:limit]).detach().cpu().reshape(-1).tolist() if part.rewards is not None else None
    components = {
        name: hydrate(values[:limit]).detach().cpu().reshape(-1).tolist()
        for name, values in (part.component_rewards or {}).items()
    }
    records = []
    groups = part.group_ids
    for row, sample_id in enumerate(part.sample_ids[:limit]):
        media = None
        if preview is not None and row < len(preview):
            media = MediaItem(
                image=preview.images[row] if preview.images else None,
                video=preview.videos[row] if preview.videos else None,
                audio=preview.audios[row] if preview.audios else None,
                audio_sample_rate=preview.audio_sample_rate,
            )
        records.append(
            SampleRecord(
                phase=phase,
                step=step,
                track="ar" if isinstance(part.sampling_params, ARSamplingParams) else "diffusion",
                sample_id=sample_id,
                group_id=groups[row],
                root_id=ancestor_id(sample_id, 0),
                prompt=prompts[row] if prompts is not None else None,
                context_texts=[values[row] for values in texts[1:]],
                output_text=outputs.texts[row] if isinstance(outputs, Texts) else None,
                media=media,
                reward=float(rewards[row]) if rewards is not None else None,
                component_rewards={name: float(values[row]) for name, values in components.items()},
                metadata=part.metadata[row] if part.metadata else {},
            )
        )
    return records
