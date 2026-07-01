"""Request-side primitive extraction helpers the adapters' ``build_inputs`` call.

Pure and family-agnostic. The HI3 prompt construction (the task presets +
the per-prompt entry builder) lives with the HI3 sub-adapters in
``adapters/hi3.py``.
"""

from __future__ import annotations

from typing import List, Tuple

import PIL.Image

from unirl.types.primitives import Texts
from unirl.types.rollout_req import RolloutReq


def texts_from_req(req: RolloutReq) -> Texts:
    texts = req.primitives["text"]
    if len(texts.texts) != len(req.sample_ids):
        raise ValueError(f"prompt count {len(texts.texts)} != sample_ids count {len(req.sample_ids)}")
    return texts


def grouped_texts_from_req(req: RolloutReq, *, samples_per_prompt: int, caller: str) -> Tuple[List[str], int]:
    """Collapse contiguous sample-level prompt groups into vLLM-Omni prompt requests.

    Diffusion trainers expand each source prompt into ``samples_per_prompt``
    consecutive samples before dispatch. vLLM-Omni's native shape is instead
    one prompt request with ``num_outputs_per_prompt`` images. This helper
    validates that the current shard still contains whole, contiguous prompt
    groups, then returns one text per group.
    """
    texts = texts_from_req(req)
    spp = int(samples_per_prompt or 1)
    if spp < 1:
        raise ValueError(f"{caller}: samples_per_prompt must be >= 1, got {spp}")
    if spp == 1:
        return list(texts.texts), 1

    n = len(texts.texts)
    if n % spp != 0:
        raise RuntimeError(
            f"{caller}: shard sample count {n} is not divisible by samples_per_prompt={spp}; "
            "DP scatter split a prompt group. Use a batch/DP layout where each shard contains whole groups."
        )

    grouped: List[str] = []
    group_ids = list(req.group_ids or [])
    for start in range(0, n, spp):
        end = start + spp
        group_texts = texts.texts[start:end]
        if any(t != group_texts[0] for t in group_texts):
            raise RuntimeError(
                f"{caller}: prompt group at [{start}:{end}] is not contiguous/repeated; "
                "cannot map samples_per_prompt to num_outputs_per_prompt safely."
            )
        if group_ids:
            group = group_ids[start:end]
            if any(g != group[0] for g in group):
                raise RuntimeError(
                    f"{caller}: group_ids at [{start}:{end}] are not contiguous; "
                    "cannot map samples_per_prompt to num_outputs_per_prompt safely."
                )
        grouped.append(group_texts[0])
    return grouped, spp


def pil_images_from_req(req: RolloutReq, n: int) -> List[PIL.Image.Image]:
    """Extract ``req.primitives['image']`` (Images) as a list of PIL images.

    Returns an empty list when there's no image primitive. Asserts batch
    alignment when present; the conversion itself is :meth:`Images.to_pils`.
    """
    images = req.primitives.get("image")
    if images is None:
        return []
    if len(images) != n:
        raise ValueError(f"image batch {len(images)} != prompt count {n}")
    return images.to_pils()


__all__ = [
    "grouped_texts_from_req",
    "pil_images_from_req",
    "texts_from_req",
]
