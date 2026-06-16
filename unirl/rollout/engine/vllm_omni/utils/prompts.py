"""Request-side primitive extraction helpers the adapters' ``build_inputs`` call.

Pure and family-agnostic. The HI3 prompt construction (the task presets +
the per-prompt entry builder) lives with the HI3 sub-adapters in
``adapters/hi3.py``.
"""

from __future__ import annotations

from typing import List

import PIL.Image

from unirl.types.primitives import Texts
from unirl.types.rollout_req import RolloutReq


def texts_from_req(req: RolloutReq) -> Texts:
    texts = req.primitives["text"]
    if len(texts.texts) != len(req.sample_ids):
        raise ValueError(f"prompt count {len(texts.texts)} != sample_ids count {len(req.sample_ids)}")
    return texts


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
    "pil_images_from_req",
    "texts_from_req",
]
