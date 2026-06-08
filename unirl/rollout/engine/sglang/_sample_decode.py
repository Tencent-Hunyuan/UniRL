"""Decoded sample normalization for SGLang ``OutputBatch`` results (engine-private).

SGLang returns generated samples as bare tensors / numpy arrays / PIL
images / 2-tuples (video, audio), without a typed wrapper. These helpers
coerce the payload into a canonical channels-first tensor that the
``RolloutResp`` typed primitives can consume.

Underscore-prefixed module name marks this as private to the sglang
rollout-engine package; the only consumer is
:mod:`unirl.rollout.engine.sglang.response`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch

from unirl.config.require import require

if TYPE_CHECKING:
    import numpy as np
    from PIL.Image import Image as PILImage


def _tensorize(value: Any) -> Optional[torch.Tensor]:
    """Best-effort coercion of arbitrary values into ``torch.Tensor``.

    Handles tensors / numpy arrays / PIL images and homogeneous lists/tuples
    of the same. Returns ``None`` when the value is absent or of an
    unrecognized type — callers decide whether absence is an error. A
    recognized payload that cannot be converted (e.g. a ragged image list)
    raises, surfacing the malformed input rather than masking it as ``None``.
    """
    if value is None:
        return None
    if torch.is_tensor(value):
        return value

    import numpy as np
    from PIL import Image

    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    if isinstance(value, Image.Image):
        return torch.from_numpy(np.array(value))
    if isinstance(value, (list, tuple)) and value:
        if all(torch.is_tensor(v) for v in value):
            return torch.stack([v.detach() for v in value], dim=0)
        if all(isinstance(v, np.ndarray) for v in value):
            return torch.from_numpy(np.stack(value, axis=0))
        if all(isinstance(v, Image.Image) for v in value):
            return torch.from_numpy(np.stack([np.array(v) for v in value], axis=0))

    return None


def normalize_media(sample: torch.Tensor) -> torch.Tensor:
    """Permute a decoded sample to channels-first canonical layout.

    Recognized inputs: ``[C,H,W]``, ``[H,W,C]``, ``[C,T,H,W]``,
    ``[T,C,H,W]``, ``[T,H,W,C]``. Returns ``[C,H,W]`` for 3D image,
    ``[C,T,H,W]`` for 4D video. Raises ``ValueError`` for unrecognized
    layouts.
    """
    if sample.dim() == 3:
        if sample.shape[0] in (1, 3, 4):
            return sample
        require(sample.shape[-1] in (1, 3, 4), f"Unrecognized 3D media layout: {tuple(sample.shape)}")
        return sample.permute(2, 0, 1)

    require(sample.dim() == 4, f"Unrecognized media tensor dim {sample.dim()}: shape={tuple(sample.shape)}")

    if sample.shape[0] in (1, 3, 4):
        return sample
    if sample.shape[1] in (1, 3, 4):
        return sample.permute(1, 0, 2, 3)
    require(sample.shape[-1] in (1, 3, 4), f"Unrecognized 4D media layout: {tuple(sample.shape)}")
    return sample.permute(3, 0, 1, 2)


def decode_sample(
    sample: torch.Tensor | np.ndarray | PILImage | tuple | list | None,
) -> Optional[torch.Tensor]:
    """Read a SGLang ``result.samples`` payload into a canonical media tensor.

    Handles the ``(video, audio)`` 2-tuple wrap from SGLang's
    ``attach_audio_to_video_sample`` (audio is dropped — ``result.audio`` is
    the canonical channel for that). Returns ``None`` when no recognizable
    sample is present.
    """
    if isinstance(sample, (tuple, list)) and len(sample) == 2:
        sample = sample[0]
    sample_tensor = _tensorize(sample)
    if sample_tensor is None:
        return None
    canonical = normalize_media(sample_tensor.detach().cpu())
    # decoded_images contract: [C,H,W] floats in [0, 1] (matches FSDP path).
    # VAEs routinely overshoot [0, 1] by a few percent — clamp here so
    # consumers don't have to guess the range.
    if canonical.is_floating_point():
        canonical = canonical.clamp(0.0, 1.0)
    return canonical
