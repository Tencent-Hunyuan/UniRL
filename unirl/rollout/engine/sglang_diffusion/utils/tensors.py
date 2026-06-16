"""Low-level tensor / media / text mechanics the adapter conversion methods lean on.

Pure: no SGLang import, no engine state. These are the generic, model-agnostic
helpers — the *customizable* conversion steps live as overridable methods on the
adapters (``build_segment`` / ``build_decoded`` / ``build_condition``); these are
the bits those methods call. Ported from the old engine's ``_text_fusion`` and
``_sample_decode`` modules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch

from unirl.config.require import require

if TYPE_CHECKING:
    import numpy as np
    from PIL.Image import Image as PILImage


def fuse_encoder_outputs(value: Any) -> Optional[torch.Tensor]:
    """Reduce a per-result text-conditioning field to a single tensor.

    SGLang may surface a text-conditioning field as a single tensor or a
    list/tuple — one entry per text encoder (SDXL/SD3/FLUX use several). The
    concatenation axis follows the fusion convention:

    - ≥3-D (token-level, ``[B, seq, hidden]``) → ``cat(dim=-2)`` along seq
      (SD3/FLUX merging CLIP and T5 token streams).
    - 2-D (pooled, ``[B, hidden]``) → ``cat(dim=-1)`` along hidden
      (SDXL stacking CLIP-L and CLIP-G pooled vectors).

    Returns ``None`` when the input is missing or holds no tensor.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        tensors = [item for item in value if torch.is_tensor(item)]
        if not tensors:
            return None
        if len(tensors) == 1:
            value = tensors[0]
        elif tensors[0].dim() >= 3:
            value = torch.cat(tensors, dim=-2)
        else:
            value = torch.cat(tensors, dim=-1)
    if not torch.is_tensor(value):
        return None
    return value


def tensorize(value: Any) -> Optional[torch.Tensor]:
    """Best-effort coercion of arbitrary values into ``torch.Tensor``.

    Handles tensors / numpy arrays / PIL images and homogeneous lists/tuples
    of the same. Returns ``None`` when no sensible conversion exists, instead
    of raising — callers decide whether absence is an error.
    """
    if value is None:
        return None
    if torch.is_tensor(value):
        return value
    try:
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
    except Exception:
        pass
    return None


def normalize_media(sample: torch.Tensor) -> torch.Tensor:
    """Permute a decoded sample to channels-first canonical layout.

    Recognized inputs: ``[C,H,W]``, ``[H,W,C]``, ``[C,T,H,W]``, ``[T,C,H,W]``,
    ``[T,H,W,C]``. Returns ``[C,H,W]`` for 3-D image, ``[C,T,H,W]`` for 4-D
    video. Raises for unrecognized layouts.
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
    sample: "torch.Tensor | np.ndarray | PILImage | tuple | list | None",
) -> Optional[torch.Tensor]:
    """Read a SGLang ``result.samples`` payload into a canonical media tensor.

    Handles the ``(video, audio)`` 2-tuple wrap from SGLang's
    ``attach_audio_to_video_sample`` (audio is dropped — ``result.audio`` is the
    canonical channel). Returns ``None`` when no recognizable sample is present.
    Floating-point output is clamped to ``[0, 1]`` (decoded-images contract).
    """
    if isinstance(sample, (tuple, list)) and len(sample) == 2:
        sample = sample[0]
    sample_tensor = tensorize(sample)
    if sample_tensor is None:
        return None
    canonical = normalize_media(sample_tensor.detach().cpu())
    if canonical.is_floating_point():
        canonical = canonical.clamp(0.0, 1.0)
    return canonical


__all__ = ["fuse_encoder_outputs", "tensorize", "normalize_media", "decode_sample"]
