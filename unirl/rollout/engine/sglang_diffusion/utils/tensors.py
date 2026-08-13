"""Low-level tensor / media / text mechanics the adapter conversion methods lean on."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch

from unirl.config.require import require

if TYPE_CHECKING:
    import numpy as np
    from PIL.Image import Image as PILImage


def fuse_encoder_outputs(value: Any) -> Optional[torch.Tensor]:
    """Fuse a text-conditioning field: token-level ``[B, seq, hidden]`` on dim -2, pooled ``[B, hidden]`` on -1."""
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
    """Best-effort coercion of arbitrary values into ``torch.Tensor``."""
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
    """Permute a decoded sample to channels-first: ``[C, H, W]`` for 3-D, ``[C, T, H, W]`` for 4-D."""
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
    """Read a SGLang ``result.samples`` payload into a canonical media tensor, clamped to ``[0, 1]``."""
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
