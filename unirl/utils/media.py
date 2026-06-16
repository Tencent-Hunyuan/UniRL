"""Media conversion helpers shared across runtime layers.

This module is deliberately wandb-agnostic: it only converts between tensors
and PIL images. The wandb-side construction of ``wandb.Image`` / ``wandb.Video``
lives in ``unirl.utils.wandb_logger`` so that ``utils/media.py`` and the
``types/`` layer have zero dependency on wandb.
"""

from __future__ import annotations

from typing import Any, List

import torch


def tensor_frame_to_pil(frame: torch.Tensor) -> Any:
    """Convert one CHW image tensor to a PIL image."""
    from PIL import Image

    if frame.dim() != 3:
        raise ValueError(f"Expected CHW frame tensor, got shape={tuple(frame.shape)}")

    frame = frame.detach().float().cpu()
    # Old code: it will cause black images in wandb
    # if frame.max().item() > 1.0:
    #     frame = frame / 255.0
    # A temporary fix for wandb - TODO: check the dataflow of wandb media logging
    frame = frame.clamp(0.0, 1.0)
    if frame.shape[0] == 1:
        frame = frame.repeat(3, 1, 1)

    img = frame.permute(1, 2, 0).mul(255).byte().numpy()
    return Image.fromarray(img)


def tensor_to_pil(images: torch.Tensor) -> List[Any]:
    """Convert batched image/video tensors to PIL previews."""
    pil_images = []
    images = images.detach().cpu()

    if images.dim() == 5:
        frame_count = images.shape[2]
        images = images[:, :, frame_count // 2]

    for img in images:
        pil_images.append(tensor_frame_to_pil(img))

    return pil_images


def hstack_pils(left: Any, right: Any) -> Any:
    """Stack two PIL images side by side ("input | output"), matching heights."""
    try:
        from PIL import Image
    except Exception:
        return right
    if right.height != left.height:
        new_w = max(1, int(round(right.width * left.height / right.height)))
        right = right.resize((new_w, left.height))
    canvas = Image.new("RGB", (left.width + right.width, left.height))
    canvas.paste(left.convert("RGB"), (0, 0))
    canvas.paste(right.convert("RGB"), (left.width, 0))
    return canvas


__all__ = ["tensor_frame_to_pil", "tensor_to_pil", "hstack_pils"]
