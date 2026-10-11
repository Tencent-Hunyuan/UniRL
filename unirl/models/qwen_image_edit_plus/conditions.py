"""QwenImageEditPlusConditions — typed conditions for Qwen-Image-Edit-Plus."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Dict, Optional

import torch

from unirl.distributed.tensor.batch import Batch, FieldKind, concat_field, field
from unirl.types.conditions import Condition, Modality, TextEmbedCondition


@dataclass
class QwenImageEditPlusLatentCondition(Condition):
    """Per-sample ordered ``[C, H_i, W_i]`` source-latent rows."""

    modality: ClassVar[Modality] = Modality.IMAGE
    latents: list[list[torch.Tensor]] = concat_field(default_factory=list)

    def __post_init__(self) -> None:
        normalized = []
        for row, value in enumerate(self.latents):
            if isinstance(value, torch.Tensor):
                values = [value]
            elif isinstance(value, (list, tuple)):
                values = list(value)
            else:
                raise TypeError(
                    f"QwenImageEditPlusLatentCondition.latents[{row}] must be a tensor row, got {type(value).__name__}."
                )
            if any(not isinstance(latent, torch.Tensor) for latent in values):
                raise TypeError(f"QwenImageEditPlusLatentCondition.latents[{row}] must contain tensors.")
            normalized.append(values)
        self.latents = normalized

    def __len__(self) -> int:
        return len(self.latents)


@dataclass
class QwenImageEditPlusConditions(Batch):
    """Typed conditions container for Qwen-Image-Edit-Plus diffusion."""

    text: Optional[TextEmbedCondition] = field(kind=FieldKind.CONCAT, default=None)
    negative_text: Optional[TextEmbedCondition] = field(kind=FieldKind.CONCAT, default=None)
    image_latent: Optional[QwenImageEditPlusLatentCondition] = field(kind=FieldKind.CONCAT, default=None)

    @classmethod
    def from_dict(cls, d: Dict[str, Condition]) -> "QwenImageEditPlusConditions":
        """Build from the generic ``Conditions`` dict shape."""
        text = d.get("text")
        if not isinstance(text, TextEmbedCondition):
            raise TypeError(
                f"QwenImageEditPlusConditions.from_dict: expected d['text'] to be a "
                f"TextEmbedCondition, got "
                f"{type(text).__name__ if text is not None else 'None'}"
            )
        negative_text = d.get("negative_text")
        if negative_text is not None and not isinstance(negative_text, TextEmbedCondition):
            raise TypeError(
                f"QwenImageEditPlusConditions.from_dict: expected d['negative_text'] to be a "
                f"TextEmbedCondition or absent, got {type(negative_text).__name__}"
            )
        image_latent = d.get("image_latent")
        if image_latent is not None and not isinstance(image_latent, QwenImageEditPlusLatentCondition):
            raise TypeError(
                f"QwenImageEditPlusConditions.from_dict: expected d['image_latent'] to be an "
                f"QwenImageEditPlusLatentCondition or absent, got {type(image_latent).__name__}"
            )
        return cls(text=text, negative_text=negative_text, image_latent=image_latent)

    def to_dict(self) -> Dict[str, Condition]:
        """Convert back to the generic ``Conditions`` dict shape."""
        if self.text is None:
            raise ValueError("QwenImageEditPlusConditions.to_dict: text field is None")
        out: Dict[str, Condition] = {"text": self.text}
        if self.negative_text is not None:
            out["negative_text"] = self.negative_text
        if self.image_latent is not None:
            out["image_latent"] = self.image_latent
        return out


__all__ = ["QwenImageEditPlusConditions", "QwenImageEditPlusLatentCondition"]
