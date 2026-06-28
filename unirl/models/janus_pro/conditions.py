from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import torch

from unirl.distributed.tensor.batch import Batch, FieldKind, field
from unirl.types.conditions import TextTokenCondition


def _pad_seq_tensor(value: Optional[torch.Tensor], target_seq_len: int) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if value.dim() < 2 or int(value.shape[1]) == target_seq_len:
        return value
    if int(value.shape[1]) > target_seq_len:
        raise ValueError(
            f"Cannot pad Janus-Pro sequence tensor with seq_len={value.shape[1]} "
            f"to shorter target={target_seq_len}"
        )
    target_shape = list(value.shape)
    target_shape[1] = int(target_seq_len)
    out = value.new_zeros(target_shape)
    out[:, : int(value.shape[1]), ...] = value
    return out


def _cat_optional_tensors(values: Sequence[Optional[torch.Tensor]], *, field_name: str) -> Optional[torch.Tensor]:
    if all(v is None for v in values):
        return None
    if any(v is None for v in values):
        raise ValueError(f"JanusProARConditions.concat: mixed None/tensor values for {field_name}")
    return torch.cat([v for v in values if v is not None], dim=0)


def _stack_seq_rows(value: Any, target_seq_len: Optional[int]) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return _pad_seq_tensor(value, target_seq_len) if target_seq_len is not None else value
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"JanusProARConditions: expected images_seq_mask tensor/list, got {type(value).__name__}")

    rows = [row for row in value if row is not None]
    if len(rows) != len(value):
        raise ValueError("JanusProARConditions: mixed None/tensor values for images_seq_mask")
    if not rows:
        if target_seq_len is None:
            target_seq_len = 0
        return torch.empty((0, int(target_seq_len)), dtype=torch.bool)

    if target_seq_len is None:
        target_seq_len = max(int(row.reshape(-1).shape[0]) for row in rows)

    padded = []
    for row in rows:
        if not isinstance(row, torch.Tensor):
            raise TypeError(f"JanusProARConditions: images_seq_mask row must be tensor, got {type(row).__name__}")
        flat = row.reshape(-1)
        if int(flat.shape[0]) > int(target_seq_len):
            raise ValueError(
                f"Cannot pad Janus-Pro images_seq_mask row with seq_len={flat.shape[0]} "
                f"to shorter target={target_seq_len}"
            )
        out = flat.new_zeros((int(target_seq_len),))
        out[: int(flat.shape[0])] = flat
        padded.append(out)
    return torch.stack(padded, dim=0)


def _require_tensor(d: Dict[str, Any], key: str) -> torch.Tensor:
    value = d.get(key)
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"JanusProARConditions.from_dict: expected d[{key!r}] to be a "
            f"torch.Tensor, got {type(value).__name__ if value is not None else 'None'}"
        )
    return value


@dataclass
class JanusProARConditions(Batch):
    """Conditions for Janus-Pro multimodal understanding.

    The official processor already pads image slots to a rectangular
    ``[B, N, ...]`` layout, so the image tensors can travel as ordinary
    CONCAT fields. ``images_seq_mask`` marks the image placeholder tokens in the
    prompt, and ``images_emb_mask`` marks valid image embeddings.
    """

    prompt: Optional[TextTokenCondition] = field(kind=FieldKind.CONCAT, default=None)
    pixel_values: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)
    images_seq_mask: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)
    images_emb_mask: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)

    @classmethod
    def concat(cls, items: Sequence["JanusProARConditions"]) -> "JanusProARConditions":
        if not items or len(items) == 1:
            return Batch.concat.__func__(cls, items)

        prompts = [item.prompt for item in items]
        if any(not isinstance(prompt, TextTokenCondition) for prompt in prompts):
            return Batch.concat.__func__(cls, items)

        prompt = TextTokenCondition.concat([prompt for prompt in prompts if prompt is not None])
        target_seq_len = None
        if prompt.input_ids is not None and prompt.input_ids.dim() >= 2:
            target_seq_len = int(prompt.input_ids.shape[1])
        elif prompt.attention_mask is not None and prompt.attention_mask.dim() >= 2:
            target_seq_len = int(prompt.attention_mask.shape[1])

        images_seq_masks = [item.images_seq_mask for item in items]
        if target_seq_len is not None:
            images_seq_masks = [_pad_seq_tensor(mask, target_seq_len) for mask in images_seq_masks]

        return cls(
            prompt=prompt,
            pixel_values=_cat_optional_tensors([item.pixel_values for item in items], field_name="pixel_values"),
            images_seq_mask=_cat_optional_tensors(images_seq_masks, field_name="images_seq_mask"),
            images_emb_mask=_cat_optional_tensors(
                [item.images_emb_mask for item in items],
                field_name="images_emb_mask",
            ),
        )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "JanusProARConditions":
        prompt = d.get("prompt")
        if not isinstance(prompt, TextTokenCondition):
            raise TypeError(
                "JanusProARConditions.from_dict: expected d['prompt'] to be a "
                f"TextTokenCondition, got {type(prompt).__name__ if prompt is not None else 'None'}"
            )
        target_seq_len = None
        if prompt.input_ids is not None and prompt.input_ids.dim() >= 2:
            target_seq_len = int(prompt.input_ids.shape[1])
        elif prompt.attention_mask is not None and prompt.attention_mask.dim() >= 2:
            target_seq_len = int(prompt.attention_mask.shape[1])
        pixel_values = _require_tensor(d, "pixel_values")
        images_emb_mask = _require_tensor(d, "images_emb_mask")
        return cls(
            prompt=prompt,
            pixel_values=pixel_values,
            images_seq_mask=_stack_seq_rows(d.get("images_seq_mask"), target_seq_len),
            images_emb_mask=images_emb_mask,
        )

    def to_dict(self) -> Dict[str, Any]:
        if self.prompt is None:
            raise ValueError("JanusProARConditions.to_dict: prompt field is None")
        if not isinstance(self.prompt, TextTokenCondition):
            raise TypeError(
                f"JanusProARConditions.to_dict: prompt must be TextTokenCondition, got {type(self.prompt).__name__}"
            )
        if self.pixel_values is None:
            raise ValueError("JanusProARConditions.to_dict: pixel_values field is None")
        if not isinstance(self.pixel_values, torch.Tensor):
            raise TypeError(
                "JanusProARConditions.to_dict: pixel_values must be torch.Tensor, "
                f"got {type(self.pixel_values).__name__}"
            )
        if self.images_seq_mask is None:
            raise ValueError("JanusProARConditions.to_dict: images_seq_mask field is None")
        if not isinstance(self.images_seq_mask, torch.Tensor):
            raise TypeError(
                "JanusProARConditions.to_dict: images_seq_mask must be torch.Tensor, "
                f"got {type(self.images_seq_mask).__name__}"
            )
        if self.images_emb_mask is None:
            raise ValueError("JanusProARConditions.to_dict: images_emb_mask field is None")
        if not isinstance(self.images_emb_mask, torch.Tensor):
            raise TypeError(
                "JanusProARConditions.to_dict: images_emb_mask must be torch.Tensor, "
                f"got {type(self.images_emb_mask).__name__}"
            )
        images_seq_mask: Any = self.images_seq_mask
        if isinstance(images_seq_mask, torch.Tensor):
            images_seq_mask = [images_seq_mask[i].clone() for i in range(int(images_seq_mask.shape[0]))]
        return {
            "prompt": self.prompt,
            "pixel_values": self.pixel_values,
            "images_seq_mask": images_seq_mask,
            "images_emb_mask": self.images_emb_mask,
        }


@dataclass
class JanusProImageARConditions(Batch):
    """Conditions for Janus-Pro Text -> Image autoregressive token generation."""

    prompt: Optional[TextTokenCondition] = field(kind=FieldKind.CONCAT, default=None)
    cfg_prompt: Optional[TextTokenCondition] = field(kind=FieldKind.CONCAT, default=None)
    cfg_weight: float = field(kind=FieldKind.SHARED, default=5.0)

    @classmethod
    def concat(cls, items: Sequence["JanusProImageARConditions"]) -> "JanusProImageARConditions":
        if not items or len(items) == 1:
            return Batch.concat.__func__(cls, items)

        prompts = [item.prompt for item in items]
        cfg_prompts = [item.cfg_prompt for item in items]
        if any(not isinstance(prompt, TextTokenCondition) for prompt in prompts):
            return Batch.concat.__func__(cls, items)
        if any(not isinstance(prompt, TextTokenCondition) for prompt in cfg_prompts):
            return Batch.concat.__func__(cls, items)

        return cls(
            prompt=TextTokenCondition.concat([prompt for prompt in prompts if prompt is not None]),
            cfg_prompt=TextTokenCondition.concat([prompt for prompt in cfg_prompts if prompt is not None]),
            cfg_weight=float(items[0].cfg_weight),
        )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "JanusProImageARConditions":
        prompt = d.get("prompt")
        if not isinstance(prompt, TextTokenCondition):
            raise TypeError(
                "JanusProImageARConditions.from_dict: expected d['prompt'] to be a "
                f"TextTokenCondition, got {type(prompt).__name__ if prompt is not None else 'None'}"
            )
        cfg_prompt = d.get("cfg_prompt")
        if not isinstance(cfg_prompt, TextTokenCondition):
            raise TypeError(
                "JanusProImageARConditions.from_dict: expected d['cfg_prompt'] to be a "
                f"TextTokenCondition, got {type(cfg_prompt).__name__ if cfg_prompt is not None else 'None'}"
            )
        return cls(
            prompt=prompt,
            cfg_prompt=cfg_prompt,
            cfg_weight=float(d.get("cfg_weight", 5.0)),
        )

    def to_dict(self) -> Dict[str, Any]:
        if self.prompt is None:
            raise ValueError("JanusProImageARConditions.to_dict: prompt field is None")
        if not isinstance(self.prompt, TextTokenCondition):
            raise TypeError(
                "JanusProImageARConditions.to_dict: prompt must be TextTokenCondition, "
                f"got {type(self.prompt).__name__}"
            )
        if self.cfg_prompt is None:
            raise ValueError("JanusProImageARConditions.to_dict: cfg_prompt field is None")
        if not isinstance(self.cfg_prompt, TextTokenCondition):
            raise TypeError(
                "JanusProImageARConditions.to_dict: cfg_prompt must be TextTokenCondition, "
                f"got {type(self.cfg_prompt).__name__}"
            )
        return {
            "prompt": self.prompt,
            "cfg_prompt": self.cfg_prompt,
            "cfg_weight": float(self.cfg_weight),
        }


__all__ = ["JanusProARConditions", "JanusProImageARConditions"]
