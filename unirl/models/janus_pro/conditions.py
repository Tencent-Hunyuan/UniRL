from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import torch

from unirl.distributed.tensor.batch import Batch, FieldKind, field
from unirl.types.conditions import TextTokenCondition


def _finite_cfg_weight(value: Any, *, where: str) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{where} must be a finite number, got {value!r}.") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"{where} must be finite, got {value!r}.")
    return normalized


def _pad_seq_tensor(value: torch.Tensor, target_seq_len: int) -> torch.Tensor:
    if value.dim() != 2:
        raise ValueError(f"Janus-Pro sequence tensors must be rank 2, got shape={tuple(value.shape)}.")
    if value.shape[1] == target_seq_len:
        return value
    if value.shape[1] > target_seq_len:
        raise ValueError(
            f"Cannot pad Janus-Pro sequence tensor with seq_len={value.shape[1]} to shorter target={target_seq_len}"
        )
    target_shape = list(value.shape)
    target_shape[1] = target_seq_len
    out = value.new_zeros(target_shape)
    out[:, : value.shape[1], ...] = value
    return out


def _stack_seq_rows(value: Any, target_seq_len: int) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return _pad_seq_tensor(value, target_seq_len)
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"JanusProARConditions: expected images_seq_mask tensor/list, got {type(value).__name__}")

    rows = list(value)
    if not rows:
        return torch.empty((0, target_seq_len), dtype=torch.bool)
    if any(not isinstance(row, torch.Tensor) for row in rows):
        bad = next(row for row in rows if not isinstance(row, torch.Tensor))
        raise TypeError(f"JanusProARConditions: images_seq_mask row must be tensor, got {type(bad).__name__}")

    padded = []
    for row in rows:
        flat = row.reshape(-1)
        if flat.shape[0] > target_seq_len:
            raise ValueError(
                f"Cannot pad Janus-Pro images_seq_mask row with seq_len={flat.shape[0]} "
                f"to shorter target={target_seq_len}"
            )
        out = flat.new_zeros((target_seq_len,))
        out[: flat.shape[0]] = flat
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


def _optional_tensor(d: Dict[str, Any], key: str) -> Optional[torch.Tensor]:
    value = d.get(key)
    if value is not None and not isinstance(value, torch.Tensor):
        raise TypeError(f"Janus-Pro conditions require d[{key!r}] to be a tensor or None.")
    return value


def _require_text_condition(d: Dict[str, Any], key: str) -> TextTokenCondition:
    value = d.get(key)
    if not isinstance(value, TextTokenCondition):
        raise TypeError(
            f"Janus-Pro conditions require d[{key!r}] to be a TextTokenCondition, "
            f"got {type(value).__name__ if value is not None else 'None'}"
        )
    return value


def _prompt_seq_len(prompt: TextTokenCondition) -> int:
    for value in (prompt.input_ids, prompt.attention_mask):
        if value is not None:
            if value.dim() != 2:
                raise ValueError(f"Janus-Pro prompt tensors must be rank 2, got shape={tuple(value.shape)}.")
            return value.shape[1]
    raise ValueError("Janus-Pro prompt requires input_ids or attention_mask.")


@dataclass
class JanusProARConditions(Batch):
    """Batched prompt and cached-or-raw image conditioning for Janus-Pro."""

    prompt: TextTokenCondition = field(kind=FieldKind.CONCAT)
    images_seq_mask: torch.Tensor = field(kind=FieldKind.CONCAT)
    images_emb_mask: torch.Tensor = field(kind=FieldKind.CONCAT)
    pixel_values: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)
    image_embeds: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)

    def __post_init__(self) -> None:
        if (self.pixel_values is None) == (self.image_embeds is None):
            raise ValueError("JanusProARConditions requires exactly one of pixel_values or image_embeds.")

    @classmethod
    def concat(cls, items: Sequence["JanusProARConditions"]) -> "JanusProARConditions":
        if not items or len(items) == 1:
            return Batch.concat.__func__(cls, items)

        prompt = TextTokenCondition.concat([item.prompt for item in items])
        target_seq_len = _prompt_seq_len(prompt)
        images_seq_masks = [_pad_seq_tensor(item.images_seq_mask, target_seq_len) for item in items]
        cached = items[0].image_embeds is not None
        if any((item.image_embeds is not None) != cached for item in items[1:]):
            raise ValueError("JanusProARConditions.concat cannot mix pixels and cached image embeddings.")

        return cls(
            prompt=prompt,
            images_seq_mask=torch.cat(images_seq_masks, dim=0),
            images_emb_mask=torch.cat([item.images_emb_mask for item in items], dim=0),
            pixel_values=None if cached else torch.cat([item.pixel_values for item in items], dim=0),
            image_embeds=torch.cat([item.image_embeds for item in items], dim=0) if cached else None,
        )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "JanusProARConditions":
        prompt = _require_text_condition(d, "prompt")
        return cls(
            prompt=prompt,
            images_seq_mask=_stack_seq_rows(d["images_seq_mask"], _prompt_seq_len(prompt)),
            images_emb_mask=_require_tensor(d, "images_emb_mask"),
            pixel_values=_optional_tensor(d, "pixel_values"),
            image_embeds=_optional_tensor(d, "image_embeds"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "pixel_values": self.pixel_values,
            "images_seq_mask": [row.clone() for row in self.images_seq_mask],
            "images_emb_mask": self.images_emb_mask,
            "image_embeds": self.image_embeds,
        }


@dataclass
class JanusProImageARConditions(Batch):
    """Conditions for Janus-Pro Text -> Image autoregressive token generation."""

    cfg_weight: float = field(kind=FieldKind.SHARED)
    prompt: TextTokenCondition = field(kind=FieldKind.CONCAT)
    cfg_prompt: TextTokenCondition = field(kind=FieldKind.CONCAT)

    def __post_init__(self) -> None:
        self.cfg_weight = _finite_cfg_weight(
            self.cfg_weight,
            where="JanusProImageARConditions.cfg_weight",
        )

    @classmethod
    def concat(cls, items: Sequence["JanusProImageARConditions"]) -> "JanusProImageARConditions":
        if not items or len(items) == 1:
            return Batch.concat.__func__(cls, items)

        cfg_weight = items[0].cfg_weight
        if any(item.cfg_weight != cfg_weight for item in items[1:]):
            raise ValueError("JanusProImageARConditions.concat requires one shared cfg_weight.")

        return cls(
            prompt=TextTokenCondition.concat([item.prompt for item in items]),
            cfg_prompt=TextTokenCondition.concat([item.cfg_prompt for item in items]),
            cfg_weight=cfg_weight,
        )

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "JanusProImageARConditions":
        prompt = _require_text_condition(d, "prompt")
        cfg_prompt = _require_text_condition(d, "cfg_prompt")
        return cls(
            prompt=prompt,
            cfg_prompt=cfg_prompt,
            cfg_weight=d["cfg_weight"],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "cfg_prompt": self.cfg_prompt,
            "cfg_weight": self.cfg_weight,
        }


__all__ = ["JanusProARConditions", "JanusProImageARConditions"]
