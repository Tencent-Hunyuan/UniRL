"""Text conditioning types.

``TextEmbedCondition`` carries the output of a frozen text encoder (token-level
hidden states + optional pooled vector + attention mask). For dual-encoder
diffusion models (SD3, Flux) where text is pre-encoded into ``[B, T, hidden]``
before reaching the diffusion transformer.

``TextTokenCondition`` carries pre-encoder token IDs ready for an in-model
embedding lookup. For unified-vocab multimodal models (HunyuanImage 3.0, Janus,
Chameleon, …) where the diffusion / AR transformer owns its own embedding
table and consumes ``input_ids`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional, Sequence

import torch
import torch.nn.functional as F

from unirl.distributed.tensor.batch import Batch, FieldKind, field
from unirl.types.conditions.base import Condition, Modality


def _pad_text_tensor(value: Optional[torch.Tensor], target_seq_len: int) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if value.dim() < 2 or int(value.shape[1]) == target_seq_len:
        return value
    if int(value.shape[1]) > target_seq_len:
        raise ValueError(f"Cannot pad text tensor with seq_len={value.shape[1]} to shorter target={target_seq_len}")
    pad_spec = [0] * (2 * value.dim())
    pad_spec[(value.dim() - 1 - 1) * 2 + 1] = target_seq_len - int(value.shape[1])
    return F.pad(value, pad_spec, value=0)


@dataclass
class TextEmbedCondition(Condition):
    """Frozen-encoder text conditioning carried as token-level hidden states."""

    modality: ClassVar[Modality] = Modality.TEXT

    embeds: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)
    pooled: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)
    attn_mask: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)

    @classmethod
    def concat(cls, items: Sequence["TextEmbedCondition"]) -> "TextEmbedCondition":
        """Concat text embeddings, padding variable token lengths with zeros.

        Qwen-Image encodes each request chunk to that chunk's max prompt length,
        so different rollout shards can carry different sequence lengths. The
        generic batch concat uses plain ``torch.cat(dim=0)``; pad dim 1 first
        so shard merges preserve attention-mask semantics.
        """
        if not items or len(items) == 1:
            return Batch.concat.__func__(cls, items)

        seq_lens = [
            int(t.shape[1])
            for item in items
            for t in (item.embeds, item.attn_mask)
            if isinstance(t, torch.Tensor) and t.dim() >= 2
        ]
        if not seq_lens or len(set(seq_lens)) <= 1:
            return Batch.concat.__func__(cls, items)

        target_seq_len = max(seq_lens)
        padded = [
            cls(
                embeds=_pad_text_tensor(item.embeds, target_seq_len),
                pooled=item.pooled,
                attn_mask=_pad_text_tensor(item.attn_mask, target_seq_len),
            )
            for item in items
        ]
        return Batch.concat.__func__(cls, padded)


@dataclass
class TextTokenCondition(Condition):
    """Pre-encoder text conditioning carried as token IDs.

    For unified-vocab multimodal models where the transformer owns its own
    embedding table — the receiving stage looks ``input_ids`` up in that
    shared table rather than receiving pre-computed embeddings.
    """

    modality: ClassVar[Modality] = Modality.TEXT

    input_ids: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)
    attention_mask: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)

    @classmethod
    def concat(cls, items: Sequence["TextTokenCondition"]) -> "TextTokenCondition":
        """Concat token-id conditions, right-padding variable lengths first.

        SGLang's per-shard ``build_rollout_resp`` right-pads prompt
        ``input_ids`` to that shard's in-batch max, so different rollout
        actors carry different sequence lengths. The generic batch concat
        uses plain ``torch.cat(dim=0)``; pad dim 1 to the global max with
        zeros (attention_mask zeros the same positions out, so the pad
        token value is masked at attend-time and the receiver is unaffected).
        """
        if not items or len(items) == 1:
            return Batch.concat.__func__(cls, items)

        seq_lens = [
            int(t.shape[1])
            for item in items
            for t in (item.input_ids, item.attention_mask)
            if isinstance(t, torch.Tensor) and t.dim() >= 2
        ]
        if not seq_lens or len(set(seq_lens)) <= 1:
            return Batch.concat.__func__(cls, items)

        target_seq_len = max(seq_lens)
        padded = [
            cls(
                input_ids=_pad_text_tensor(item.input_ids, target_seq_len),
                attention_mask=_pad_text_tensor(item.attention_mask, target_seq_len),
            )
            for item in items
        ]
        return Batch.concat.__func__(cls, padded)


__all__ = ["TextEmbedCondition", "TextTokenCondition"]
