"""FusedMultimodalCondition — token sequence over a unified text+image vocabulary.

For unified-vocab multimodal models (HunyuanImage 3.0, Janus, Chameleon, Show-o,
OmniGen, …) where text and other modalities share one transformer and one
embedding table. The model consumes a single fused token sequence with
multimodal tokens interleaved, and embeds them itself via its own ``wte``.

This primitive describes only *the fused sequence itself* — token IDs, the
4D causal+image-bidirectional attention mask, position IDs, and an optional
pre-computed rope cache. Model-specific scatter layouts (which positions of
the fused sequence get specific encoded content scattered into them) live in
subclasses (e.g. ``HunyuanImage3FusedMultimodalCondition``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional, Tuple

import torch

from unirl.distributed.tensor.batch import FieldKind, field, shared_field
from unirl.types.conditions.base import Condition, Modality


@dataclass
class FusedMultimodalCondition(Condition):
    """Generic fused-multimodal-sequence input.

    Field shapes (let ``B = batch``, ``L = fused sequence length``,
    ``D = head_dim``):

        input_ids       : [B, L]          long
        attention_mask  : [B, 1, L, L]    bool — 4D causal+image-bidir
        position_ids    : [B, L]          long
        rope_cache      : (cos, sin)      each [B, L, D] float — optional
                                          pre-computed rope tables (perf cache)
    """

    modality: ClassVar[Modality] = Modality.MULTIMODAL

    input_ids: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)
    attention_mask: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)
    position_ids: Optional[torch.Tensor] = field(kind=FieldKind.CONCAT, default=None)
    # Tuple-valued; SHARED. The smoke / single-process path doesn't cross
    # a transfer queue. Real cross-process replay would need a tuple-aware
    # FieldKind; out of scope for this primitive.
    rope_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = shared_field(default=None)


__all__ = ["FusedMultimodalCondition"]
