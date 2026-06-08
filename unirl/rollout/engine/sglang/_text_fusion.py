"""Text-encoder fusion for SGLang ``OutputBatch`` results (engine-private).

SGLang's ``OutputBatch`` may surface a per-result text-conditioning field as
either a single tensor or a list/tuple of tensors — one entry per text
encoder. SDXL/SD3/FLUX use multiple encoders (CLIP-L, CLIP-G, T5) whose
outputs must be fused before being handed to the diffusion forward pass.

Underscore-prefixed module name marks this as private to the sglang
rollout-engine package; the only consumer is
:mod:`unirl.rollout.engine.sglang.response`.
"""

from __future__ import annotations

from typing import Any, Optional

import torch


def fuse_text_encoder_outputs(value: Any) -> Optional[torch.Tensor]:
    """Reduce a per-result text-conditioning field to a single tensor.

    Concatenation axis follows the SDXL/SD3/FLUX fusion convention:
    - ≥3-D (token-level, ``[B, seq, hidden]``) → ``cat(dim=-2)`` along seq
      (e.g. SD3/FLUX merging CLIP and T5 token streams).
    - 2-D (pooled, ``[B, hidden]``) → ``cat(dim=-1)`` along hidden
      (e.g. SDXL stacking CLIP-L and CLIP-G pooled vectors side-by-side).

    Returns ``None`` when the input is missing or contains no tensor.
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
