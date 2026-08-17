"""HunyuanImage3DiffusionState — KV-cache thread across diffusion steps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass
class HunyuanImage3DiffusionState:
    """Per-rollout KV-cache state: ``position_ids [N, L']``, ``attention_mask [N, 1, L', L]`` gathered per step."""

    past_key_values: Optional[Any] = None
    position_ids: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    gen_timestep_scatter_index: Optional[torch.Tensor] = None


__all__ = ["HunyuanImage3DiffusionState"]
