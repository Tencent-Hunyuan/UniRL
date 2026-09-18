"""The fork's in-memory LoRA request struct, ``SetLoraFromTensorsReq``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union


@dataclass
class SetLoraFromTensorsReq:
    lora_nickname: str
    lora_tensors: dict
    target: Union[str, List[str]] = "all"
    strength: Union[float, List[float]] = 1.0
    lora_alpha: Optional[float] = None
