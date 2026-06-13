"""The fork's in-memory LoRA request struct, ``SetLoraFromTensorsReq``.

Stock upstream sglang has no in-memory LoRA-from-tensors request; the fork added
``SetLoraFromTensorsReq`` to ``runtime/entrypoints/utils.py`` next to the
file-path ``SetLoraReq``. Defining it here is the **single definition site** --
both ``patch_scheduler`` (which keys ``request_handlers`` by ``type(req)``) and
the UniRL adapter (``rollout/engine/sglang/engine.py``, which builds and
sends the request) import it from here, so dispatch matches on class identity.

Copied verbatim from
``sglang-drl/.../runtime/entrypoints/utils.py`` (only stdlib types; no sglang
import needed, so this module is import-safe on any interpreter).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Union


@dataclass
class SetLoraFromTensorsReq:
    lora_nickname: str
    lora_tensors: dict  # dict[str, torch.Tensor]
    target: Union[str, List[str]] = "all"
    strength: Union[float, List[float]] = 1.0
