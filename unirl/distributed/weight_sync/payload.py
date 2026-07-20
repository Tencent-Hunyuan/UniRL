"""Payload helper for :class:`~unirl.distributed.weight_sync.lora.base.LoraWeightSyncBase`.

``_peft_config_dict`` turns trainer-side state into a JSON/Ray-safe PEFT adapter
config for the LoRA path (``set_lora_from_tensors``).

Imported lazily from ``lora/base.py`` so the driver can reference the handler class
for ``remote(...)`` without eagerly pulling torch.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn


def _normalize_module_selection_field(peft_dict: dict, field: str) -> None:
    """Make a PEFT module selector deterministic and JSON/Ray-safe in-place."""
    value = peft_dict.get(field)
    if isinstance(value, str) or value is None:
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        peft_dict[field] = sorted(value) if isinstance(value, (set, frozenset)) else list(value)
        return
    raise RuntimeError(
        f"_peft_config_dict: peft_config[{field!r}] has unsupported type "
        f"{type(value).__name__}; expected str / list / set / tuple."
    )


def _resolve_peft_config_obj(
    model: nn.Module,
    adapter_name: str = "default",
) -> Any:
    """Walk model wrap layers and return the per-adapter PEFT config object."""
    cur: Any = model
    seen: set[int] = set()
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        pc = getattr(cur, "peft_config", None)
        if isinstance(pc, dict) and adapter_name in pc:
            return pc[adapter_name]
        cur = getattr(cur, "module", None) or getattr(cur, "base_model", None)
    return None


def _peft_config_dict(model: nn.Module, adapter_name: str = "default") -> dict:
    """Return a JSON/Ray-safe PEFT config dict for one adapter."""
    peft_cfg_obj = _resolve_peft_config_obj(model, adapter_name)
    if peft_cfg_obj is None:
        raise RuntimeError(f"_peft_config_dict: model has no peft_config[{adapter_name!r}] entry.")

    if hasattr(peft_cfg_obj, "to_dict"):
        peft_dict = peft_cfg_obj.to_dict()
    else:
        peft_dict = dict(peft_cfg_obj)

    _normalize_module_selection_field(peft_dict, "target_modules")
    _normalize_module_selection_field(peft_dict, "exclude_modules")

    for required in ("r", "lora_alpha", "target_modules"):
        if peft_dict.get(required) in (None, "", [], ()):
            raise RuntimeError(
                f"_peft_config_dict: peft_config[{required!r}] is "
                f"missing or empty (got {peft_dict.get(required)!r}); "
                f"rollout LoRA receive will reject this."
            )

    return peft_dict
