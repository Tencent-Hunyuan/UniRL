"""Plain LoRA adapter injection."""

from __future__ import annotations

import json
import logging
import os
import re
from contextlib import contextmanager
from functools import partial
from typing import Iterator, Optional, Sequence, Union

from torch import nn

from unirl.train.deferred import _stamp

logger = logging.getLogger(__name__)


ModuleSelection = Union[str, Sequence[str]]
PeftModuleSelection = Union[str, list[str]]


def normalize_module_selection(modules: ModuleSelection) -> PeftModuleSelection:
    """Preserve PEFT regex/shorthand strings; materialize other sequences."""
    if isinstance(modules, str):
        return modules
    if not isinstance(modules, Sequence) or any(not isinstance(module, str) for module in modules):
        raise TypeError(
            "LoRA module selectors must be a regex/shorthand string or a sequence of strings; "
            f"got {type(modules).__name__}"
        )
    return list(modules)


def normalize_optional_module_selection(
    modules: Optional[ModuleSelection],
) -> Optional[PeftModuleSelection]:
    """Normalize an optional PEFT module selector without changing its semantics."""
    return None if modules is None else normalize_module_selection(modules)


def resolve_target_modules_pattern(
    *,
    target_modules: ModuleSelection,
    module_prefix: str = "",
) -> tuple[PeftModuleSelection, str]:
    """Resolve PEFT targets, optionally restricting suffixes to one subtree."""
    normalized = normalize_module_selection(target_modules)
    if not module_prefix:
        logged = normalized if isinstance(normalized, str) else tuple(normalized)
        return normalized, str(logged)

    if isinstance(normalized, str):
        raise ValueError(
            "resolve_target_modules_pattern: module_prefix cannot be combined "
            "with a regex or 'all-linear' target_modules string; provide an "
            "explicit sequence of target-module suffixes."
        )
    if not normalized:
        raise ValueError(
            "resolve_target_modules_pattern: module_prefix is set but "
            "target_modules is empty; provide at least one target-module suffix."
        )
    normalized_prefix = str(module_prefix).strip(".")
    if not normalized_prefix:
        raise ValueError(
            "resolve_target_modules_pattern: module_prefix must contain a model subtree name, not only dots."
        )
    prefix_re = re.escape(normalized_prefix)
    leaves_re = "|".join(re.escape(str(target)) for target in normalized)
    pattern = rf"^{prefix_re}\.(?:.*\.)?(?:{leaves_re})$"
    return pattern, pattern


def inject_lora(
    model: nn.Module,
    *,
    rank: int,
    alpha: int,
    target_modules: ModuleSelection,
    module_prefix: str = "",
    exclude_modules: Optional[ModuleSelection] = None,
    dropout: float = 0.0,
    bias: str = "none",
    task_type: str = "FEATURE_EXTRACTION",
    adapter_name: str = "default",
) -> None:
    """Inject a single LoRA adapter.  No Shadow, no EMA."""
    from peft import LoraConfig, inject_adapter_in_model

    peft_target_modules, log_target = resolve_target_modules_pattern(
        target_modules=target_modules,
        module_prefix=module_prefix,
    )

    peft_cfg = LoraConfig(
        r=int(rank),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        target_modules=peft_target_modules,
        exclude_modules=normalize_optional_module_selection(exclude_modules),
        bias=str(bias),
        task_type=str(task_type),
    )
    inject_adapter_in_model(peft_cfg, model, adapter_name=adapter_name)

    if _current_rank() == 0:
        n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
        logged_exclusions = (
            exclude_modules if isinstance(exclude_modules, str) or exclude_modules is None else tuple(exclude_modules)
        )
        logger.info(
            "inject_lora: adapter %r (rank=%d, alpha=%d, target_modules=%s, exclude_modules=%s) — %d trainable params",
            adapter_name,
            rank,
            alpha,
            log_target,
            logged_exclusions,
            n_trainable,
        )

    _stamp(model, partial(_reset_adapter, name=adapter_name))


def _reset_adapter(model: nn.Module, *, name: str) -> None:
    from peft.tuners.lora import LoraLayer

    n_reset = 0
    for m in model.modules():
        if isinstance(m, LoraLayer):
            m.reset_lora_parameters(name, init_lora_weights=True)
            n_reset += 1
    if _current_rank() == 0:
        logger.info("_reset_adapter(%r): %d LoraLayer(s)", name, n_reset)


def _activate(model: nn.Module, adapter_name: str) -> None:
    from peft.tuners.lora import LoraLayer

    for m in model.modules():
        if isinstance(m, LoraLayer):
            m.set_adapter(adapter_name)


def _set_adapter_requires_grad(model: nn.Module, name: str, requires_grad: bool) -> None:
    from peft.tuners.lora import LoraLayer

    for m in model.modules():
        if not isinstance(m, LoraLayer):
            continue
        for key in ("lora_A", "lora_B"):
            bank = getattr(m, key, {})
            if name in bank:
                bank[name].weight.requires_grad = requires_grad


def adapter_names(model: nn.Module) -> set:
    """Names of every LoRA adapter present on ``model``'s LoraLayers."""
    from peft.tuners.lora import LoraLayer

    names: set = set()
    for m in model.modules():
        if isinstance(m, LoraLayer):
            names.update(getattr(m, "lora_A", {}).keys())
    return names


@contextmanager
def adapter_active(model: nn.Module, name: str, *, trainable: str = "default") -> Iterator[None]:
    """Temporarily route every LoraLayer through the frozen adapter ``name``."""
    if name == trainable:
        raise ValueError(f"adapter_active: {name!r} is the trainable adapter; only frozen adapters can be routed.")
    _activate(model, name)
    _set_adapter_requires_grad(model, trainable, True)
    _set_adapter_requires_grad(model, name, False)
    try:
        yield
    finally:
        _activate(model, trainable)
        _set_adapter_requires_grad(model, trainable, True)
        _set_adapter_requires_grad(model, name, False)


def _resolve_adapter_checkpoint(path: str) -> tuple:
    """Resolve a peft adapter checkpoint to local ``(config_path, weight_path)``."""
    if os.path.isdir(path):
        config_path = os.path.join(path, "adapter_config.json")
        weight_path = os.path.join(path, "adapter_model.safetensors")
        if not os.path.exists(weight_path):
            weight_path = os.path.join(path, "adapter_model.bin")
        if not os.path.exists(config_path) or not os.path.exists(weight_path):
            raise FileNotFoundError(
                f"_resolve_adapter_checkpoint: {path!r} is a directory but lacks "
                "adapter_config.json + adapter_model.safetensors/.bin."
            )
        return config_path, weight_path

    from huggingface_hub import hf_hub_download

    parts = path.split("/")
    if len(parts) < 2:
        raise ValueError(
            f"_resolve_adapter_checkpoint: {path!r} is neither a local directory nor an "
            "HF repo id ('org/repo' or 'org/repo/subfolder')."
        )
    dl_kwargs = {"repo_id": "/".join(parts[:2])}
    if len(parts) > 2:
        dl_kwargs["subfolder"] = "/".join(parts[2:])
    config_path = hf_hub_download(filename="adapter_config.json", **dl_kwargs)
    try:
        weight_path = hf_hub_download(filename="adapter_model.safetensors", **dl_kwargs)
    except Exception:
        weight_path = hf_hub_download(filename="adapter_model.bin", **dl_kwargs)
    return config_path, weight_path


def inject_frozen_adapter(
    model: nn.Module,
    *,
    name: str,
    path: str,
    trainable_adapter: str = "default",
) -> None:
    """Inject a frozen, inference-only LoRA adapter with weights from ``path``."""
    from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict
    from peft.tuners.lora import LoraLayer

    if not name or name == trainable_adapter:
        raise ValueError(
            f"inject_frozen_adapter: adapter name must be non-empty and differ from "
            f"the trainable adapter {trainable_adapter!r}; got {name!r}."
        )
    if any(p.is_meta for p in model.parameters()):
        raise NotImplementedError(
            "inject_frozen_adapter: the trainable module is meta-initialized; frozen "
            "adapters load real weights at injection time and require an eager bundle."
        )
    if name in adapter_names(model):
        raise ValueError(f"inject_frozen_adapter: adapter {name!r} already exists on the model.")

    config_path, weight_path = _resolve_adapter_checkpoint(path)
    with open(config_path) as f:
        adapter_cfg = json.load(f)

    peft_cfg = LoraConfig(
        r=int(adapter_cfg["r"]),
        lora_alpha=int(adapter_cfg.get("lora_alpha", adapter_cfg["r"])),
        target_modules=adapter_cfg.get("target_modules") or [],
        lora_dropout=0.0,
        bias=str(adapter_cfg.get("bias", "none")),
    )
    inject_adapter_in_model(peft_cfg, model, adapter_name=name)

    covered = [m for m in model.modules() if isinstance(m, LoraLayer) and name in getattr(m, "lora_A", {})]
    if not covered:
        raise ValueError(
            f"inject_frozen_adapter: adapter {name!r} matched no module — its "
            f"target_modules {peft_cfg.target_modules!r} found nothing on this model. "
            "Wrong checkpoint for this architecture?"
        )

    if weight_path.endswith(".safetensors"):
        from safetensors.torch import load_file as _load_weights
    else:
        import torch

        def _load_weights(p):
            return torch.load(p, map_location="cpu", weights_only=True)

    load_result = set_peft_model_state_dict(model, _load_weights(weight_path), adapter_name=name)
    unexpected = list(getattr(load_result, "unexpected_keys", []) or [])
    if unexpected:
        raise ValueError(
            f"inject_frozen_adapter: {len(unexpected)} tensor(s) in {weight_path!r} matched no "
            f"parameter of adapter {name!r} (first: {unexpected[:3]}). The checkpoint does not "
            "line up with this model — refusing a silently partial teacher."
        )
    # strict=False ``missing_keys`` spans the whole model; only THIS adapter's lora
    # banks indicate a truncated checkpoint (peft zero-inits ``lora_B`` -> a missing
    # tensor silently nulls the teacher on that layer).
    missing: list = []
    for key in getattr(load_result, "missing_keys", None) or []:
        parts = key.split(".")
        for bank in ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B"):
            if bank in parts:
                bank_idx = parts.index(bank)
                if bank_idx + 1 < len(parts) and parts[bank_idx + 1] == name:
                    missing.append(key)
                break
    if missing:
        raise ValueError(
            f"inject_frozen_adapter: {len(missing)} tensor(s) of adapter {name!r} are absent from "
            f"{weight_path!r} (first: {missing[:3]}). peft zero-initializes the missing side, which "
            "would silently disable the teacher on those layers — refusing a partial teacher."
        )

    _set_adapter_requires_grad(model, name, False)
    # peft's inject/set paths flip other adapters' requires_grad; restore the resting state.
    _activate(model, trainable_adapter)
    _set_adapter_requires_grad(model, trainable_adapter, True)
    # Mirror inject_nft: keep diffusers' PeftAdapterMixin bookkeeping consistent.
    if hasattr(model, "_hf_peft_config_loaded"):
        model._hf_peft_config_loaded = True

    if _current_rank() == 0:
        total = sum(1 for m in model.modules() if isinstance(m, LoraLayer))
        n_params = sum(p.numel() for m in covered for p in (m.lora_A[name].weight, m.lora_B[name].weight))
        logger.info(
            "inject_frozen_adapter: %r from %s — rank=%d, %d/%d LoraLayer(s) covered, %d params (frozen)",
            name,
            path,
            peft_cfg.r,
            len(covered),
            total,
            n_params,
        )


@contextmanager
def adapters_disabled(model: nn.Module) -> Iterator[None]:
    """Temporarily route every PEFT LoRA layer through its frozen base weights."""
    from peft.tuners.lora import LoraLayer

    layers = [m for m in model.modules() if isinstance(m, LoraLayer)]
    prev = [bool(getattr(m, "_disable_adapters", False)) for m in layers]
    try:
        for m in layers:
            m._disable_adapters = True
        yield
    finally:
        for m, was_disabled in zip(layers, prev):
            m._disable_adapters = was_disabled


def _current_rank() -> int:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    return 0


__all__ = [
    "ModuleSelection",
    "adapter_active",
    "adapter_names",
    "adapters_disabled",
    "inject_frozen_adapter",
    "inject_lora",
    "normalize_module_selection",
    "normalize_optional_module_selection",
    "resolve_target_modules_pattern",
]
