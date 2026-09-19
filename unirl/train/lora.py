"""Plain LoRA adapter injection."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from contextlib import contextmanager
from functools import partial
from typing import Iterator, Optional, Sequence, Union

from torch import nn

from unirl.models.types.post_materialize import defer_after_materialize

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

    defer_after_materialize(model, partial(_reset_adapter, name=adapter_name))


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
) -> str:
    """Inject a frozen LoRA adapter now, load its weights after materialization; returns the weight sha256."""
    from peft import LoraConfig, inject_adapter_in_model
    from peft.tuners.lora import LoraLayer

    if not name or name == trainable_adapter:
        raise ValueError(
            f"inject_frozen_adapter: adapter name must be non-empty and differ from "
            f"the trainable adapter {trainable_adapter!r}; got {name!r}."
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

    _set_adapter_requires_grad(model, name, False)
    # peft's inject path flips other adapters' requires_grad; restore the resting state.
    _activate(model, trainable_adapter)
    _set_adapter_requires_grad(model, trainable_adapter, True)
    # Mirror inject_nft: keep diffusers' PeftAdapterMixin bookkeeping consistent.
    if hasattr(model, "_hf_peft_config_loaded"):
        model._hf_peft_config_loaded = True

    # Weights need real (sharded) storage: FSDP/VeOmni materialize after injection.
    defer_after_materialize(
        model,
        partial(_load_frozen_adapter, name=name, weight_path=weight_path, trainable_adapter=trainable_adapter),
    )

    if _current_rank() == 0:
        total = sum(1 for m in model.modules() if isinstance(m, LoraLayer))
        n_params = sum(p.numel() for m in covered for p in (m.lora_A[name].weight, m.lora_B[name].weight))
        logger.info(
            "inject_frozen_adapter: %r from %s — rank=%d, %d/%d LoraLayer(s) covered, %d params (frozen, deferred load)",
            name,
            path,
            peft_cfg.r,
            len(covered),
            total,
            n_params,
        )
    return _file_sha256(weight_path)


_LORA_BANKS = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")
_PEFT_PREFIX = "base_model.model."


def adapter_of_lora_key(key: str) -> Optional[str]:
    """Adapter name of a model state-dict LoRA key (``...lora_A.<adapter>.weight``), else None."""
    parts = key.split(".")
    for bank in _LORA_BANKS:
        if bank in parts:
            idx = parts.index(bank)
            return parts[idx + 1] if idx + 1 < len(parts) else None
    return None


def _to_model_lora_key(key: str, name: str) -> str:
    """Map a peft checkpoint key (``base_model.model.<m>.lora_A.weight``) to ``<m>.lora_A.<name>.weight``."""
    if key.startswith(_PEFT_PREFIX):
        key = key[len(_PEFT_PREFIX) :]
    parts = key.split(".")
    for bank in _LORA_BANKS:
        if bank in parts:
            idx = parts.index(bank)
            return ".".join(parts[: idx + 1] + [name] + parts[idx + 1 :])
    return key


def _load_frozen_adapter(model: nn.Module, *, name: str, weight_path: str, trainable_adapter: str) -> None:
    """Post-materialize op: load ``weight_path`` into adapter ``name`` on every rank and re-freeze it."""
    from unirl.train.backend.sharded_state import load_model_state_dict

    if weight_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        raw = load_file(weight_path, device="cpu")
    else:
        import torch

        raw = torch.load(weight_path, map_location="cpu", weights_only=True)

    model_sd = model.state_dict()
    expected = {k for k in model_sd if adapter_of_lora_key(k) == name}
    mapped = {_to_model_lora_key(k, name): v for k, v in raw.items()}
    unexpected = sorted(set(mapped) - expected)
    if unexpected:
        raise ValueError(
            f"inject_frozen_adapter: {len(unexpected)} tensor(s) in {weight_path!r} matched no "
            f"parameter of adapter {name!r} (first: {unexpected[:3]}). The checkpoint does not "
            "line up with this model — refusing a silently partial teacher."
        )
    # peft zero-inits ``lora_B``: a missing tensor would silently null the teacher on that layer.
    missing = sorted(expected - set(mapped))
    if missing:
        raise ValueError(
            f"inject_frozen_adapter: {len(missing)} tensor(s) of adapter {name!r} are absent from "
            f"{weight_path!r} (first: {missing[:3]}) — refusing a partial teacher."
        )
    for key, value in mapped.items():
        want = model_sd[key].shape
        if tuple(value.shape) != tuple(want):
            raise ValueError(
                f"inject_frozen_adapter: {key!r} has shape {tuple(value.shape)} in {weight_path!r}, "
                f"model expects {tuple(want)} (adapter rank mismatch?)."
            )
    # Every rank holds the full (small) adapter dict; DCP slices each rank's own shard.
    load_model_state_dict(model, mapped, strict=False, broadcast_from_rank0=False)

    _set_adapter_requires_grad(model, name, False)
    _activate(model, trainable_adapter)
    _set_adapter_requires_grad(model, trainable_adapter, True)
    if _current_rank() == 0:
        logger.info("_load_frozen_adapter(%r): %d tensor(s) from %s", name, len(mapped), weight_path)


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    "adapter_of_lora_key",
    "adapters_disabled",
    "inject_frozen_adapter",
    "inject_lora",
    "normalize_module_selection",
    "normalize_optional_module_selection",
    "resolve_target_modules_pattern",
]
