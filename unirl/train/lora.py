"""LoRA adapter injection, switching, and frozen sibling adapters."""

from __future__ import annotations

import hashlib
import logging
import os
import re
from contextlib import contextmanager
from functools import partial
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from unirl.models.types.post_materialize import defer_after_materialize
from unirl.train.configs import normalize_frozen_adapters
from unirl.utils.peft_merge import _strip_peft_prefix

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


def _resolve_adapter_checkpoint(path: str) -> Tuple[str, Optional[str]]:
    """Split a peft adapter location into ``(model_id, subfolder)``: a local directory or ``org/repo[/subfolder]``."""
    if os.path.isdir(path):
        return path, None
    parts = path.split("/")
    if len(parts) < 2:
        raise ValueError(
            f"_resolve_adapter_checkpoint: {path!r} is neither a local directory nor an "
            "HF repo id ('org/repo' or 'org/repo/subfolder')."
        )
    return "/".join(parts[:2]), "/".join(parts[2:]) or None


def _inject_frozen_adapter(
    model: nn.Module,
    *,
    name: str,
    path: str,
) -> str:
    """Inject a frozen LoRA adapter now, load its weights after materialization; returns its content sha256."""
    from peft import LoraConfig, inject_adapter_in_model
    from peft.tuners.lora import LoraLayer
    from peft.utils import load_peft_weights

    if name in adapter_names(model):
        raise ValueError(f"inject_frozen_adapter: adapter {name!r} already exists on the model.")

    model_id, subfolder = _resolve_adapter_checkpoint(path)
    # The full saved config: scaling depends on use_rslora / alpha_pattern / rank_pattern, not just r and alpha.
    peft_cfg = LoraConfig.from_pretrained(model_id, subfolder=subfolder)
    init = peft_cfg.init_lora_weights
    if isinstance(init, str) and init not in _DELTA_INITS:
        # pissa / olora / corda / loftq / mica / lora_ga rewrite the (shared) base weight at inject time, and an
        # unconverted adapter of that kind is only valid on top of that rewritten base.
        raise ValueError(
            f"inject_frozen_adapter: {path!r} was saved with init_lora_weights={init!r}, so its weights assume "
            "a modified base model. Re-save it with save_pretrained(path_initial_model_for_weight_conversion=...) "
            "to convert it to a plain LoRA delta."
        )
    peft_cfg.lora_dropout = 0.0
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
    _activate(model, "default")
    _set_adapter_requires_grad(model, "default", True)
    # Mirror inject_nft: keep diffusers' PeftAdapterMixin bookkeeping consistent.
    if hasattr(model, "_hf_peft_config_loaded"):
        model._hf_peft_config_loaded = True

    raw = load_peft_weights(model_id, device="cpu", subfolder=subfolder)
    # peft ``base_model.model.<m>.lora_A.weight`` -> model ``<m>.lora_A.<name>.weight``.
    weights = {
        _LORA_BANK_RE.sub(lambda m: f"{m.group(0)}.{name}", _strip_peft_prefix(k), count=1): v for k, v in raw.items()
    }

    # Weights need real (sharded) storage: FSDP/VeOmni materialize after injection.
    defer_after_materialize(
        model,
        partial(_load_frozen_adapter, name=name, weights=weights, path=path),
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
    return _weights_sha256(weights)


_DELTA_INITS = ("gaussian", "eva", "orthogonal")
_LORA_BANK_RE = re.compile(r"\.lora_(?:embedding_)?[AB](?=\.|$)")
_LORA_ADAPTER_RE = re.compile(r"\.lora_(?:embedding_)?[AB]\.([^.]+)")


def _adapter_of_lora_key(key: str) -> Optional[str]:
    """Adapter name of a model state-dict LoRA key (``...lora_A.<adapter>.weight``), else None."""
    match = _LORA_ADAPTER_RE.search(key)
    return match.group(1) if match else None


def _load_frozen_adapter(model: nn.Module, *, name: str, weights: Dict[str, torch.Tensor], path: str) -> None:
    """Post-materialize op: load ``weights`` into the (already frozen) adapter ``name`` on every rank."""
    from unirl.train.backend.sharded_state import load_model_state_dict

    model_sd = model.state_dict()
    expected = {k for k in model_sd if _adapter_of_lora_key(k) == name}
    unexpected = sorted(set(weights) - expected)
    if unexpected:
        raise ValueError(
            f"inject_frozen_adapter: {len(unexpected)} tensor(s) in {path!r} matched no "
            f"parameter of adapter {name!r} (first: {unexpected[:3]}). The checkpoint does not "
            "line up with this model — refusing a silently partial teacher."
        )
    # peft zero-inits ``lora_B``: a missing tensor would silently null the teacher on that layer.
    missing = sorted(expected - set(weights))
    if missing:
        raise ValueError(
            f"inject_frozen_adapter: {len(missing)} tensor(s) of adapter {name!r} are absent from "
            f"{path!r} (first: {missing[:3]}) — refusing a partial teacher."
        )
    for key, value in weights.items():
        want = model_sd[key].shape
        if tuple(value.shape) != tuple(want):
            raise ValueError(
                f"inject_frozen_adapter: {key!r} has shape {tuple(value.shape)} in {path!r}, "
                f"model expects {tuple(want)} (adapter rank mismatch?)."
            )
    # Every rank holds the full (small) adapter dict; DCP slices each rank's own shard.
    # set_model_state_dict fills its input with every model entry, hence the copy.
    load_model_state_dict(model, dict(weights), strict=False, broadcast_from_rank0=False)

    if _current_rank() == 0:
        logger.info("_load_frozen_adapter(%r): %d tensor(s) from %s", name, len(weights), path)


def _weights_sha256(weights: Dict[str, torch.Tensor]) -> str:
    """Content hash over sorted ``(key, dtype, shape, bytes)``; independent of file format and metadata."""
    digest = hashlib.sha256()
    for key in sorted(weights):
        tensor = weights[key].detach().cpu().contiguous()
        digest.update(f"{key}|{tensor.dtype}|{tuple(tensor.shape)}|".encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class FrozenAdapters:
    """Frozen sibling LoRA adapters: kept out of adapter checkpoints and pinned by content sha256 on resume."""

    def __init__(self, shas: Optional[Dict[str, str]] = None) -> None:
        self.shas: Dict[str, str] = dict(shas or {})  # name -> content sha256

    @classmethod
    def inject(cls, model: nn.Module, specs: Any) -> FrozenAdapters:
        """Inject every ``lora_cfg.frozen_adapters`` entry: structure now, weights after materialization."""
        return cls(
            {s.name: _inject_frozen_adapter(model, name=s.name, path=s.path) for s in normalize_frozen_adapters(specs)}
        )

    def is_trainable_lora_key(self, key: str) -> bool:
        """True for ``lora_A`` / ``lora_B`` keys of a non-frozen adapter — what adapter checkpoints hold."""
        if "lora_A" not in key and "lora_B" not in key:
            return False
        return not self.shas or _adapter_of_lora_key(key) not in self.shas

    def check_resume(self, lora_config: Optional[Dict[str, object]]) -> None:
        """Refuse a checkpoint that recorded a different frozen set; an empty or absent record imposes nothing."""
        recorded = (lora_config or {}).get("frozen_adapters")
        if not recorded or recorded == self.shas:
            return
        added = sorted(set(self.shas) - set(recorded))
        removed = sorted(set(recorded) - set(self.shas))
        changed = sorted(
            f"{name} ({recorded[name][:12]}... -> {self.shas[name][:12]}...)"
            for name in set(recorded) & set(self.shas)
            if recorded[name] != self.shas[name]
        )
        raise RuntimeError(
            f"resume: frozen_adapters differ from the checkpoint's (added: {added}, removed: {removed}, "
            f"weights changed: {changed}). Resume with the same frozen adapter checkpoints or start a new run."
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
    "FrozenAdapters",
    "ModuleSelection",
    "adapter_active",
    "adapter_names",
    "adapters_disabled",
    "inject_lora",
    "normalize_module_selection",
    "normalize_optional_module_selection",
    "resolve_target_modules_pattern",
]
