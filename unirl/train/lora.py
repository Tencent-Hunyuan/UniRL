"""Plain LoRA adapter injection.

Build-time structural mutation only: :func:`inject_lora` installs a single
peft adapter on the trainable stage and stamps the post-materialize reset via
``unirl.train.deferred``.  No Shadow, no EMA — the dual-adapter NFT variant
lives in ``unirl.train.ema``.
"""

from __future__ import annotations

import logging
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


def inject_lora(
    model: nn.Module,
    *,
    rank: int,
    alpha: int,
    target_modules: ModuleSelection,
    exclude_modules: Optional[ModuleSelection] = None,
    dropout: float = 0.0,
    bias: str = "none",
    task_type: str = "FEATURE_EXTRACTION",
    adapter_name: str = "default",
) -> None:
    """Inject a single LoRA adapter.  No Shadow, no EMA."""
    from peft import LoraConfig, inject_adapter_in_model

    peft_cfg = LoraConfig(
        r=int(rank),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        target_modules=normalize_module_selection(target_modules),
        exclude_modules=normalize_optional_module_selection(exclude_modules),
        bias=str(bias),
        task_type=str(task_type),
    )
    inject_adapter_in_model(peft_cfg, model, adapter_name=adapter_name)
    _enforce_model_lora_trainability_contract(model)

    if _current_rank() == 0:
        n_trainable = sum(1 for p in model.parameters() if p.requires_grad)
        logged_targets = target_modules if isinstance(target_modules, str) else tuple(target_modules)
        logged_exclusions = (
            exclude_modules if isinstance(exclude_modules, str) or exclude_modules is None else tuple(exclude_modules)
        )
        logger.info(
            "inject_lora: adapter %r (rank=%d, alpha=%d, target_modules=%s, exclude_modules=%s) — %d trainable params",
            adapter_name,
            rank,
            alpha,
            logged_targets,
            logged_exclusions,
            n_trainable,
        )

    _stamp(model, partial(_reset_adapter, name=adapter_name))


def _enforce_model_lora_trainability_contract(model: nn.Module) -> None:
    """Apply optional model-owned freeze/validation markers after injection."""
    if bool(getattr(model, "_unirl_freeze_mtp_after_lora", False)):
        code_predictor = getattr(model, "code_predictor", None)
        if code_predictor is None:
            raise RuntimeError("Model requested post-LoRA MTP freezing but has no code_predictor")
        code_predictor.requires_grad_(False)
    if bool(getattr(model, "_unirl_freeze_embeddings_after_lora", False)):
        get_embeddings = getattr(model, "get_input_embeddings", None)
        if not callable(get_embeddings):
            raise RuntimeError("Model requested post-LoRA embedding freezing but has no embedding accessor")
        get_embeddings().requires_grad_(False)
    if not bool(getattr(model, "_unirl_phase1_layer0_lora_only", False)):
        return

    named_parameters = list(model.named_parameters())
    trainable = [name for name, parameter in named_parameters if parameter.requires_grad]
    invalid = [
        name
        for name in trainable
        if "lora_" not in name or name.startswith("code_predictor.") or ".code_predictor." in name
    ]
    mtp_lora = [
        name
        for name, _parameter in named_parameters
        if "lora_" in name and (name.startswith("code_predictor.") or ".code_predictor." in name)
    ]
    if not trainable or invalid or mtp_lora:
        raise RuntimeError(
            "Phase-1 Talker trainability contract requires LoRA only outside MTP; "
            f"trainable={trainable[:12]}, invalid={invalid[:12]}, mtp_lora={mtp_lora[:12]}"
        )


def _reset_adapter(model: nn.Module, *, name: str) -> None:
    from peft.tuners.lora import LoraLayer

    n_reset = 0
    for m in model.modules():
        if isinstance(m, LoraLayer):
            m.reset_lora_parameters(name, init_lora_weights=True)
            n_reset += 1
    if _current_rank() == 0:
        logger.info("_reset_adapter(%r): %d LoraLayer(s)", name, n_reset)


@contextmanager
def adapters_disabled(model: nn.Module) -> Iterator[None]:
    """Temporarily route every PEFT LoRA layer through its frozen base weights.

    This mirrors PEFT's adapter-disabling behavior without changing
    ``requires_grad``. The beta KL reference replay wraps this in ``no_grad`` so
    the shared FSDP model can act as pi_ref while preserving the trainable adapter
    state.
    """
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
    "adapters_disabled",
    "inject_lora",
    "normalize_module_selection",
    "normalize_optional_module_selection",
]
