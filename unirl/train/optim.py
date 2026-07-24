"""Optimizer / LR-scheduler factories and their duck-typed protocols.

Kept separate from the config dataclasses (``unirl/train/backend/base.py``)
so the config layer stays torch-free — launcher scripts, linters, and schema
tools can import the typed dataclasses without pulling ``torch``.

The protocols capture the method surface the unirl training code actually
uses; the standard ``torch.optim.AdamW`` /
``torch.optim.lr_scheduler.LRScheduler`` instances satisfy them structurally.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Protocol, Tuple, runtime_checkable

import torch

from unirl.train.backend.base import LrSchedulerConfig, OptimizerConfig


@runtime_checkable
class OptimizerProtocol(Protocol):
    state: Dict[Any, Dict[str, Any]]

    def step(self) -> None: ...
    def zero_grad(self) -> None: ...
    def state_dict(self) -> Dict[str, Any]: ...
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None: ...


@runtime_checkable
class LRSchedulerProtocol(Protocol):
    def step(self) -> None: ...
    def state_dict(self) -> Dict[str, Any]: ...
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None: ...
    def get_last_lr(self) -> List[float]: ...


def _build_named_param_groups(
    named_params: Iterable[Tuple[str, torch.nn.Parameter]],
    *,
    base_lr: float,
    group_lrs: Dict[str, float],
) -> List[dict]:
    """Split trainable named params into AdamW param groups by name substring.

    Each ``(substring -> lr)`` in ``group_lrs`` becomes a group whose params'
    names contain that substring (first match wins); everything else lands in a
    base group at ``base_lr``. Empty groups are dropped. Used for per-expert LRs
    (e.g. BAGEL UniGRPO: ``{"moe_gen": <gen_lr>}`` so the image experts train at
    a different LR than the text/und experts within one optimizer).
    """
    base: List[torch.nn.Parameter] = []
    matched: Dict[str, List[torch.nn.Parameter]] = {sub: [] for sub in group_lrs}
    for name, p in named_params:
        if not p.requires_grad:
            continue
        hit = next((sub for sub in group_lrs if sub in name), None)
        (matched[hit] if hit is not None else base).append(p)
    groups: List[dict] = []
    if base:
        groups.append({"params": base, "lr": float(base_lr)})
    for sub, plist in matched.items():
        if plist:
            groups.append({"params": plist, "lr": float(group_lrs[sub])})
    return groups


def build_optimizer(
    config: OptimizerConfig,
    *,
    params: Iterable[torch.nn.Parameter],
    backend: Any = None,
    actor: Any = None,
    named_params: Optional[Iterable[Tuple[str, torch.nn.Parameter]]] = None,
) -> OptimizerProtocol:
    """Build an optimizer from a typed :class:`OptimizerConfig`.

    If ``backend`` is provided and its ``build_optimizer`` hook returns a
    non-None value, that takes precedence. Otherwise the default
    ``torch.optim.AdamW`` construction is used.

    ``params`` is consulted only on the default path; backend overrides
    are expected to pull parameters from their own model reference. When
    ``config.param_group_lrs`` is set AND ``named_params`` is provided, the
    trainable params are split into per-substring LR groups (see
    :func:`_build_named_param_groups`); otherwise a single LR is used.
    """
    del actor
    if backend is not None:
        backend_optimizer = backend.build_optimizer(config)
        if backend_optimizer is not None:
            return backend_optimizer

    # ``foreach=False`` disables the multi-tensor kernel path. Required whenever
    # the param list mixes regular ``torch.Tensor`` (non-FSDP-wrapped sub-modules
    # — e.g. SD3's embed/norm layers when only transformer blocks are
    # ``fully_shard``-wrapped) with ``DTensor`` (FSDP-wrapped block params):
    # ``_foreach_lerp_`` rejects the mixed bag and trips ``RuntimeError: got mixed
    # torch.Tensor and DTensor``. Single-tensor kernels handle each independently.
    adam_kwargs = dict(
        betas=(float(config.adam_beta1), float(config.adam_beta2)),
        eps=float(config.adam_epsilon),
        weight_decay=float(config.weight_decay),
        foreach=False,
    )

    param_group_lrs = getattr(config, "param_group_lrs", None)
    if param_group_lrs and named_params is not None:
        groups = _build_named_param_groups(
            named_params,
            base_lr=float(config.learning_rate),
            group_lrs=dict(param_group_lrs),
        )
        if groups:
            return torch.optim.AdamW(groups, lr=float(config.learning_rate), **adam_kwargs)

    trainable = [p for p in params if p.requires_grad]
    return torch.optim.AdamW(trainable, lr=float(config.learning_rate), **adam_kwargs)


def build_lr_scheduler(
    config: LrSchedulerConfig,
    *,
    optimizer: OptimizerProtocol,
    backend: Any = None,
    actor: Any = None,
) -> Optional[LRSchedulerProtocol]:
    """Build an LR scheduler from a typed :class:`LrSchedulerConfig`.

    Supports the same backend-override path as :func:`build_optimizer`.
    Returns ``None`` if ``config.type`` is not one of the supported values
    (``constant`` / ``linear`` / ``cosine``) and the backend did not provide
    an override.
    """
    del actor
    if backend is not None:
        backend_scheduler = backend.build_scheduler(config, optimizer)
        if backend_scheduler is not None:
            return backend_scheduler

    scheduler_type = str(config.type)
    warmup_steps = int(config.warmup_steps)
    total_steps = int(config.total_steps)

    if scheduler_type == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)

    if scheduler_type in {"linear", "cosine"}:
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}.")
        if total_steps < 1:
            raise ValueError(f"total_steps must be >= 1, got {total_steps}.")
        if warmup_steps >= total_steps:
            raise ValueError(
                "warmup_steps must be less than total_steps for a decaying "
                f"schedule, got warmup_steps={warmup_steps}, total_steps={total_steps}."
            )

    if scheduler_type == "linear":

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            return max(0.0, 1.0 - (step - warmup_steps) / (total_steps - warmup_steps))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if scheduler_type == "cosine":

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            progress = min(1.0, max(0.0, progress))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return None


__all__ = ["build_lr_scheduler", "build_optimizer", "OptimizerProtocol", "LRSchedulerProtocol"]
