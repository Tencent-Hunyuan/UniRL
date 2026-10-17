"""Optimizer and LR-scheduler factories."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch.distributed.tensor import DTensor

from unirl.train.backend.base import LrSchedulerConfig, OptimizerConfig


def _build_named_param_groups(
    named_params: Iterable[Tuple[str, torch.nn.Parameter]],
    *,
    base_lr: float,
    group_lrs: Dict[str, float],
) -> List[dict]:
    """Split trainable named params into AdamW param groups by name substring."""
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
    named_params: Optional[Iterable[Tuple[str, torch.nn.Parameter]]] = None,
) -> torch.optim.AdamW:
    """Build an AdamW optimizer from a typed :class:`OptimizerConfig`."""
    if config.param_group_lrs and named_params is not None:
        groups = _build_named_param_groups(
            named_params,
            base_lr=float(config.learning_rate),
            group_lrs=dict(config.param_group_lrs),
        )
    else:
        groups = []
    if not groups:
        groups = [{"params": [p for p in params if p.requires_grad]}]
    for group in groups:
        wrapped = [isinstance(p, DTensor) for p in group["params"]]
        if any(wrapped) and not all(wrapped):
            group["foreach"] = False
    return torch.optim.AdamW(
        groups,
        lr=float(config.learning_rate),
        betas=(float(config.adam_beta1), float(config.adam_beta2)),
        eps=float(config.adam_epsilon),
        weight_decay=float(config.weight_decay),
    )


def build_lr_scheduler(
    config: LrSchedulerConfig,
    *,
    optimizer: torch.optim.Optimizer,
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    """Build an LR scheduler from a typed :class:`LrSchedulerConfig`."""
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

    if scheduler_type == "linear_warmup":
        constant = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=1.0,
            total_iters=1,
        )
        if warmup_steps <= 0:
            return constant
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, constant],
            milestones=[warmup_steps],
        )

    if scheduler_type == "cosine":

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            progress = min(1.0, max(0.0, progress))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return None


__all__ = ["build_lr_scheduler", "build_optimizer"]
