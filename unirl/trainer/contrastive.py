"""Contrastive rollout selection for Sol-RL-style diffusion training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import torch
from omegaconf import OmegaConf

from unirl.distributed.tensor import hydrate
from unirl.types.sample import Part


@dataclass(frozen=True)
class ContrastiveRolloutConfig:
    """Configure driver-owned two-stage rollout selection."""

    mode: str = "scout_regen"
    top_k: int = 8
    bottom_k: int = 8
    prompt_chunk_size: int = 16
    trace_dir: Optional[str] = None
    trace_interval: int = 1

    def __post_init__(self) -> None:
        if self.mode not in {"naive", "scout_regen"}:
            raise ValueError(f"contrastive_rollout.mode must be naive|scout_regen, got {self.mode!r}.")
        if self.top_k < 0 or self.bottom_k < 0 or self.selected_count < 1:
            raise ValueError(
                "contrastive_rollout needs non-negative top_k/bottom_k with a positive sum; "
                f"got top_k={self.top_k}, bottom_k={self.bottom_k}."
            )
        if self.prompt_chunk_size < 1:
            raise ValueError(f"contrastive_rollout.prompt_chunk_size must be >= 1, got {self.prompt_chunk_size}.")
        if self.trace_interval < 1:
            raise ValueError(f"contrastive_rollout.trace_interval must be >=1, got {self.trace_interval}.")

    @property
    def selected_count(self) -> int:
        return int(self.top_k + self.bottom_k)


def build_contrastive_config(value: Any) -> ContrastiveRolloutConfig | None:
    """Parse an optional plain YAML/OmegaConf mapping."""

    if value is None:
        return None
    if OmegaConf.is_config(value):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, Mapping):
        raise TypeError(f"contrastive_rollout must be a mapping, got {type(value).__name__}.")
    unknown = sorted(
        set(value)
        - {
            "mode",
            "top_k",
            "bottom_k",
            "prompt_chunk_size",
            "trace_dir",
            "trace_interval",
        }
    )
    if unknown:
        raise ValueError(f"contrastive_rollout has unknown field(s) {unknown}.")
    return ContrastiveRolloutConfig(**dict(value))


def select_top_bottom_indices(part: Part, *, top_k: int, bottom_k: int) -> torch.Tensor:
    """Return deterministic top/bottom reward rows for every prompt group."""

    if part.rewards is None:
        raise ValueError("select_top_bottom_indices requires a scored Part.")
    rewards = hydrate(part.rewards).to(torch.float32)
    if rewards.ndim != 1 or int(rewards.numel()) != part.batch_size:
        raise ValueError(
            "select_top_bottom_indices expects one scalar reward per row; "
            f"got shape {tuple(rewards.shape)} for batch {part.batch_size}."
        )
    if not torch.isfinite(rewards).all():
        bad = torch.nonzero(~torch.isfinite(rewards), as_tuple=False).flatten().tolist()
        raise ValueError(f"select_top_bottom_indices rejects non-finite rewards at rows {bad[:8]}.")

    group_ids = part.group_ids
    unique_groups = list(dict.fromkeys(group_ids))
    if not unique_groups or part.batch_size % len(unique_groups):
        raise ValueError(
            f"contrastive selection requires uniform prompt groups; batch={part.batch_size}, "
            f"groups={len(unique_groups)}."
        )
    group_size = part.batch_size // len(unique_groups)
    expected = [group_id for group_id in unique_groups for _ in range(group_size)]
    if group_ids != expected:
        raise ValueError("contrastive selection requires group-contiguous candidate rows.")
    if top_k + bottom_k > group_size:
        raise ValueError(f"top_k({top_k}) + bottom_k({bottom_k}) exceeds scout group size {group_size}.")

    selected: list[torch.Tensor] = []
    for group_index in range(len(unique_groups)):
        start = group_index * group_size
        group_rewards = rewards[start : start + group_size]
        descending = torch.argsort(group_rewards, descending=True, stable=True)
        ascending = torch.argsort(group_rewards, descending=False, stable=True)
        top = descending[:top_k]
        available = torch.ones(group_size, dtype=torch.bool, device=rewards.device)
        available[top] = False
        bottom = ascending[available.index_select(0, ascending)][:bottom_k]
        local = torch.cat((top, bottom))
        selected.append(local + start)
    return torch.cat(selected).to(dtype=torch.long)


__all__ = [
    "ContrastiveRolloutConfig",
    "build_contrastive_config",
    "select_top_bottom_indices",
]
