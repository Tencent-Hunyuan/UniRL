"""Schema dataclasses for the training stack.

Config-layer schemas:

* :class:`OptimizerConfig` — AdamW hyperparameters
* :class:`LrSchedulerConfig` — LR schedule hyperparameters
* :class:`TrainTopology` — DP/TP/PP topology hints
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from unirl.config.require import require


@dataclass
class OptimizerConfig:
    """AdamW-style optimizer hyperparameters consumed by the training actor."""

    learning_rate: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    weight_decay: float


@dataclass
class LrSchedulerConfig:
    """Learning-rate scheduler hyperparameters."""

    type: str
    warmup_steps: int
    total_steps: int


@dataclass
class TrainTopology:
    """Unified training topology — injected into the concrete backend at build time.

    ``dp_size`` / ``dp_shard_size`` are ``Optional[int]``: ``None`` means
    "derive from ``torch.distributed.get_world_size()`` at runtime"; an
    explicit int means "use this value". ``dp_replicate_size`` defaults to 1
    (no replication). ``actor_count`` cross-checks the Ray placement's
    train-actor count at bootstrap time.
    """

    dp_size: Optional[int] = None
    dp_replicate_size: int = 1
    dp_shard_size: Optional[int] = None
    tp_size: int = 1
    pp_size: int = 1
    sp_size: int = 1
    ep_size: int = 1
    cp_size: int = 1
    actor_count: Optional[int] = 1

    def __post_init__(self) -> None:
        require(
            self.dp_size is None or self.dp_size >= 1,
            f"TrainTopology.dp_size must be >= 1 when set; got {self.dp_size!r}",
        )
        require(
            self.dp_shard_size is None or self.dp_shard_size >= 1,
            f"TrainTopology.dp_shard_size must be >= 1 when set; got {self.dp_shard_size!r}",
        )
        for name in ("dp_replicate_size", "tp_size", "pp_size", "sp_size", "ep_size", "cp_size"):
            value = getattr(self, name)
            require(value >= 1, f"TrainTopology.{name} must be >= 1; got {value!r}")
        require(
            self.actor_count is None or self.actor_count >= 1,
            f"TrainTopology.actor_count must be >= 1 when set; got {self.actor_count!r}",
        )

    def as_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "dp_replicate_size": int(self.dp_replicate_size),
            "tp_size": int(self.tp_size),
            "pp_size": int(self.pp_size),
            "sp_size": int(self.sp_size),
            "ep_size": int(self.ep_size),
            "cp_size": int(self.cp_size),
        }
        if self.dp_size is not None:
            d["dp_size"] = int(self.dp_size)
        if self.dp_shard_size is not None:
            d["dp_shard_size"] = int(self.dp_shard_size)
        if self.actor_count is not None:
            d["actor_count"] = int(self.actor_count)
        return d


__all__ = [
    "LrSchedulerConfig",
    "OptimizerConfig",
    "TrainTopology",
]
