"""Duck-typed protocols for optimizer and LR scheduler objects.

These protocols capture the method surface the unirl training code
actually uses, verified by grep across ``ray/train_actor.py`` and
``training/stack.py``. The standard ``torch.optim.AdamW`` /
``torch.optim.lr_scheduler.LRScheduler`` instances satisfy the protocols
structurally.
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable


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


__all__ = ["OptimizerProtocol", "LRSchedulerProtocol"]
