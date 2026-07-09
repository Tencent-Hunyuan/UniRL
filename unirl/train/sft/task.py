"""SFTTask — the task-adapter contract for the supervised-training domain.

:class:`~unirl.train.sft.policy.SFTPolicy` is model-agnostic: it resolves a task
adapter from ``task_target`` via ``get_class(...).from_config(model_config)`` and
then drives it. Everything family-specific — which model to load, how to unpack a
record, and (crucially) *what the loss is* — lives in the task. This is why the
SAME skeleton serves both autoregressive families (next-token cross-entropy) and
diffusion families (flow-matching MSE): the loss difference is pushed entirely
into ``compute_loss``.

This module codifies that contract as a runtime-checkable Protocol plus a small
ABC (:class:`SFTTaskBase`) tasks may inherit for the shared boilerplate. The
policy itself only duck-types against the Protocol, so inheriting is optional —
but declaring the contract in one place keeps new adapters (qwen3, sd3, …) honest.

Contract (see :meth:`SFTPolicy.initialize` / ``train_batch`` / ``sample_media``):

* classmethod ``from_config(config) -> task`` — build the task (loads the bundle).
* attribute ``bundle`` — a model bundle exposing the trainable ``.transformer``
  submodule (``FSDPBackend`` wraps ``bundle.transformer`` via ``trainable_attr``).
* attribute ``block_class_names`` — tuple of transformer-block class names for
  FSDP2 per-block wrapping (e.g. ``("Qwen3DecoderLayer",)`` / ``("JointTransformerBlock",)``).
* ``load_record(record: dict) -> dict`` — worker-side; ``record`` is an opaque
  JSONL row with ``_root`` (manifest dir) injected. Load tensors from paths here
  so nothing heavy crosses the driver/Ray boundary.
* ``compute_loss(loaded: dict, *, generator=None) -> (loss_tensor, metrics: dict[str, float])``
  — one differentiable scalar loss for ONE sample + scalar metrics. The policy
  scales by ``1/len(shard)`` and calls ``.backward()``. ``metrics["loss/total"]``
  is the value the trainer's log line reads.
* ``sample(loaded: dict, *, generator=None) -> dict`` — eval-time generation
  (text / image / …); every FSDP rank must enter (collective weight gather) but
  only dp rank 0's return value is kept.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Protocol, Tuple, runtime_checkable

import torch


@runtime_checkable
class SFTTask(Protocol):
    """Structural contract the SFT policy drives (duck-typed; see module docstring)."""

    #: Model bundle; must expose the trainable ``.transformer`` submodule.
    bundle: Any
    #: Transformer-block class names for FSDP2 per-block wrapping.
    block_class_names: Tuple[str, ...]

    @classmethod
    def from_config(cls, config: Any) -> "SFTTask": ...

    def load_record(self, record: Dict[str, Any]) -> Dict[str, Any]: ...

    def compute_loss(
        self, loaded: Dict[str, Any], *, generator: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]: ...

    def sample(
        self, loaded: Dict[str, Any], *, generator: Optional[torch.Generator] = None
    ) -> Dict[str, Any]: ...


class SFTTaskBase(ABC):
    """Optional ABC for SFT tasks; enforces the 4 methods + declares the 2 attrs.

    Subclasses set ``block_class_names`` as a class attribute and assign
    ``self.bundle`` in ``__init__``. Inheriting is not required (the policy only
    checks the Protocol), but it gives a clear ``NotImplementedError`` surface
    and a single place to grow shared helpers.
    """

    #: Override per family, e.g. ``("Qwen3DecoderLayer",)``.
    block_class_names: Tuple[str, ...] = ()

    #: Set in ``__init__`` (must expose ``.transformer``).
    bundle: Any = None

    @classmethod
    @abstractmethod
    def from_config(cls, config: Any) -> "SFTTaskBase": ...

    @abstractmethod
    def load_record(self, record: Dict[str, Any]) -> Dict[str, Any]: ...

    @abstractmethod
    def compute_loss(
        self, loaded: Dict[str, Any], *, generator: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]: ...

    @abstractmethod
    def sample(
        self, loaded: Dict[str, Any], *, generator: Optional[torch.Generator] = None
    ) -> Dict[str, Any]: ...


__all__ = ["SFTTask", "SFTTaskBase"]
