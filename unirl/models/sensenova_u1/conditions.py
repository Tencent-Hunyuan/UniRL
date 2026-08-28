"""Typed prompt-cache conditions for SenseNova-U1.5 pixel diffusion."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Tuple

import torch

from unirl.config.require import require
from unirl.distributed.tensor.batch import concat_field
from unirl.types.conditions.base import Condition, Modality


def _cache_tensors(cache: Any):
    """Yield tensor attributes stored by Transformers cache layers."""
    for layer in getattr(cache, "layers", ()):
        for name in ("keys", "values", "flash_k_cache", "flash_v_cache"):
            value = getattr(layer, name, None)
            if isinstance(value, torch.Tensor):
                yield layer, name, value


def _move_cache(cache: Any, device: str | torch.device) -> Any:
    """Move a Transformers cache without aliasing the source across devices."""
    if cache is None:
        return None
    target = torch.device(device)
    tensors = list(_cache_tensors(cache))
    if all(value.device == target for _, _, value in tensors):
        return cache
    moved = copy.deepcopy(cache)
    for layer, name, value in _cache_tensors(moved):
        setattr(layer, name, value.to(target))
    return moved


def _move_shared_caches(caches: List[Any], device: str | torch.device) -> List[Any]:
    """Move each distinct cache once while preserving same-prompt aliases."""
    memo: Dict[int, Any] = {}
    moved: List[Any] = []
    for cache in caches:
        if cache is None:
            moved.append(None)
            continue
        key = id(cache)
        if key not in memo:
            memo[key] = _move_cache(cache, device)
        moved.append(memo[key])
    return moved


@dataclass
class SenseNovaU1Conditions(Condition):
    """Frozen text-prefix caches and spatial metadata, one entry per sample."""

    modality: ClassVar[Modality] = Modality.IMAGE

    prompts: List[str] = concat_field(default_factory=list)
    condition_caches: List[Any] = concat_field(default_factory=list)
    uncondition_caches: List[Any] = concat_field(default_factory=list)
    condition_image_indexes: List[Any] = concat_field(default_factory=list)
    uncondition_image_indexes: List[Any] = concat_field(default_factory=list)
    image_shapes: List[Tuple[int, int]] = concat_field(default_factory=list)

    @property
    def batch_size(self) -> int:
        return len(self.prompts)

    def validate(self) -> None:
        """Require all per-sample fields to remain aligned after batching/slicing."""
        n = self.batch_size
        for name in (
            "condition_caches",
            "uncondition_caches",
            "condition_image_indexes",
            "uncondition_image_indexes",
            "image_shapes",
        ):
            values = getattr(self, name)
            require(
                len(values) == n,
                f"SenseNovaU1Conditions.{name} has {len(values)} entries for batch_size={n}.",
            )

    def single(self, index: int = 0) -> Tuple[str, Any, Any, Any, Any, Tuple[int, int]]:
        """Return one sample's prompt, caches, indexes, and ``(H, W)`` shape."""
        self.validate()
        require(0 <= index < self.batch_size, f"SenseNovaU1Conditions.single: index {index} is out of range.")
        return (
            self.prompts[index],
            self.condition_caches[index],
            self.uncondition_caches[index],
            self.condition_image_indexes[index],
            self.uncondition_image_indexes[index],
            tuple(self.image_shapes[index]),
        )

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "SenseNovaU1Conditions":
        """Read conditions from a generated Part."""
        conditions = values.get("sensenova_u1")
        if not isinstance(conditions, cls):
            raise TypeError(
                "SenseNovaU1Conditions.from_dict expected values['sensenova_u1'] "
                f"to be SenseNovaU1Conditions, got {type(conditions).__name__}."
            )
        return conditions

    def to_dict(self) -> Dict[str, Any]:
        """Write conditions into a generated Part."""
        self.validate()
        return {"sensenova_u1": self}

    def to_device(self, device: str | torch.device) -> "SenseNovaU1Conditions":
        """Move image indexes and opaque prefix KV caches together."""
        moved = super().to_device(device)
        moved.condition_caches = _move_shared_caches(moved.condition_caches, device)
        moved.uncondition_caches = _move_shared_caches(moved.uncondition_caches, device)
        return moved


__all__ = ["SenseNovaU1Conditions"]
