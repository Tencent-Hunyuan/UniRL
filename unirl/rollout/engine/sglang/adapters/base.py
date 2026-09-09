"""Driver-side ``Sample`` → ``Sample`` conversion: the adapter ABC + registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from unirl.config.require import require
from unirl.rollout.engine.sglang.backends import RawResult
from unirl.rollout.engine.sglang.utils import ResolvedSampling
from unirl.types.sample import Sample

_REGISTRY: Dict[str, type["ModelAdapter"]] = {}


def register_adapter(key: str):
    """Class decorator: register an adapter under its ``model_family`` key."""

    def deco(cls: type["ModelAdapter"]) -> type["ModelAdapter"]:
        require(
            key not in _REGISTRY,
            f"adapter key {key!r} already registered by {_REGISTRY.get(key)!r}",
        )
        _REGISTRY[key] = cls
        cls.model_family = key
        return cls

    return deco


def get_adapter(key: str) -> type["ModelAdapter"]:
    """Look up the adapter class for a ``model_family`` key."""
    require(
        key in _REGISTRY,
        f"unknown model_family {key!r}; registered: {sorted(_REGISTRY)}",
    )
    return _REGISTRY[key]


def registered_adapters() -> Tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


@dataclass
class MMEncoding:
    """One VLM sample's multimodal input for the SRT rollout."""

    image: Any = None
    text: Optional[str] = None
    input_ids: Optional[List[int]] = None
    pixel_values: Any = None
    image_grid_thw: Any = None


@dataclass
class PreparedInputs:
    """One ``generate`` call's prepared driver-side state."""

    wire: List[Dict[str, Any]] = field(default_factory=list)
    prompt_token_ids: List[List[int]] = field(default_factory=list)
    resolved_n: int = 1
    mm: Optional[List[MMEncoding]] = None


class ModelAdapter(ABC):
    """Thin ABC: registry key + boilerplate defaults + the two conversion seams."""

    model_family: str = ""

    def __init__(
        self,
        config: Any,
        model_config: Any = None,
        *,
        tokenizer: Any,
        processor: Any = None,
    ) -> None:
        self.cfg = config
        self.model_config = model_config
        self._tokenizer = tokenizer
        self._processor = processor
        self.validate()

    def boot_kwargs(self) -> Dict[str, Any]:
        """Extra SGLang ServerArgs intent a model needs beyond the generic set."""
        return {}

    def validate(self) -> None:
        require(
            bool(getattr(self.cfg, "pretrained_model_ckpt_path", "")),
            f"{type(self).__name__} requires config.pretrained_model_ckpt_path",
        )
        require(
            self._tokenizer is not None,
            f"{type(self).__name__} requires a tokenizer",
        )

    def pad_token_id(self) -> int:
        pad = getattr(self._tokenizer, "pad_token_id", None) or getattr(self._tokenizer, "eos_token_id", None)
        return int(pad) if pad is not None else 0

    @abstractmethod
    def build_inputs(self, sample: Sample, *, sampling: ResolvedSampling) -> PreparedInputs:
        """Translate a request ``Sample`` into per-prompt SRT ``/generate`` payloads."""

    @abstractmethod
    def build_response(self, sample: Sample, prepared: PreparedInputs, raw: List[RawResult]) -> Sample:
        """Fill the frontier gen ``Part`` from the seam's results; return the ``Sample``."""


__all__ = [
    "MMEncoding",
    "ModelAdapter",
    "PreparedInputs",
    "get_adapter",
    "register_adapter",
    "registered_adapters",
]
