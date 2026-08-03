"""Model-specific boundary for the FastVideo rollout engine."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from unirl.config.require import require
from unirl.sde.noise import _derive_group_seed

_REGISTRY: Dict[str, type["FastVideoModelAdapter"]] = {}


def register_adapter(*keys: str):
    normalized = tuple(str(key).strip().lower() for key in keys)

    def decorator(cls: type["FastVideoModelAdapter"]) -> type["FastVideoModelAdapter"]:
        for key in normalized:
            require(key and key not in _REGISTRY, f"FastVideo adapter key {key!r} is already registered")
            _REGISTRY[key] = cls
        cls.model_families = normalized
        return cls

    return decorator


def get_adapter(key: str) -> type["FastVideoModelAdapter"]:
    normalized = str(key).strip().lower()
    require(normalized in _REGISTRY, f"unknown FastVideo model_family {key!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[normalized]


def registered_adapters() -> Tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


class FastVideoModelAdapter(ABC):
    """Own every model-specific schedule, shape, and response assumption."""

    model_families: Tuple[str, ...] = ()

    def __init__(self, config: Any, model_config: Any, *, strategy: Any = None) -> None:
        self.cfg = config
        self.model_config = model_config
        self.strategy = strategy
        self.validate()

    def validate(self) -> None:
        require(
            self.model_config is not None and bool(getattr(self.model_config, "pretrained_model_ckpt_path", None)),
            f"{type(self).__name__} requires model_config.pretrained_model_ckpt_path",
        )

    def schedule_policy(self) -> Any:
        from unirl.sde.runtime import FlowMatchSchedulePolicy

        return FlowMatchSchedulePolicy.from_pretrained(
            self.model_config.pretrained_model_ckpt_path,
            shift=float(self.model_config.shift),
            require_dynamic=bool(getattr(self.model_config, "use_dynamic_shifting", False)),
            dynamic_overrides=getattr(self.model_config, "dynamic_shift_overrides", None),
        )

    def per_sample_seeds(self, req: Any, params: Any) -> List[int]:
        batch_size = int(req.batch_size)
        base_seed = int(params.seed)
        keys = getattr(req, "init_noise_group_ids", None)
        if not (isinstance(keys, (list, tuple)) and len(keys) == batch_size):
            same = bool(getattr(params, "init_same_noise", False))
            keys = list(req.group_ids) if same else list(req.sample_ids)
        if not (isinstance(keys, (list, tuple)) and len(keys) == batch_size):
            return [base_seed] * batch_size
        return [_derive_group_seed(base_seed, str(key)) for key in keys]

    @abstractmethod
    def align_runtime_args(self, fastvideo_args: Any) -> None:
        """Validate and apply model-specific FastVideo boot configuration."""

    @abstractmethod
    def build_forward_batch(
        self,
        *,
        prompt: str,
        seed: int,
        params: Any,
        sigmas: Any,
        fastvideo_args: Any,
    ) -> Any:
        """Build one FastVideo ``ForwardBatch`` without importing it in engine.py."""

    @abstractmethod
    def collect_output(self, output: Any) -> Dict[str, Any]:
        """Normalize one FastVideo output into CPU-side adapter data."""

    @abstractmethod
    def build_response(self, req: Any, params: Any, samples: List[Dict[str, Any]]) -> Any:
        """Build the typed UniRL response for a list of per-sample outputs."""


__all__ = [
    "FastVideoModelAdapter",
    "get_adapter",
    "register_adapter",
    "registered_adapters",
]
