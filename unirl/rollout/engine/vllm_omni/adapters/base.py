"""Driver-side ``Sample`` → ``Sample`` conversion: the adapter ABC + registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

from unirl.config.require import require
from unirl.rollout.engine.vllm_omni.backends import GenerateCall, OmniRawResult
from unirl.types.sample import Sample

_REGISTRY: Dict[str, type["ModelAdapter"]] = {}


def register_adapter(key: str):
    """Class decorator: register an adapter under its ``modality`` key."""

    def deco(cls: type["ModelAdapter"]) -> type["ModelAdapter"]:
        require(
            key not in _REGISTRY,
            f"adapter key {key!r} already registered by {_REGISTRY.get(key)!r}",
        )
        _REGISTRY[key] = cls
        cls.modality = key
        return cls

    return deco


def get_adapter(key: str) -> type["ModelAdapter"]:
    """Look up the adapter class for a ``modality`` key."""
    require(
        key in _REGISTRY,
        f"unknown modality {key!r}; registered: {sorted(_REGISTRY)}",
    )
    return _REGISTRY[key]


def registered_adapters() -> Tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


class ModelAdapter(ABC):
    """Thin ABC: registry key + topology knobs + the two conversion seams."""

    modality: str = ""

    stage_yaml: str = ""
    omni_mode: Optional[str] = None
    needs_sigmas: bool = True
    needs_driver_tokenizer: bool = True
    ar_lora_passthrough: bool = False
    clear_cuda_visible: bool = False
    lora_copy_transport: bool = False

    def __init__(
        self,
        config: Any,
        model_config: Any,
        *,
        strategy: Any = None,
        tokenize_fn: Optional[Callable[..., List[int]]] = None,
    ) -> None:
        self.cfg = config
        self.model_config = model_config
        self.tokenize_fn = tokenize_fn
        self._sde_label = self.resolve_sde_label(strategy)
        self.validate()

    @staticmethod
    def resolve_sde_label(strategy: Any) -> Optional[str]:
        """Deliberately ``None``: vllm-omni rides raw ``eta`` + ``sde_indices`` via ``extra_args``; do not complete."""
        del strategy
        return None

    def boot_kwargs(self) -> Dict[str, Any]:
        """Model-specific boot intent beyond the generic config spelling."""
        require(bool(self.stage_yaml), f"{type(self).__name__} must set stage_yaml")
        kwargs: Dict[str, Any] = {
            "stage_yaml": self.stage_yaml,
            "needs_driver_tokenizer": bool(self.needs_driver_tokenizer),
            "clear_cuda_visible": bool(self.clear_cuda_visible),
        }
        if self.omni_mode is not None:
            kwargs["mode"] = self.omni_mode
        return kwargs

    def schedule_policy(self) -> Any:
        from unirl.sde.runtime import FlowMatchSchedulePolicy

        mc = self.model_config
        return FlowMatchSchedulePolicy.from_pretrained(
            self.cfg.model_path,
            shift=float(mc.shift),
            require_dynamic=bool(getattr(mc, "use_dynamic_shifting", False)),
            dynamic_overrides=getattr(mc, "dynamic_shift_overrides", None),
        )

    def validate(self) -> None:
        if self.needs_sigmas:
            mc = self.model_config
            require(
                mc is not None and hasattr(mc, "shift"),
                f"{type(self).__name__} requires model_config.shift; got "
                f"{type(mc).__name__}. Use a registered model preset "
                f"(e.g. ``sd3``, ``wan21``, ``wan22``, ``hunyuan_image3``).",
            )

    def validate_request(self, sample: Sample) -> None:
        """Modality-specific request gate; default accepts everything."""

    @abstractmethod
    def build_inputs(self, sample: Sample) -> List[GenerateCall]:
        """Translate a request ``Sample`` into the seam's generate calls."""

    @abstractmethod
    def build_response(self, sample: Sample, per_request: List[List[OmniRawResult]]) -> Sample:
        """Fill the request ``Sample``'s gen Parts from the seam's per-request-grouped results."""


__all__ = [
    "ModelAdapter",
    "get_adapter",
    "register_adapter",
    "registered_adapters",
]
