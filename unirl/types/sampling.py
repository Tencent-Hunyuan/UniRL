"""Sampling data types shared across engines, samplers, and actors."""

from __future__ import annotations

import math
from abc import ABC
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Set

from unirl.config.require import require

if TYPE_CHECKING:
    import torch

    from unirl.sde.index_schedule import TimestepScheduler

_BOOL_TRUE = {"true", "1"}
_BOOL_FALSE = {"false", "0"}


def _as_int(value: Any, *, name: str, optional: bool = False) -> int | None:
    """Coerce ``value`` to ``int``, or ``None`` when ``optional``."""
    if value is None:
        if optional:
            return None
        raise TypeError(f"{name} must be an integer, got None")
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        try:
            number = float(text)
        except ValueError as exc:
            raise TypeError(f"{name} must be an integer, got {value!r}") from exc
    elif isinstance(value, Real):
        number = float(value)
    else:
        raise TypeError(f"{name} must be an integer, got {value!r}")
    if not math.isfinite(number) or number != int(number):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return int(number)


def _as_float(value: Any, *, name: str, optional: bool = False) -> float | None:
    """Coerce ``value`` to a finite ``float``, or ``None`` when ``optional``."""
    if value is None:
        if optional:
            return None
        raise TypeError(f"{name} must be a finite float, got None")
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite float, got {value!r}")
    if isinstance(value, str):
        value = value.strip()
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite float, got {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite float, got {value!r}")
    return number


def _as_bool(value: Any, *, name: str) -> bool:
    """Coerce ``value`` to ``bool`` without treating nonempty strings as true."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        key = value.strip().lower()
        if key in _BOOL_TRUE:
            return True
        if key in _BOOL_FALSE:
            return False
        raise TypeError(f"{name} must be a boolean, got {value!r}")
    if isinstance(value, Integral) and value in (0, 1):
        return bool(value)
    raise TypeError(f"{name} must be a boolean, got {value!r}")


def _as_int_list(value: Any, *, name: str, optional: bool = False) -> list[int] | None:
    """Coerce ``value`` to ``list[int]``, or ``None`` when ``optional``."""
    if value is None:
        if optional:
            return None
        raise TypeError(f"{name} must be a sequence of integers, got None")
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a sequence of integers, got {value!r}")
    return [_as_int(item, name=f"{name}[{index}]") for index, item in enumerate(value)]


@dataclass
class BaseSamplingParams(ABC):
    """Marker base for all sampling config dataclasses."""

    samples_per_prompt: int = 1

    def __post_init__(self) -> None:
        self.samples_per_prompt = _as_int(
            self.samples_per_prompt,
            name=f"{type(self).__name__}.samples_per_prompt",
        )


def _is_param_dict(sampling: Any) -> bool:
    """True iff ``sampling`` is a modality-keyed mapping rather than a single sampling-params object."""
    return isinstance(sampling, Mapping) and ("diffusion" in sampling or "ar" in sampling)


def total_samples_per_prompt(sampling: Any) -> int:
    """Per-prompt rollout fan-out: the product of each modality's ``samples_per_prompt``."""
    if sampling is None:
        return 1
    if _is_param_dict(sampling):
        total = 1
        for params in sampling.values():
            total *= int(getattr(params, "samples_per_prompt", 1))
        return total
    return int(getattr(sampling, "samples_per_prompt", 1))


def is_forward_process(sde_indices: Optional[Sequence[int]]) -> bool:
    """True when the rollout records no SDE steps (deterministic ODE forward process)."""
    return not sde_indices


def compute_trajectory_positions(sde_indices: Set[int], num_steps: int) -> List[int]:
    """Return sorted positions needed for ``(x_t, x_{t+1})`` pairs at SDE boundaries."""
    positions: Set[int] = set()
    for i in sde_indices:
        positions.add(max(0, min(i, num_steps)))
        positions.add(max(0, min(i + 1, num_steps)))
    return sorted(positions)


@dataclass
class DiffusionSamplingParams(BaseSamplingParams):
    """Canonical diffusion sampling params — single source of truth."""

    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    height: int = 256
    width: int = 256
    num_frames: int = 16
    seed: Optional[int] = 42
    init_same_noise: bool = False
    noise_group_ids: Optional[List[str]] = None
    init_noise_latent_shape: Optional[List[int]] = None
    # Debug opt-out: let each rollout engine generate its own initial noise.
    disable_driver_xt: bool = False
    sigmas: Optional[torch.Tensor] = None

    eta: float = 1.0
    sde_strategy: Any = None
    scheduler: Any = None
    sde_indices: Optional[List[int]] = None

    sampler_kwargs: Dict[str, Any] = field(default_factory=dict)

    autocast_precision: str = "bf16"
    trajectory_precision: str = "fp16"
    logprob_precision: str = "fp32"

    max_sequence_length: Optional[int] = None
    taylor_cache_interval: Optional[int] = None
    taylor_cache_order: Optional[int] = None
    distilled_guidance_scale: Optional[float] = None
    guidance_scale_2: Optional[float] = None
    strength: Optional[float] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        prefix = type(self).__name__
        self.num_inference_steps = _as_int(self.num_inference_steps, name=f"{prefix}.num_inference_steps")
        self.guidance_scale = _as_float(self.guidance_scale, name=f"{prefix}.guidance_scale")
        self.height = _as_int(self.height, name=f"{prefix}.height")
        self.width = _as_int(self.width, name=f"{prefix}.width")
        self.num_frames = _as_int(self.num_frames, name=f"{prefix}.num_frames")
        self.seed = _as_int(self.seed, name=f"{prefix}.seed", optional=True)
        self.init_same_noise = _as_bool(self.init_same_noise, name=f"{prefix}.init_same_noise")
        self.init_noise_latent_shape = _as_int_list(
            self.init_noise_latent_shape,
            name=f"{prefix}.init_noise_latent_shape",
            optional=True,
        )
        self.disable_driver_xt = _as_bool(self.disable_driver_xt, name=f"{prefix}.disable_driver_xt")
        self.eta = _as_float(self.eta, name=f"{prefix}.eta")
        self.sde_indices = _as_int_list(self.sde_indices, name=f"{prefix}.sde_indices", optional=True)
        self.max_sequence_length = _as_int(
            self.max_sequence_length,
            name=f"{prefix}.max_sequence_length",
            optional=True,
        )
        self.taylor_cache_interval = _as_int(
            self.taylor_cache_interval,
            name=f"{prefix}.taylor_cache_interval",
            optional=True,
        )
        self.taylor_cache_order = _as_int(
            self.taylor_cache_order,
            name=f"{prefix}.taylor_cache_order",
            optional=True,
        )
        self.distilled_guidance_scale = _as_float(
            self.distilled_guidance_scale,
            name=f"{prefix}.distilled_guidance_scale",
            optional=True,
        )
        self.guidance_scale_2 = _as_float(self.guidance_scale_2, name=f"{prefix}.guidance_scale_2", optional=True)
        self.strength = _as_float(self.strength, name=f"{prefix}.strength", optional=True)
        reserved = {f.name for f in fields(self) if f.name != "sampler_kwargs"}
        shadowed = reserved & set(self.sampler_kwargs)
        require(
            not shadowed,
            f"{prefix}.sampler_kwargs cannot contain reserved keys {sorted(shadowed)}; set them as fields instead",
        )

    def resolve_sde_indices(self, rollout_id: int) -> List[int]:
        """Resolve which denoising steps record SDE log-probs for ``rollout_id``."""
        if self.sde_indices is not None:
            return [int(i) for i in self.sde_indices]
        scheduler: Optional[TimestepScheduler] = self.scheduler
        if scheduler is not None:
            return sorted(scheduler.get_sde_indices(int(rollout_id)))
        return list(range(int(self.num_inference_steps)))


@dataclass
class ARSamplingParams(BaseSamplingParams):
    """AR (autoregressive) sampling parameters for LLM-based PE generation."""

    emits_fixed_length: ClassVar[bool] = False

    temperature: float = 0.7
    max_new_tokens: int = 512
    top_p: float = 0.9
    top_k: int = 0
    stop_token_id: int | None = None
    seed: Optional[int] = None  # engines with per-request seeded sampling derive child seeds from this + sample_id

    def __post_init__(self) -> None:
        super().__post_init__()
        prefix = type(self).__name__
        self.temperature = _as_float(self.temperature, name=f"{prefix}.temperature")
        self.max_new_tokens = _as_int(self.max_new_tokens, name=f"{prefix}.max_new_tokens")
        self.top_p = _as_float(self.top_p, name=f"{prefix}.top_p")
        self.top_k = _as_int(self.top_k, name=f"{prefix}.top_k")
        self.stop_token_id = _as_int(self.stop_token_id, name=f"{prefix}.stop_token_id", optional=True)
        self.seed = _as_int(self.seed, name=f"{prefix}.seed", optional=True)
