"""Sampling data types shared across engines, samplers, and actors."""

from __future__ import annotations

import math
from abc import ABC
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Set

from unirl.config.require import require

if TYPE_CHECKING:
    import torch

    from unirl.sde.index_schedule import TimestepScheduler


def _coerce(
    value: Any,
    target: type[int] | type[float] | type[bool],
    *,
    name: str,
    optional: bool = False,
) -> int | float | bool | None:
    """Coerce a scalar config value to ``target``, preserving optional ``None``."""
    if target is int:
        expected = "an integer"
    elif target is float:
        expected = "a finite float"
    elif target is bool:
        expected = "a boolean"
    else:
        raise ValueError(f"unsupported coercion target {target!r}")

    if value is None:
        if optional:
            return None
        raise TypeError(f"{name} must be {expected}, got None")

    if target is int:
        if type(value) is int:
            return value
        if isinstance(value, str):
            try:
                return int(value.strip(), 10)
            except ValueError as exc:
                raise TypeError(f"{name} must be {expected}, got {value!r}") from exc
    elif target is float and (type(value) in (int, float) or isinstance(value, str)):
        try:
            number = float(value.strip() if isinstance(value, str) else value)
        except (ValueError, OverflowError) as exc:
            raise TypeError(f"{name} must be {expected}, got {value!r}") from exc
        if not math.isfinite(number):
            raise ValueError(f"{name} must be {expected}, got {value!r}")
        return number
    elif target is bool and type(value) is bool:
        return value
    elif target is bool and isinstance(value, str):
        key = value.strip().lower()
        if key == "true":
            return True
        if key == "false":
            return False
    raise TypeError(f"{name} must be {expected}, got {value!r}")


@dataclass
class BaseSamplingParams(ABC):
    """Marker base for all sampling config dataclasses."""

    samples_per_prompt: int = 1

    def __post_init__(self) -> None:
        self.samples_per_prompt = _coerce(
            self.samples_per_prompt,
            int,
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
        self.num_inference_steps = _coerce(
            self.num_inference_steps,
            int,
            name=f"{prefix}.num_inference_steps",
        )
        self.guidance_scale = _coerce(self.guidance_scale, float, name=f"{prefix}.guidance_scale")
        self.height = _coerce(self.height, int, name=f"{prefix}.height")
        self.width = _coerce(self.width, int, name=f"{prefix}.width")
        self.num_frames = _coerce(self.num_frames, int, name=f"{prefix}.num_frames")
        self.seed = _coerce(self.seed, int, name=f"{prefix}.seed", optional=True)
        self.init_same_noise = _coerce(self.init_same_noise, bool, name=f"{prefix}.init_same_noise")
        self.disable_driver_xt = _coerce(self.disable_driver_xt, bool, name=f"{prefix}.disable_driver_xt")
        self.eta = _coerce(self.eta, float, name=f"{prefix}.eta")
        self.max_sequence_length = _coerce(
            self.max_sequence_length,
            int,
            name=f"{prefix}.max_sequence_length",
            optional=True,
        )
        self.taylor_cache_interval = _coerce(
            self.taylor_cache_interval,
            int,
            name=f"{prefix}.taylor_cache_interval",
            optional=True,
        )
        self.taylor_cache_order = _coerce(
            self.taylor_cache_order,
            int,
            name=f"{prefix}.taylor_cache_order",
            optional=True,
        )
        self.distilled_guidance_scale = _coerce(
            self.distilled_guidance_scale,
            float,
            name=f"{prefix}.distilled_guidance_scale",
            optional=True,
        )
        self.guidance_scale_2 = _coerce(
            self.guidance_scale_2,
            float,
            name=f"{prefix}.guidance_scale_2",
            optional=True,
        )
        self.strength = _coerce(self.strength, float, name=f"{prefix}.strength", optional=True)
        if self.init_noise_latent_shape is not None:
            name = f"{prefix}.init_noise_latent_shape"
            if isinstance(self.init_noise_latent_shape, (str, bytes)) or not isinstance(
                self.init_noise_latent_shape, Sequence
            ):
                raise TypeError(f"{name} must be a sequence of integers, got {self.init_noise_latent_shape!r}")
            self.init_noise_latent_shape = [
                _coerce(item, int, name=f"{name}[{index}]") for index, item in enumerate(self.init_noise_latent_shape)
            ]
        if self.sde_indices is not None:
            name = f"{prefix}.sde_indices"
            if isinstance(self.sde_indices, (str, bytes)) or not isinstance(self.sde_indices, Sequence):
                raise TypeError(f"{name} must be a sequence of integers, got {self.sde_indices!r}")
            self.sde_indices = [
                _coerce(item, int, name=f"{name}[{index}]") for index, item in enumerate(self.sde_indices)
            ]
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
        self.temperature = _coerce(self.temperature, float, name=f"{prefix}.temperature")
        self.max_new_tokens = _coerce(self.max_new_tokens, int, name=f"{prefix}.max_new_tokens")
        self.top_p = _coerce(self.top_p, float, name=f"{prefix}.top_p")
        self.top_k = _coerce(self.top_k, int, name=f"{prefix}.top_k")
        self.stop_token_id = _coerce(self.stop_token_id, int, name=f"{prefix}.stop_token_id", optional=True)
        self.seed = _coerce(self.seed, int, name=f"{prefix}.seed", optional=True)
