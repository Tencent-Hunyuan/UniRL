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


def _coerce_type(
    value: Any,
    target: type[int] | type[float] | type[bool],
    name: str,
    optional: bool = False,
) -> int | float | bool | None:
    """Coerce a scalar config value to ``target``, preserving optional ``None``."""
    expected = "finite float" if target is float else target.__name__
    message = f"{name} must be {expected}, got {value!r}"
    if value is None:
        if optional:
            return None
        raise TypeError(message)

    if type(value) is target:
        if target is float and not math.isfinite(value):
            raise ValueError(message)
        return value
    if target is float and (type(value) is int or isinstance(value, str)):
        try:
            number = float(value)
        except (ValueError, OverflowError) as exc:
            raise TypeError(message) from exc
        if not math.isfinite(number):
            raise ValueError(message)
        return number
    if target is int and isinstance(value, str):
        try:
            return int(value.strip(), 10)
        except ValueError as exc:
            raise TypeError(message) from exc
    if target is bool and isinstance(value, str):
        key = value.strip().lower()
        if key in ("true", "false"):
            return key == "true"
    raise TypeError(message)


@dataclass
class BaseSamplingParams(ABC):
    """Marker base for all sampling config dataclasses."""

    samples_per_prompt: int = 1

    def __post_init__(self) -> None:
        cls = type(self).__name__
        self.samples_per_prompt = _coerce_type(self.samples_per_prompt, int, f"{cls}.samples_per_prompt")


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
    # Rollout-engine execution policy. Core model pipelines intentionally ignore
    # this field; engines with a quantized inference copy may use ``fp8`` for a
    # low-precision scout while the BF16 regeneration keeps ``bf16``.
    rollout_precision: str = "bf16"
    # Optional scout-only transport optimization: resize decoded images on the
    # rollout GPU before reward dispatch (preference scorers resize internally
    # anyway). The high-fidelity train/eval configs leave this unset.
    reward_image_size: Optional[int] = None

    max_sequence_length: Optional[int] = None
    taylor_cache_interval: Optional[int] = None
    taylor_cache_order: Optional[int] = None
    distilled_guidance_scale: Optional[float] = None
    guidance_scale_2: Optional[float] = None
    strength: Optional[float] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        cls = type(self).__name__
        self.num_inference_steps = _coerce_type(self.num_inference_steps, int, f"{cls}.num_inference_steps")
        self.guidance_scale = _coerce_type(self.guidance_scale, float, f"{cls}.guidance_scale")
        self.height = _coerce_type(self.height, int, f"{cls}.height")
        self.width = _coerce_type(self.width, int, f"{cls}.width")
        self.num_frames = _coerce_type(self.num_frames, int, f"{cls}.num_frames")
        self.seed = _coerce_type(self.seed, int, f"{cls}.seed", True)
        self.init_same_noise = _coerce_type(self.init_same_noise, bool, f"{cls}.init_same_noise")
        self.disable_driver_xt = _coerce_type(self.disable_driver_xt, bool, f"{cls}.disable_driver_xt")
        self.eta = _coerce_type(self.eta, float, f"{cls}.eta")
        self.max_sequence_length = _coerce_type(self.max_sequence_length, int, f"{cls}.max_sequence_length", True)
        self.taylor_cache_interval = _coerce_type(
            self.taylor_cache_interval,
            int,
            f"{cls}.taylor_cache_interval",
            True,
        )
        self.taylor_cache_order = _coerce_type(self.taylor_cache_order, int, f"{cls}.taylor_cache_order", True)
        self.distilled_guidance_scale = _coerce_type(
            self.distilled_guidance_scale,
            float,
            f"{cls}.distilled_guidance_scale",
            True,
        )
        self.guidance_scale_2 = _coerce_type(self.guidance_scale_2, float, f"{cls}.guidance_scale_2", True)
        self.strength = _coerce_type(self.strength, float, f"{cls}.strength", True)
        if self.init_noise_latent_shape is not None:
            name = f"{cls}.init_noise_latent_shape"
            if isinstance(self.init_noise_latent_shape, (str, bytes)) or not isinstance(
                self.init_noise_latent_shape, Sequence
            ):
                raise TypeError(f"{name} must be a sequence of integers, got {self.init_noise_latent_shape!r}")
            self.init_noise_latent_shape = [
                _coerce_type(item, int, f"{name}[{index}]") for index, item in enumerate(self.init_noise_latent_shape)
            ]
        if self.sde_indices is not None:
            name = f"{cls}.sde_indices"
            if isinstance(self.sde_indices, (str, bytes)) or not isinstance(self.sde_indices, Sequence):
                raise TypeError(f"{name} must be a sequence of integers, got {self.sde_indices!r}")
            self.sde_indices = [
                _coerce_type(item, int, f"{name}[{index}]") for index, item in enumerate(self.sde_indices)
            ]
        reserved = {f.name for f in fields(self) if f.name != "sampler_kwargs"}
        shadowed = reserved & set(self.sampler_kwargs)
        require(
            not shadowed,
            f"{cls}.sampler_kwargs cannot contain reserved keys {sorted(shadowed)}; set them as fields instead",
        )
        require(
            self.rollout_precision in {"bf16", "fp8"},
            f"DiffusionSamplingParams.rollout_precision must be bf16|fp8, got {self.rollout_precision!r}",
        )
        require(
            self.reward_image_size is None
            or (
                isinstance(self.reward_image_size, int)
                and not isinstance(self.reward_image_size, bool)
                and self.reward_image_size > 0
            ),
            f"DiffusionSamplingParams.reward_image_size must be positive when set, got {self.reward_image_size!r}",
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
        cls = type(self).__name__
        self.temperature = _coerce_type(self.temperature, float, f"{cls}.temperature")
        self.max_new_tokens = _coerce_type(self.max_new_tokens, int, f"{cls}.max_new_tokens")
        self.top_p = _coerce_type(self.top_p, float, f"{cls}.top_p")
        self.top_k = _coerce_type(self.top_k, int, f"{cls}.top_k")
        self.stop_token_id = _coerce_type(self.stop_token_id, int, f"{cls}.stop_token_id", True)
        self.seed = _coerce_type(self.seed, int, f"{cls}.seed", True)
