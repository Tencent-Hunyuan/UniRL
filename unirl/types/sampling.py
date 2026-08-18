"""Sampling data types shared across engines, samplers, and actors."""

from __future__ import annotations

from abc import ABC
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Set

from unirl.config.require import require

if TYPE_CHECKING:
    import torch

    from unirl.utils.scheduler_utils import TimestepScheduler


@dataclass
class BaseSamplingParams(ABC):
    """Marker base for all sampling config dataclasses."""

    samples_per_prompt: int = 1


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

    num_samples_per_prompt: int = 1

    def __post_init__(self) -> None:
        if self.num_samples_per_prompt != 1 and self.samples_per_prompt == 1:
            object.__setattr__(self, "samples_per_prompt", self.num_samples_per_prompt)
        elif self.samples_per_prompt != 1 and self.num_samples_per_prompt == 1:
            object.__setattr__(self, "num_samples_per_prompt", self.samples_per_prompt)

        reserved = {f.name for f in fields(self) if f.name != "sampler_kwargs"}
        shadowed = reserved & set(self.sampler_kwargs)
        require(
            not shadowed,
            f"DiffusionSamplingParams.sampler_kwargs cannot contain reserved keys {sorted(shadowed)}; set them as fields instead",
        )
        require(
            self.rollout_precision in {"bf16", "fp8"},
            f"DiffusionSamplingParams.rollout_precision must be bf16|fp8, got {self.rollout_precision!r}",
        )
        require(
            self.reward_image_size is None or int(self.reward_image_size) > 0,
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

    temperature: float = 0.7
    max_new_tokens: int = 512
    top_p: float = 0.9
    top_k: int = 0
    stop_token_id: int | None = None
