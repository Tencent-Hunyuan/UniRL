"""Sampling data types shared across engines, samplers, and actors."""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

from unirl.config.require import require

if TYPE_CHECKING:
    from unirl.utils.scheduler_utils import TimestepScheduler


@dataclass
class BaseSamplingParams(ABC):
    """Marker base for all sampling config dataclasses.

    Used as the type annotation / base class for the per-modality sampling
    config dataclasses.

    Holds the universal ``samples_per_prompt`` field — the per-prompt
    rollout fanout. For atomic params (Diffusion, AR) this is the samples
    generated per upstream input to that modality. For composed params it
    is the multiplicative total across modalities, computed in
    ``__post_init__``.
    """

    samples_per_prompt: int = 1


def get_diffusion_params(sampling: Any) -> "DiffusionSamplingParams":
    """Extract ``DiffusionSamplingParams`` from either pure or composed config.

    During the transition period, ``cfg.sampling`` may be either a bare
    ``DiffusionSamplingParams`` (legacy recipes) or a
    ``ComposedSamplingParams`` (composed recipes with ``.diffusion`` attr).
    This helper normalizes access.
    """
    return sampling.diffusion if hasattr(sampling, "diffusion") else sampling


def get_ar_params(sampling: Any) -> Optional["ARSamplingParams"]:
    """Extract ``ARSamplingParams`` from composed or bare AR config.

    Returns ``None`` for pure diffusion configs that have no AR component.
    """
    if hasattr(sampling, "ar"):
        return sampling.ar
    if isinstance(sampling, ARSamplingParams):
        return sampling
    return None


def is_forward_process(sde_indices: Optional[Sequence[int]]) -> bool:
    """True when the rollout records no SDE steps (deterministic ODE forward process).

    ``sde_indices`` (from :meth:`DiffusionSamplingParams.resolve_sde_indices`) names
    the denoising steps that draw per-step SDE noise. A non-empty list is an SDE
    rollout (FlowGRPO et al.); an empty list -- or ``None`` when no SDE params were
    set -- means every step is deterministic ODE, i.e. the DiffusionNFT-style forward
    process whose only output of interest is the final clean latent.

    This is the single source of truth for that interpretation: call sites read
    ``is_forward_process(sde_indices)`` instead of re-deriving it ad hoc from
    ``not x`` / ``x is None`` / ``len(x) == 0`` (those idioms had drifted apart and
    caused a trajectory-bandwidth regression once).
    """
    return not sde_indices


@dataclass
class DiffusionSamplingParams(BaseSamplingParams):
    """Canonical diffusion sampling params — single source of truth.

    Flows unchanged from YAML config → rollout pipeline → model pipeline.
    """

    # --- common (all diffusion models) ---
    # samples_per_prompt is inherited from BaseSamplingParams.
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    height: int = 256
    width: int = 256
    num_frames: int = 16
    seed: Optional[int] = 42
    init_same_noise: bool = False
    noise_group_ids: Optional[List[str]] = None

    # --- SDE ---
    eta: float = 1.0
    sde_strategy: Any = None  # StepStrategy
    scheduler: Any = None  # TimestepScheduler
    sde_indices: Optional[List[int]] = None

    # --- engine knobs ---
    sampler_kwargs: Dict[str, Any] = field(default_factory=dict)

    # --- precision ---
    autocast_precision: str = "bf16"
    trajectory_precision: str = "fp16"
    logprob_precision: str = "fp32"

    # --- model-specific (optional, unused fields ignored) ---
    max_sequence_length: Optional[int] = None
    taylor_cache_interval: Optional[int] = None
    taylor_cache_order: Optional[int] = None
    distilled_guidance_scale: Optional[float] = None
    guidance_scale_2: Optional[float] = None

    # --- backward compat (removed once all consumers migrate) ---
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

    def resolve_sde_indices(self, rollout_id: int) -> List[int]:
        """Resolve which denoising steps record SDE log-probs for ``rollout_id``.

        Precedence: an explicit static ``sde_indices`` list wins; else a
        ``scheduler`` instance (dynamic, keyed on ``rollout_id`` — window /
        sparse curricula); else every step. Setting ``sde_indices`` thus
        overrides any configured ``scheduler``.

        Not resolved at construction: a ``scheduler`` returns a different set
        per ``rollout_id``, so the result can't be frozen at init. The driver
        calls this per rollout and stamps the result onto a per-request copy
        (with ``scheduler=None``).
        """
        if self.sde_indices is not None:
            return [int(i) for i in self.sde_indices]
        scheduler: Optional[TimestepScheduler] = self.scheduler
        if scheduler is not None:
            return sorted(scheduler.get_sde_indices(int(rollout_id)))
        return list(range(int(self.num_inference_steps)))


@dataclass
class ARSamplingParams(BaseSamplingParams):
    """AR (autoregressive) sampling parameters for LLM-based PE generation."""

    # samples_per_prompt is inherited from BaseSamplingParams.
    temperature: float = 0.7
    max_new_tokens: int = 512
    top_p: float = 0.9
    top_k: int = 1024
    stop_token_id: int | None = None


@dataclass(kw_only=True)
class ComposedSamplingParams(BaseSamplingParams):
    """Composed sampling config with per-modality typed sampling params.

    ``kw_only=True`` is required because the inherited
    ``samples_per_prompt`` has a default while ``diffusion`` / ``ar``
    do not — without kw-only ordering, Python's dataclass rule
    "non-default after default" would fire.
    """

    diffusion: Any  # DiffusionSamplingParams
    ar: Any  # ARSamplingParams

    def __post_init__(self) -> None:
        # Per-prompt fanout for composed = product across modalities.
        # Each prompt → ar.samples_per_prompt AR outputs, each AR output →
        # diffusion.samples_per_prompt diffusion samples.
        self.samples_per_prompt = int(self.diffusion.samples_per_prompt) * int(self.ar.samples_per_prompt)
