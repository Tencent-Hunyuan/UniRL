"""Reusable single-stream diffusion sampling and replay.

Model packages own transformer adaptation, conditioning, latent geometry, and
small per-step kwargs. This implementation owns the invariant control flow
shared by image and video stages with one dense latent stream:

* schedule validation and strategy initialization;
* initial-latent validation / fallback generation;
* sparse trajectory and SDE log-prob collection;
* stored-transition replay and Gaussian-mean collection;
* autocast and precision policy;
* single-step noise prediction and the default trainable surface.

The hook surface is deliberately narrow. A model should override latent
specification and only the kwargs/state/segment behavior its kernel actually
needs. Packed geometry and multi-stream policies whose transition itself is
different (for example BAGEL Navit and LTX2 joint audio+video SDE) remain
specialized stages.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Generic,
    List,
    Mapping,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    runtime_checkable,
)

import torch

from unirl.sde.kernels import StepStrategy
from unirl.types.sampling import DiffusionSamplingParams, compute_trajectory_positions
from unirl.types.segments.latent import LatentSegment, make_video_segment
from unirl.utils.dtypes import parse_torch_dtype

if TYPE_CHECKING:
    from unirl.models.types.replay_result import ReplayResult

B = TypeVar("B")
C = TypeVar("C")


@dataclass(frozen=True)
class SingleStreamLatentSpec:
    """Device, batch, and dense per-sample shape for one latent stream."""

    device: torch.device
    batch_size: int
    shape: Tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "device", torch.device(self.device))
        if self.batch_size <= 0:
            raise ValueError(f"SingleStreamLatentSpec.batch_size must be positive; got {self.batch_size}.")
        if not self.shape or any(int(size) <= 0 for size in self.shape):
            raise ValueError(f"SingleStreamLatentSpec.shape must contain positive dimensions; got {self.shape!r}.")


@runtime_checkable
class SingleStreamDiffusionStep(Protocol[B, C]):
    """One transition for a dense latent stream on a scalar sigma schedule.

    The kernel is stateless: it takes a model bundle, conditions, and an SDE
    strategy per call, runs the model forward, and applies one transition.
    ``prev_sample=None`` selects sampling; a stored ``prev_sample`` selects
    log-prob replay.
    """

    def forward(
        self,
        *,
        strategy: "StepStrategy",
        noise_pred: torch.Tensor,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        prev_sample: Optional[torch.Tensor] = None,
        sigma_max: float = 0.99,
        eta: float = 1.0,
        step_index: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]: ...

    def step(
        self,
        model: B,
        conditions: C,
        *,
        strategy: "StepStrategy",
        sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        guidance_scale: float,
        prev_sample: Optional[torch.Tensor] = None,
        sigma_max: float = 0.99,
        eta: float = 1.0,
        step_index: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]: ...

    def step_with_logp(
        self,
        model: B,
        conditions: C,
        *,
        strategy: "StepStrategy",
        sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        guidance_scale: float,
        prev_sample: Optional[torch.Tensor] = None,
        sigma_max: float = 0.99,
        eta: float = 1.0,
        step_index: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]: ...


@contextmanager
def temporary_eval(module: torch.nn.Module):
    """Temporarily put ``module`` in eval mode and restore its prior state."""
    was_training = module.training
    module.eval()
    try:
        yield
    finally:
        module.train(was_training)


class SingleStreamDiffusionRunner(ABC, Generic[B, C]):
    """Denoising and stored-transition replay for one dense latent stream.

    Subclasses implement :meth:`_latent_spec`.  Optional hooks expose
    model-specific step kwargs, state, segment modality, replay validation, and
    temporary model mode without copying either loop.
    """

    SEGMENT_FACTORY: ClassVar[Callable[..., LatentSegment]] = LatentSegment
    SIGMA_MAX_AS_FLOAT: ClassVar[bool] = False

    def __init__(
        self,
        *,
        model: B,
        step: Any,
        strategy: StepStrategy,
        autocast_precision: str = "bf16",
        trajectory_precision: str = "fp16",
        logprob_precision: str = "fp32",
    ) -> None:
        self.model = model
        self.step = step
        self.strategy = strategy
        self.autocast_dtype = parse_torch_dtype(autocast_precision, field_name="autocast_precision")
        self.trajectory_dtype = parse_torch_dtype(trajectory_precision, field_name="trajectory_precision")
        self.logprob_dtype = parse_torch_dtype(logprob_precision, field_name="logprob_precision")

    @property
    def _runner_name(self) -> str:
        return type(self).__name__

    @abstractmethod
    def _latent_spec(self, conditions: C, params: DiffusionSamplingParams) -> SingleStreamLatentSpec:
        """Resolve the request's device, batch size, and per-sample latent shape."""

    def _prepare_initial_latents(
        self,
        spec: SingleStreamLatentSpec,
        params: DiffusionSamplingParams,
        initial_latents: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Validate driver-provided x_T or use the framework fallback generator."""
        if initial_latents is not None:
            expected = (spec.batch_size, *spec.shape)
            if tuple(initial_latents.shape) != expected:
                raise ValueError(
                    f"{self._runner_name}.diffuse: initial_latents shape "
                    f"{tuple(initial_latents.shape)} != expected {expected}."
                )
            return initial_latents.to(device=spec.device, dtype=self.trajectory_dtype)

        from unirl.sde.noise import generate_latents

        return generate_latents(
            batch_size=spec.batch_size,
            latent_shape=spec.shape,
            device=spec.device,
            dtype=self.trajectory_dtype,
            init_same_noise=bool(params.init_same_noise),
            samples_per_prompt=int(params.samples_per_prompt),
            noise_group_ids=params.noise_group_ids,
            base_seed=None if params.seed is None else int(params.seed),
        )

    def _prepare_conditions(
        self,
        conditions: C,
        params: DiffusionSamplingParams,
        *,
        mode: str,
    ) -> C:
        """Prepare model conditions for sampling, replay, or prediction."""
        del params, mode
        return conditions

    def _sampling_state(
        self,
        conditions: C,
        params: DiffusionSamplingParams,
        *,
        schedule: torch.Tensor,
        spec: SingleStreamLatentSpec,
    ) -> Any:
        """Create optional state shared by every step of one rollout."""
        del conditions, params, schedule, spec
        return None

    def _before_diffuse(
        self,
        conditions: C,
        params: DiffusionSamplingParams,
        *,
        schedule: torch.Tensor,
        spec: SingleStreamLatentSpec,
    ) -> None:
        """Optional model-mode/cache setup immediately before the sampling loop."""
        del conditions, params, schedule, spec

    def _step_kwargs(
        self,
        conditions: C,
        params: DiffusionSamplingParams,
        *,
        sample: torch.Tensor,
        step_index: int,
        num_steps: int,
        mode: str,
        state: Any,
    ) -> Mapping[str, Any]:
        """Extra kwargs for a concrete model's step kernel."""
        del conditions, params, sample, step_index, num_steps, mode, state
        return {}

    def _step_generator(
        self,
        params: DiffusionSamplingParams,
        *,
        spec: SingleStreamLatentSpec,
        step_index: int,
        eta: float,
        sde_sample_keys: Optional[List[str]],
    ) -> Any:
        """Return an optional model-specific SDE generator for one step."""
        del params, spec, step_index, eta, sde_sample_keys
        return None

    def _guidance_scale(
        self,
        params: DiffusionSamplingParams,
        *,
        step_index: int,
        num_steps: int,
        mode: str,
    ) -> float:
        del step_index, num_steps, mode
        return float(params.guidance_scale)

    def _sigma_max(self, schedule: torch.Tensor) -> Any:
        value: Any = schedule[1].float() if int(schedule.shape[0]) > 1 else schedule.new_tensor(0.99).float()
        return float(value.item()) if self.SIGMA_MAX_AS_FLOAT else value

    def _autocast_context(self, device: torch.device) -> AbstractContextManager:
        if device.type == "cuda" and self.autocast_dtype in (torch.float16, torch.bfloat16):
            return torch.autocast("cuda", self.autocast_dtype)
        return nullcontext()

    def _make_segment(
        self,
        *,
        latents: torch.Tensor,
        sigmas: torch.Tensor,
        indices: torch.Tensor,
        sde_logp: Optional[torch.Tensor],
        sde_indices: Optional[torch.Tensor],
    ) -> LatentSegment:
        return type(self).SEGMENT_FACTORY(
            latents=latents,
            sigmas=sigmas,
            indices=indices,
            sde_logp=sde_logp,
            sde_indices=sde_indices,
        )

    def diffuse(
        self,
        conditions: C,
        *,
        schedule: torch.Tensor,
        params: DiffusionSamplingParams,
        initial_latents: Optional[torch.Tensor] = None,
        sde_sample_keys: Optional[List[str]] = None,
    ) -> LatentSegment:
        """Run the shared denoising loop and store sparse replay transitions."""
        spec = self._latent_spec(conditions, params)
        num_steps = int(params.num_inference_steps)
        if int(schedule.shape[0]) != num_steps + 1:
            raise ValueError(
                f"{self._runner_name}.diffuse: schedule length {schedule.shape[0]} "
                f"!= num_inference_steps+1 ({num_steps + 1})."
            )
        schedule = schedule.to(spec.device)
        self.strategy.init_schedule(schedule)
        latents = self._prepare_initial_latents(spec, params, initial_latents)
        conditions = self._prepare_conditions(conditions, params, mode="sample")

        sde_set = {int(index) for index in (params.sde_indices or [])}
        sde_sorted = sorted(sde_set)
        needed = set(compute_trajectory_positions(sde_set, num_steps))
        needed.add(num_steps)

        stored_pairs: List[tuple[int, torch.Tensor]] = []
        if 0 in needed:
            stored_pairs.append((0, latents.detach().clone()))
        sde_log_probs: List[torch.Tensor] = []

        sigma_max = self._sigma_max(schedule)
        state = self._sampling_state(conditions, params, schedule=schedule, spec=spec)
        self._before_diffuse(conditions, params, schedule=schedule, spec=spec)

        for step_index in range(num_steps):
            sigma = schedule[step_index].to(spec.device)
            sigma_next = schedule[step_index + 1].to(spec.device)
            step_eta = float(params.eta) if step_index in sde_set else 0.0
            step_kwargs = dict(
                self._step_kwargs(
                    conditions,
                    params,
                    sample=latents,
                    step_index=step_index,
                    num_steps=num_steps,
                    mode="sample",
                    state=state,
                )
            )
            generator = self._step_generator(
                params,
                spec=spec,
                step_index=step_index,
                eta=step_eta,
                sde_sample_keys=sde_sample_keys,
            )
            if generator is not None:
                step_kwargs["generator"] = generator
            with torch.no_grad(), self._autocast_context(spec.device):
                new_latents, log_prob, _ = self.step.step_with_logp(
                    self.model,
                    conditions,
                    strategy=self.strategy,
                    sample=latents,
                    sigma=sigma,
                    sigma_next=sigma_next,
                    guidance_scale=self._guidance_scale(
                        params,
                        step_index=step_index,
                        num_steps=num_steps,
                        mode="sample",
                    ),
                    eta=step_eta,
                    sigma_max=sigma_max,
                    step_index=step_index,
                    **step_kwargs,
                )
            latents = new_latents.to(dtype=self.trajectory_dtype)
            if (step_index + 1) in needed:
                stored_pairs.append((step_index + 1, latents.detach().clone()))
            if log_prob is not None:
                sde_log_probs.append(log_prob.to(dtype=self.logprob_dtype))

        positions = [position for position, _ in stored_pairs]
        latents_stacked = torch.stack([value for _, value in stored_pairs], dim=1)
        sde_logp = torch.stack(sde_log_probs, dim=1) if sde_log_probs else None
        sde_indices = torch.tensor(sde_sorted, dtype=torch.long, device=spec.device) if sde_sorted else None
        indices = torch.tensor(positions, dtype=torch.long, device=spec.device)
        return self._make_segment(
            latents=latents_stacked,
            sigmas=schedule,
            indices=indices,
            sde_logp=sde_logp,
            sde_indices=sde_indices,
        )

    def _validate_replay_segment(self, segment: LatentSegment) -> None:
        if segment.sde_indices is None or segment.latents is None:
            raise ValueError(f"{self._runner_name}.replay: segment.sde_indices / latents missing.")
        if segment.sigmas is None:
            raise ValueError(f"{self._runner_name}.replay: segment.sigmas missing.")

    def _replay_device(self, segment: LatentSegment) -> torch.device:
        model_device = getattr(self.model, "device", None)
        return torch.device(model_device) if model_device is not None else segment.latents.device

    def _replay_context(self) -> AbstractContextManager:
        """Optional temporary model-mode context around replay."""
        return nullcontext()

    def _replay_batched(
        self,
        conditions: C,
        *,
        segment: LatentSegment,
        params: DiffusionSamplingParams,
        target: List[int],
        sigmas: torch.Tensor,
        sigma_max: Any,
        device: torch.device,
    ) -> Optional[ReplayResult]:
        """Optional one-forward replay fast path; serial replay is the default."""
        del conditions, segment, params, target, sigmas, sigma_max, device
        return None

    def replay(
        self,
        conditions: C,
        *,
        segment: LatentSegment,
        params: DiffusionSamplingParams,
        step_indices: Optional[List[int]] = None,
    ) -> ReplayResult:
        """Re-score stored SDE transitions with the current model."""
        self._validate_replay_segment(segment)
        conditions = self._prepare_conditions(conditions, params, mode="replay")
        available = [int(index) for index in segment.sde_indices.tolist()]
        available_set = set(available)
        target = [int(index) for index in step_indices] if step_indices is not None else available
        unknown = [index for index in target if index not in available_set]
        if unknown:
            raise ValueError(
                f"{self._runner_name}.replay: step_indices {unknown} not in "
                f"segment.sde_indices={sorted(available_set)}."
            )

        device = self._replay_device(segment)
        sigmas = segment.sigmas.to(device)
        sigma_max = self._sigma_max(sigmas)
        num_steps = int(sigmas.shape[0]) - 1

        with self._replay_context(), self._autocast_context(device):
            batched = self._replay_batched(
                conditions,
                segment=segment,
                params=params,
                target=target,
                sigmas=sigmas,
                sigma_max=sigma_max,
                device=device,
            )
            if batched is not None:
                return batched

            log_probs: List[torch.Tensor] = []
            prev_sample_means: List[torch.Tensor] = []
            for step_index in target:
                sigma = sigmas[step_index].to(dtype=torch.float32)
                sigma_next = sigmas[step_index + 1].to(dtype=torch.float32)
                sample = segment.latents_at(step_index).to(device)
                prev_sample = segment.latents_at(step_index + 1).to(device)
                step_kwargs = self._step_kwargs(
                    conditions,
                    params,
                    sample=sample,
                    step_index=step_index,
                    num_steps=num_steps,
                    mode="replay",
                    state=None,
                )
                _, log_prob, prev_mean = self.step.step_with_logp(
                    self.model,
                    conditions,
                    strategy=self.strategy,
                    sample=sample,
                    prev_sample=prev_sample,
                    sigma=sigma,
                    sigma_next=sigma_next,
                    guidance_scale=self._guidance_scale(
                        params,
                        step_index=step_index,
                        num_steps=num_steps,
                        mode="replay",
                    ),
                    eta=float(params.eta),
                    sigma_max=sigma_max,
                    step_index=step_index,
                    **step_kwargs,
                )
                if log_prob is None:
                    raise RuntimeError(
                        f"{self._runner_name}.replay: strategy returned None log-prob "
                        f"at step_index={step_index} (deterministic mode); replay "
                        "requires a stochastic SDE strategy."
                    )
                log_probs.append(log_prob)
                if prev_mean is not None:
                    prev_sample_means.append(prev_mean)

        log_probs_t = torch.stack(log_probs, dim=1).to(dtype=self.logprob_dtype)
        means_t = torch.stack(prev_sample_means, dim=1).to(dtype=self.trajectory_dtype) if prev_sample_means else None

        from unirl.models.types.replay_result import ReplayResult

        return ReplayResult(log_probs=log_probs_t, prev_sample_means=means_t)

    def _predict_noise_kwargs(
        self,
        conditions: C,
        params: DiffusionSamplingParams,
        *,
        sample: torch.Tensor,
    ) -> Mapping[str, Any]:
        return self._step_kwargs(
            conditions,
            params,
            sample=sample,
            step_index=0,
            num_steps=int(params.num_inference_steps),
            mode="predict",
            state=None,
        )

    def predict_noise_at_step(
        self,
        conditions: C,
        *,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        params: DiffusionSamplingParams,
    ) -> torch.Tensor:
        """Run the model kernel once without traversing a schedule."""
        conditions = self._prepare_conditions(conditions, params, mode="predict")
        return self.step.predict_noise(
            self.model,
            sample,
            sigma,
            conditions,
            guidance_scale=self._guidance_scale(
                params,
                step_index=0,
                num_steps=int(params.num_inference_steps),
                mode="predict",
            ),
            **self._predict_noise_kwargs(conditions, params, sample=sample),
        )

    def trainable_module(self) -> torch.nn.Module:
        """Default FSDP/LoRA surface for single-transformer bundles."""
        return self.model.transformer


class SingleStreamVideoDiffusionRunner(SingleStreamDiffusionRunner[B, C], ABC):
    """Single-stream specialization for dense 5D video latents."""

    SEGMENT_FACTORY = make_video_segment
    SIGMA_MAX_AS_FLOAT = True

    def _validate_replay_segment(self, segment: LatentSegment) -> None:
        super()._validate_replay_segment(segment)
        if segment.latents.ndim != 6:
            raise ValueError(
                f"{self._runner_name}.replay: expected video latents "
                f"[B, K, C, T, H, W], got {tuple(segment.latents.shape)}."
            )


__all__ = [
    "SingleStreamDiffusionRunner",
    "SingleStreamDiffusionStep",
    "SingleStreamLatentSpec",
    "SingleStreamVideoDiffusionRunner",
    "temporary_eval",
]
