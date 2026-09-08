"""SenseNova-U1.5 packed-pixel flow sampling and replay."""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import torch

from unirl.config.require import require
from unirl.models.types.diffusion import DiffusionStage
from unirl.models.types.replay_result import ReplayResult
from unirl.sde.kernels import FlowSDEStrategy, StepStrategy
from unirl.types.sampling import DiffusionSamplingParams, compute_trajectory_positions
from unirl.types.segments.latent import LatentSegment
from unirl.utils.dtypes import parse_torch_dtype

from .bundle import SenseNovaU1Bundle
from .conditions import SenseNovaU1Conditions
from .pixels import packed_pixel_shape, patchify_pixels, unpatchify_pixels

CFG_NORM_TYPES = ("none", "global", "channel", "cfg_zero_star")


@dataclass
class SenseNovaU1DiffusionParams(DiffusionSamplingParams):
    """Sampling knobs specific to the U1.5 pixel-flow head."""

    num_inference_steps: int = 50
    guidance_scale: float = 4.0
    height: int = 512
    width: int = 512
    eta: float = 1.0
    cfg_norm: str = "none"
    cfg_interval: Tuple[float, float] = (0.0, 1.0)
    t_eps: float = 0.02
    trajectory_precision: str = "bf16"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not isinstance(self.cfg_interval, tuple):
            self.cfg_interval = tuple(self.cfg_interval)
        require(
            len(self.cfg_interval) == 2 and 0.0 <= float(self.cfg_interval[0]) <= float(self.cfg_interval[1]) <= 1.0,
            f"SenseNovaU1DiffusionParams.cfg_interval must lie within [0, 1]; got {self.cfg_interval!r}.",
        )
        require(
            self.cfg_norm in CFG_NORM_TYPES,
            f"SenseNovaU1DiffusionParams.cfg_norm must be one of {CFG_NORM_TYPES}; got {self.cfg_norm!r}.",
        )
        require(float(self.t_eps) > 0.0, f"SenseNovaU1DiffusionParams.t_eps must be positive; got {self.t_eps}.")


def resolve_noise_scale(model: torch.nn.Module, image_shape: Tuple[int, int]) -> float:
    """Match the checkpoint's resolution-dependent initial pixel-noise scale."""
    height, width = (int(v) for v in image_shape)
    patch = int(model.patch_size)
    merge = int(1 / float(model.downsample_ratio))
    grid_h, grid_w = height // patch, width // patch
    scale = float(model.noise_scale)
    if model.noise_scale_mode in {"resolution", "dynamic", "dynamic_sqrt"}:
        image_sequence = (grid_h * grid_w) / (merge**2)
        scale *= math.sqrt(image_sequence / float(model.noise_scale_base_image_seq_len))
        if model.noise_scale_mode == "dynamic_sqrt":
            scale = math.sqrt(scale)
    return min(scale, float(model.noise_scale_max_value))


class SenseNovaU1DiffusionStep:
    """One model prediction plus a framework-owned FlowGRPO transition."""

    @staticmethod
    def _optimized_scale(positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        positive = positive.flatten(1).float()
        negative = negative.flatten(1).float()
        numerator = torch.sum(positive * negative, dim=1, keepdim=True)
        denominator = torch.sum(negative.square(), dim=1, keepdim=True) + 1e-8
        return numerator / denominator

    def _apply_cfg(
        self,
        condition_velocity: torch.Tensor,
        uncondition_velocity: torch.Tensor,
        *,
        guidance: float,
        cfg_norm: str,
        step_index: int,
    ) -> torch.Tensor:
        """Combine upstream conditional and unconditional velocity predictions."""
        if cfg_norm == "cfg_zero_star":
            if int(step_index) == 0:
                return torch.zeros_like(condition_velocity)
            alpha = self._optimized_scale(condition_velocity, uncondition_velocity).to(condition_velocity.dtype)
            alpha = alpha.reshape(-1, 1, 1)
            return uncondition_velocity * alpha + guidance * (condition_velocity - uncondition_velocity * alpha)

        velocity = uncondition_velocity + guidance * (condition_velocity - uncondition_velocity)
        if cfg_norm in {"global", "channel"}:
            device_type = condition_velocity.device.type
            if device_type == "mps":
                device_type = "cpu"
            norm_dims = (1, 2) if cfg_norm == "global" else -1
            with torch.autocast(device_type=device_type, enabled=False):
                condition_norm = torch.norm(condition_velocity, dim=norm_dims, keepdim=True)
                guided_norm = torch.norm(velocity, dim=norm_dims, keepdim=True)
                scale = (condition_norm / (guided_norm + 1e-8)).clamp(0.0, 1.0)
            velocity = velocity * scale.to(velocity.dtype)
        return velocity

    def predict_velocity(
        self,
        bundle: SenseNovaU1Bundle,
        conditions: SenseNovaU1Conditions,
        *,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        params: SenseNovaU1DiffusionParams,
        step_index: int,
    ) -> torch.Tensor:
        """Predict the upstream data-time velocity ``dx/dt`` for one sample."""
        require(
            conditions.batch_size == 1,
            f"SenseNovaU1DiffusionStep expects one prompt/cache at a time; got {conditions.batch_size}.",
        )
        _, condition_cache, uncondition_cache, condition_indexes, uncondition_indexes, image_shape = conditions.single()
        model = bundle.model
        device = torch.device(bundle.device)
        sample = sample.to(device)

        pixel_patch = int(model.patch_size) * int(1 / float(model.downsample_ratio))
        expected = packed_pixel_shape(image_shape, patch_size=pixel_patch)
        require(
            sample.ndim == 3 and tuple(sample.shape[1:]) == expected,
            f"SenseNovaU1DiffusionStep expected packed pixels [B, {expected[0]}, {expected[1]}], "
            f"got {tuple(sample.shape)}.",
        )

        normalized_pixels = unpatchify_pixels(sample, image_shape=image_shape, patch_size=pixel_patch)
        sigma = sigma.to(device=device, dtype=torch.float32)
        data_time = 1.0 - sigma
        noise_scale = resolve_noise_scale(model, image_shape)
        lo, hi = (float(v) for v in params.cfg_interval)
        use_cfg = (
            float(params.guidance_scale) > 1.0 and uncondition_cache is not None and lo <= float(data_time.item()) <= hi
        )
        prediction = bundle.transformer(
            "predict_velocity",
            normalized_pixels=normalized_pixels,
            packed_pixels=sample,
            image_indexes=condition_indexes,
            prefix_cache=condition_cache,
            data_time=data_time,
            image_shape=image_shape,
            noise_scale=noise_scale,
            uncondition_image_indexes=uncondition_indexes if use_cfg else None,
            uncondition_prefix_cache=uncondition_cache if use_cfg else None,
        )
        if not use_cfg:
            return prediction
        condition_velocity, uncondition_velocity = prediction
        return self._apply_cfg(
            condition_velocity,
            uncondition_velocity,
            guidance=float(params.guidance_scale),
            cfg_norm=params.cfg_norm,
            step_index=step_index,
        )

    def step_with_logp(
        self,
        bundle: SenseNovaU1Bundle,
        conditions: SenseNovaU1Conditions,
        *,
        strategy: StepStrategy,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        params: SenseNovaU1DiffusionParams,
        prev_sample: Optional[torch.Tensor] = None,
        sigma_max: float | torch.Tensor = 0.99,
        eta: float = 1.0,
        step_index: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Map upstream ``dx/dt`` to framework ``d x / d sigma = -dx/dt``."""
        velocity = self.predict_velocity(
            bundle,
            conditions,
            sample=sample,
            sigma=sigma,
            params=params,
            step_index=step_index,
        )
        if float(eta) < 1e-7 and prev_sample is None and isinstance(strategy, FlowSDEStrategy):
            # Match upstream t2i_generate exactly on deterministic steps: its
            # Euler update runs in the trajectory dtype before the next model
            # call. The generic FlowSDEStrategy promotes state and velocity to
            # fp32 even when eta=0, which accumulates visible drift over 50
            # BF16 inference steps.
            data_time = 1.0 - sigma.to(device=sample.device)
            next_data_time = 1.0 - sigma_next.to(device=sample.device)
            next_sample = sample + (next_data_time - data_time) * velocity
            return next_sample, None, None
        noise_scale = resolve_noise_scale(bundle.model, tuple(conditions.image_shapes[0]))
        unit_sample = sample / noise_scale
        unit_velocity = velocity / noise_scale
        unit_prev_sample = None if prev_sample is None else prev_sample / noise_scale
        next_sample, log_prob, prev_mean = strategy.denoise(
            noise_pred=-unit_velocity,
            sample=unit_sample,
            sigma=sigma,
            sigma_next=sigma_next,
            eta=float(eta),
            prev_sample=unit_prev_sample,
            sigma_max=float(sigma_max),
            step_index=int(step_index),
        )
        return (
            next_sample * noise_scale,
            log_prob,
            None if prev_mean is None else prev_mean * noise_scale,
        )


class SenseNovaU1DiffusionStage(DiffusionStage[SenseNovaU1Conditions]):
    """Rollout and grad-capable replay over packed normalized RGB pixels."""

    def __init__(
        self,
        *,
        model: SenseNovaU1Bundle,
        step: Optional[SenseNovaU1DiffusionStep] = None,
        strategy: Optional[StepStrategy] = None,
        autocast_precision: str = "bf16",
        trajectory_precision: str = "bf16",
        logprob_precision: str = "fp32",
    ) -> None:
        self.model = model
        self.step = step if step is not None else SenseNovaU1DiffusionStep()
        self.strategy = strategy if strategy is not None else FlowSDEStrategy()
        self.autocast_dtype = parse_torch_dtype(autocast_precision, field_name="autocast_precision")
        self.trajectory_dtype = parse_torch_dtype(trajectory_precision, field_name="trajectory_precision")
        self.logprob_dtype = parse_torch_dtype(logprob_precision, field_name="logprob_precision")

    def _autocast_ctx(self):
        device = torch.device(self.model.device)
        if device.type == "cuda" and self.autocast_dtype in (torch.float16, torch.bfloat16):
            return torch.autocast("cuda", self.autocast_dtype)
        if device.type == "cpu" and self.autocast_dtype == torch.bfloat16:
            return torch.autocast("cpu", torch.bfloat16)
        return nullcontext()

    def _configure_t_eps(self, params: SenseNovaU1DiffusionParams) -> None:
        """Apply the request-level inference clamp once before model forwards."""
        self.model.model.config.t_eps = float(params.t_eps)

    @staticmethod
    def _single_conditions(conditions: SenseNovaU1Conditions, index: int) -> SenseNovaU1Conditions:
        return conditions.slice(index, index + 1)

    def _diffuse_one(
        self,
        conditions: SenseNovaU1Conditions,
        *,
        schedule: torch.Tensor,
        params: SenseNovaU1DiffusionParams,
        initial_latents: torch.Tensor,
    ) -> LatentSegment:
        device = torch.device(self.model.device)
        schedule = schedule.to(device)
        total_steps = int(params.num_inference_steps)
        require(
            int(schedule.shape[0]) == total_steps + 1,
            f"SenseNovaU1DiffusionStage: schedule length {schedule.shape[0]} != {total_steps + 1}.",
        )
        self.strategy.init_schedule(schedule)
        sigma_max = schedule[1].float() if total_steps else schedule[0].float()

        image_shape = tuple(conditions.image_shapes[0])
        noise_scale = resolve_noise_scale(self.model.model, image_shape)
        state = initial_latents.to(device=device, dtype=self.trajectory_dtype) * noise_scale

        sde_set: Set[int] = set(int(i) for i in (params.sde_indices or []))
        require(
            not sde_set or float(params.eta) > 0.0,
            "SenseNovaU1DiffusionStage: sde_indices are non-empty but eta=0, "
            "so rollout would emit no transition log-probabilities.",
        )
        needed = set(compute_trajectory_positions(sde_set, total_steps))
        needed.add(total_steps)
        stored: List[Tuple[int, torch.Tensor]] = []
        if 0 in needed:
            stored.append((0, state.detach().clone()))
        log_probs: List[torch.Tensor] = []
        means: List[torch.Tensor] = []

        for index in range(total_steps):
            eta = float(params.eta) if index in sde_set else 0.0
            with torch.no_grad(), self._autocast_ctx():
                state, log_prob, mean = self.step.step_with_logp(
                    self.model,
                    conditions,
                    strategy=self.strategy,
                    sample=state,
                    sigma=schedule[index],
                    sigma_next=schedule[index + 1],
                    params=params,
                    sigma_max=sigma_max,
                    eta=eta,
                    step_index=index,
                )
            state = state.to(dtype=self.trajectory_dtype)
            if index + 1 in needed:
                stored.append((index + 1, state.detach().clone()))
            if log_prob is not None:
                log_probs.append(log_prob.to(dtype=self.logprob_dtype))
                if mean is not None:
                    means.append(mean.detach().to(dtype=self.trajectory_dtype))

        return LatentSegment(
            latents=torch.stack([value for _, value in stored], dim=1),
            sigmas=schedule,
            indices=torch.tensor([index for index, _ in stored], dtype=torch.long, device=device),
            sde_logp=torch.stack(log_probs, dim=1) if log_probs else None,
            sde_means=torch.stack(means, dim=1) if means else None,
            sde_indices=(torch.tensor(sorted(sde_set), dtype=torch.long, device=device) if sde_set else None),
        )

    def diffuse(
        self,
        conditions: SenseNovaU1Conditions,
        *,
        schedule: torch.Tensor,
        params: SenseNovaU1DiffusionParams,
        initial_latents: Optional[torch.Tensor] = None,
    ) -> LatentSegment:
        """Sample each prompt independently to keep prefix caches batch-local."""
        self._configure_t_eps(params)
        conditions.validate()
        device = torch.device(self.model.device)
        pixel_patch = int(self.model.model.patch_size) * int(1 / float(self.model.model.downsample_ratio))
        shapes = [packed_pixel_shape(shape, patch_size=pixel_patch) for shape in conditions.image_shapes]
        require(
            len(set(shapes)) == 1,
            f"SenseNovaU1DiffusionStage requires one output shape per batch; got {conditions.image_shapes}.",
        )
        expected = shapes[0]
        batch_size = conditions.batch_size

        if initial_latents is None:
            from unirl.sde.noise import generate_latents

            initial_latents = generate_latents(
                batch_size=batch_size,
                latent_shape=(3, int(params.height), int(params.width)),
                device=device,
                dtype=self.trajectory_dtype,
                init_same_noise=bool(params.init_same_noise),
                samples_per_prompt=int(params.samples_per_prompt),
                noise_group_ids=params.noise_group_ids,
                base_seed=int(params.seed or 0),
            )
        spatial_shape = (batch_size, 3, int(params.height), int(params.width))
        packed_shape = (batch_size, *expected)
        if tuple(initial_latents.shape) == spatial_shape:
            initial_latents = patchify_pixels(initial_latents, patch_size=pixel_patch)
        elif tuple(initial_latents.shape) != packed_shape:
            raise ValueError(
                f"SenseNovaU1DiffusionStage initial_latents shape {tuple(initial_latents.shape)} "
                f"must be spatial {spatial_shape} or packed {packed_shape}."
            )

        segments = [
            self._diffuse_one(
                self._single_conditions(conditions, index),
                schedule=schedule,
                params=params,
                initial_latents=initial_latents[index : index + 1],
            )
            for index in range(batch_size)
        ]
        if len(segments) == 1:
            return segments[0]
        return LatentSegment.concat(segments)

    def replay(
        self,
        conditions: SenseNovaU1Conditions,
        *,
        segment: LatentSegment,
        params: SenseNovaU1DiffusionParams,
        step_indices: Optional[List[int]] = None,
    ) -> ReplayResult:
        """Recompute SDE transition likelihoods with gradients."""
        self._configure_t_eps(params)
        if segment.sde_indices is None or segment.latents is None or segment.sigmas is None:
            raise ValueError("SenseNovaU1DiffusionStage.replay requires segment SDE indices, latents, and sigmas.")
        conditions.validate()
        require(
            int(segment.latents.shape[0]) == conditions.batch_size,
            f"SenseNovaU1DiffusionStage.replay batch mismatch: latents={segment.latents.shape[0]}, "
            f"conditions={conditions.batch_size}.",
        )
        sde_set = set(int(i) for i in segment.sde_indices.tolist())
        target = [int(i) for i in step_indices] if step_indices is not None else sorted(sde_set)
        invalid = [i for i in target if i not in sde_set]
        require(not invalid, f"SenseNovaU1DiffusionStage.replay requested non-SDE steps {invalid}.")

        device = torch.device(self.model.device)
        schedule = segment.sigmas.to(device)
        sigma_max = schedule[1].float()
        batch_log_probs: List[torch.Tensor] = []
        batch_means: List[torch.Tensor] = []
        for batch_index in range(conditions.batch_size):
            single = self._single_conditions(conditions, batch_index)
            log_probs: List[torch.Tensor] = []
            means: List[torch.Tensor] = []
            for step_index in target:
                state = segment.latents_at(step_index)[batch_index : batch_index + 1].to(device)
                previous = segment.latents_at(step_index + 1)[batch_index : batch_index + 1].to(device)
                with self._autocast_ctx():
                    _, log_prob, mean = self.step.step_with_logp(
                        self.model,
                        single,
                        strategy=self.strategy,
                        sample=state,
                        sigma=schedule[step_index],
                        sigma_next=schedule[step_index + 1],
                        params=params,
                        prev_sample=previous,
                        sigma_max=sigma_max,
                        eta=float(params.eta),
                        step_index=step_index,
                    )
                if log_prob is None or mean is None:
                    raise RuntimeError(
                        f"SenseNovaU1DiffusionStage.replay got a deterministic transition at step {step_index}."
                    )
                log_probs.append(log_prob)
                means.append(mean)
            batch_log_probs.append(torch.stack(log_probs, dim=1))
            batch_means.append(torch.stack(means, dim=1))

        return ReplayResult(
            log_probs=torch.cat(batch_log_probs, dim=0).to(dtype=self.logprob_dtype),
            prev_sample_means=torch.cat(batch_means, dim=0).to(dtype=self.trajectory_dtype),
        )

    def predict_noise_at_step(
        self,
        conditions: SenseNovaU1Conditions,
        *,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        params: SenseNovaU1DiffusionParams,
    ) -> torch.Tensor:
        """Return the framework sigma-time velocity ``dx/dsigma``."""
        self._configure_t_eps(params)
        with self._autocast_ctx():
            return -self.step.predict_velocity(
                self.model,
                conditions,
                sample=sample,
                sigma=sigma,
                params=params,
                # This API has no schedule index. Treat it as a non-initial
                # prediction so CFG-Zero* does not incorrectly zero every call.
                step_index=-1,
            )

    def trainable_module(self) -> torch.nn.Module:
        return self.model.trainable_module()


__all__ = [
    "SenseNovaU1DiffusionParams",
    "SenseNovaU1DiffusionStage",
    "SenseNovaU1DiffusionStep",
    "resolve_noise_scale",
]
