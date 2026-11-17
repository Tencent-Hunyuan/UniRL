"""Cosmos3 text-to-image and image-to-video diffusion: per-step packed forward with CFG and the rollout stage."""

from __future__ import annotations

from typing import Any, List, Optional, Set, Tuple

import torch

from unirl.config.require import require
from unirl.models.types.diffusion import DiffusionStage, DiffusionStep
from unirl.models.types.replay_result import ReplayResult
from unirl.sde.kernels import StepStrategy
from unirl.types.conditions import ImageLatentCondition, TextTokenCondition
from unirl.types.sampling import DiffusionSamplingParams, compute_trajectory_positions
from unirl.types.segments.latent import LatentSegment
from unirl.utils.dtypes import parse_torch_dtype

from .bundle import Cosmos3Bundle
from .conditions import Cosmos3Conditions
from .packing import pack_joint_sequence


class Cosmos3DiffusionStep(DiffusionStep[Cosmos3Bundle, Cosmos3Conditions]):
    """Per-step Cosmos3 kernel: one packed forward per CFG branch (README # Gotchas)."""

    def __init__(self, pipe: Any) -> None:
        self.pipe = pipe

    def _velocity(
        self,
        model: Cosmos3Bundle,
        text: TextTokenCondition,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        image_latent: Optional[ImageLatentCondition] = None,
    ) -> torch.Tensor:
        """One packed forward on the generated frames ``sample``; ``timestep`` is a 0-d fp32 ``sigma * 1000``."""
        tokens, clean_frames = sample, 0
        if image_latent is not None:  # frame 0 is never noised
            tokens, clean_frames = torch.cat([image_latent.latents.to(sample.device), sample], dim=2), 1
        kwargs, meta = pack_joint_sequence(
            self.pipe,
            input_ids=text.input_ids[0][text.attention_mask[0].bool()].tolist(),
            vision_tokens=tokens,
            condition_frame_indexes=range(clean_frames),
            vision_fps=model.config.fps,
            device=sample.device,
            compute_dtype=model.dtype,
        )
        kwargs["vision_timesteps"] = timestep.expand(meta["num_noisy_vision_tokens"])
        preds_vision = model.transformer(**kwargs)[0]  # [0] is vision on the 0.39 tuple and the 0.40 output object
        return preds_vision[0].reshape(tokens.shape)[:, :, clean_frames:]

    def predict_noise(
        self,
        model: Cosmos3Bundle,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        conditions: Cosmos3Conditions,
        *,
        guidance_scale: float,
    ) -> torch.Tensor:
        """Guided velocity ``[1, C, T_gen, H, W]`` for one sample's generated frames at fp32 scalar ``sigma``."""
        require(conditions.text is not None, "Cosmos3DiffusionStep.predict_noise: conditions.text is None")
        require(
            sample.ndim == 5 and sample.shape[0] == 1,
            f"Cosmos3DiffusionStep.predict_noise: expected sample [1, C, T_gen, H, W], got {tuple(sample.shape)}",
        )
        timestep = sigma.to(device=sample.device, dtype=torch.float32).reshape(())
        timestep = timestep * self.pipe.scheduler.config.num_train_timesteps
        cond = self._velocity(model, conditions.text, sample, timestep, conditions.image_latent)
        if guidance_scale == 1.0:
            return cond
        require(
            conditions.negative_text is not None,
            "Cosmos3DiffusionStep.predict_noise: guidance_scale != 1 requires conditions.negative_text",
        )
        uncond = self._velocity(model, conditions.negative_text, sample, timestep, conditions.image_latent)
        return uncond + guidance_scale * (cond - uncond)

    def forward(
        self,
        *,
        strategy: StepStrategy,
        noise_pred: torch.Tensor,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        prev_sample: Optional[torch.Tensor] = None,
        sigma_max: float = 0.99,
        eta: float = 1.0,
        step_index: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run one transition given a precomputed ``noise_pred``."""
        return strategy.denoise(
            noise_pred=noise_pred,
            sample=sample,
            sigma=sigma,
            sigma_next=sigma_next,
            eta=eta,
            prev_sample=prev_sample,
            sigma_max=sigma_max,
            step_index=step_index,
        )

    def step(
        self,
        model: Cosmos3Bundle,
        conditions: Cosmos3Conditions,
        *,
        strategy: StepStrategy,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        guidance_scale: float,
        prev_sample: Optional[torch.Tensor] = None,
        sigma_max: float = 0.99,
        eta: float = 1.0,
        step_index: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Model forward plus transition for one sample."""
        noise_pred = self.predict_noise(model, sample, sigma, conditions, guidance_scale=guidance_scale)
        return self.forward(
            strategy=strategy,
            noise_pred=noise_pred,
            sample=sample,
            sigma=sigma,
            sigma_next=sigma_next,
            prev_sample=prev_sample,
            sigma_max=sigma_max,
            eta=eta,
            step_index=step_index,
        )

    step_with_logp = step


class Cosmos3DiffusionStage(DiffusionStage[Cosmos3Conditions]):
    """Cosmos3 rollout stage over the generated latent frames; samples run one at a time (README # Generation)."""

    def __init__(
        self,
        *,
        model: Cosmos3Bundle,
        step: Cosmos3DiffusionStep,
        strategy: StepStrategy,
        trajectory_precision: str,
        logprob_precision: str,
    ) -> None:
        self.model = model
        self.step = step
        self.strategy = strategy
        self.trajectory_dtype = parse_torch_dtype(trajectory_precision, field_name="trajectory_precision")
        self.logprob_dtype = parse_torch_dtype(logprob_precision, field_name="logprob_precision")

    def diffuse(
        self,
        conditions: Cosmos3Conditions,
        *,
        schedule: torch.Tensor,
        params: DiffusionSamplingParams,
        initial_latents: Optional[torch.Tensor] = None,
    ) -> LatentSegment:
        """Sample ``B`` generated-frame latents; ``LatentSegment.latents`` is ``[B, K, C, T_gen, H_lat, W_lat]``."""
        from unirl.sde.noise import generate_latents

        vae_config = self.model.vae.config
        generated_frames, remainder = divmod(params.num_frames - 1, vae_config.scale_factor_temporal)
        require(
            params.num_frames >= 1 and remainder == 0,
            f"Cosmos3DiffusionStage.diffuse: num_frames must be 1 + a multiple of "
            f"{vae_config.scale_factor_temporal}, got {params.num_frames}",
        )
        require(
            (params.num_frames == 1) == (conditions.image_latent is None),
            f"Cosmos3DiffusionStage.diffuse: num_frames={params.num_frames} with image_latent "
            f"{'absent' if conditions.image_latent is None else 'present'}; video needs a source image and a "
            "single frame takes none (README # Generation)",
        )
        time_embedder_dtype = next(self.model.transformer.time_embedder.parameters()).dtype
        require(
            time_embedder_dtype == torch.float32,
            f"Cosmos3DiffusionStage.diffuse: time_embedder is stored as {time_embedder_dtype}, expected fp32 "
            "(README # Gotchas)",
        )
        device = torch.device(self.model.device)
        batch_size = conditions.batch_size
        total_steps = params.num_inference_steps
        require(
            schedule.shape[0] == total_steps + 1,
            f"Cosmos3DiffusionStage.diffuse: schedule length {schedule.shape[0]} != T+1={total_steps + 1}",
        )
        schedule = schedule.to(device)
        scale = vae_config.scale_factor_spatial
        expected = (vae_config.z_dim, max(generated_frames, 1), params.height // scale, params.width // scale)
        if initial_latents is None:
            initial_latents = generate_latents(
                batch_size=batch_size,
                latent_shape=expected,
                device=device,
                dtype=self.trajectory_dtype,
                init_same_noise=params.init_same_noise,
                samples_per_prompt=params.samples_per_prompt,
                noise_group_ids=params.noise_group_ids,
                base_seed=params.seed,
            )
        require(
            tuple(initial_latents.shape) == (batch_size, *expected),
            f"Cosmos3DiffusionStage.diffuse: initial_latents {tuple(initial_latents.shape)} != {(batch_size, *expected)}",
        )

        sde_set: Set[int] = set(params.sde_indices or ())
        require(
            not sde_set or params.eta > 0.0,
            "Cosmos3DiffusionStage.diffuse: sde_indices are non-empty but eta=0, so no log-probs would be emitted",
        )
        needed = set(compute_trajectory_positions(sde_set, total_steps))
        needed.add(total_steps)
        sigma_max = schedule[1].float()

        segments: List[LatentSegment] = []
        for sample_idx in range(batch_size):
            single = conditions.slice(sample_idx, sample_idx + 1)
            self.strategy.init_schedule(schedule)
            state = initial_latents[sample_idx : sample_idx + 1].to(device=device, dtype=self.trajectory_dtype)
            stored: List[Tuple[int, torch.Tensor]] = [(0, state)] if 0 in needed else []
            log_probs: List[torch.Tensor] = []
            for index in range(total_steps):
                with torch.no_grad():
                    state, log_prob, _ = self.step.step_with_logp(
                        self.model,
                        single,
                        strategy=self.strategy,
                        sample=state,
                        sigma=schedule[index],
                        sigma_next=schedule[index + 1],
                        guidance_scale=params.guidance_scale,
                        sigma_max=sigma_max,
                        eta=params.eta if index in sde_set else 0.0,
                        step_index=index,
                    )
                state = state.to(dtype=self.trajectory_dtype)
                if index + 1 in needed:
                    stored.append((index + 1, state))
                if log_prob is not None:
                    log_probs.append(log_prob.to(dtype=self.logprob_dtype))
            segments.append(
                LatentSegment(
                    latents=torch.stack([value for _, value in stored], dim=1),
                    sigmas=schedule,
                    indices=torch.tensor([index for index, _ in stored], dtype=torch.long, device=device),
                    sde_logp=torch.stack(log_probs, dim=1) if log_probs else None,
                    sde_indices=torch.tensor(sorted(sde_set), dtype=torch.long, device=device) if sde_set else None,
                )
            )
        return segments[0] if len(segments) == 1 else LatentSegment.concat(segments)

    def replay(
        self,
        conditions: Cosmos3Conditions,
        *,
        segment: LatentSegment,
        params: DiffusionSamplingParams,
        step_indices: Optional[List[int]] = None,
    ) -> ReplayResult:
        """Recompute SDE log-probs ``[B, S]`` and means ``[B, S, C, T_gen, H_lat, W_lat]`` with gradients."""
        require(
            all(x is not None for x in (segment.sde_indices, segment.indices, segment.latents, segment.sigmas)),
            "Cosmos3DiffusionStage.replay requires segment sde_indices, indices, latents and sigmas",
        )
        sde_steps = segment.sde_indices.tolist()
        target = step_indices if step_indices is not None else sde_steps
        require(
            set(target) <= set(sde_steps),
            f"Cosmos3DiffusionStage.replay: steps {target} are not all SDE steps {sde_steps}",
        )
        device = torch.device(self.model.device)
        schedule = segment.sigmas.to(device)
        sigma_max = schedule[1].float()
        slots = {step: slot for slot, step in enumerate(segment.indices.tolist())}
        log_probs: List[torch.Tensor] = []
        means: List[torch.Tensor] = []
        for sample_idx in range(conditions.batch_size):
            single = conditions.slice(sample_idx, sample_idx + 1)
            for step_index in target:
                _, log_prob, mean = self.step.step_with_logp(
                    self.model,
                    single,
                    strategy=self.strategy,
                    sample=segment.latents[sample_idx : sample_idx + 1, slots[step_index]].to(device),
                    prev_sample=segment.latents[sample_idx : sample_idx + 1, slots[step_index + 1]].to(device),
                    sigma=schedule[step_index],
                    sigma_next=schedule[step_index + 1],
                    guidance_scale=params.guidance_scale,
                    sigma_max=sigma_max,
                    eta=params.eta,
                    step_index=step_index,
                )
                log_probs.append(log_prob)
                means.append(mean)
        shape = (conditions.batch_size, len(target))
        return ReplayResult(
            log_probs=torch.cat(log_probs).reshape(shape).to(dtype=self.logprob_dtype),
            prev_sample_means=torch.cat(means).reshape(*shape, *means[0].shape[1:]),
        )

    def predict_noise_at_step(
        self,
        conditions: Cosmos3Conditions,
        *,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        params: DiffusionSamplingParams,
    ) -> torch.Tensor:
        """Guided velocity for ``sample`` ``[B, C, T_gen, H_lat, W_lat]`` at scalar or per-sample ``sigma``."""
        sigma = sigma.reshape(-1).expand(sample.shape[0])
        return torch.cat(
            [
                self.step.predict_noise(
                    self.model,
                    sample[i : i + 1].to(self.model.device),
                    sigma[i],
                    conditions.slice(i, i + 1),
                    guidance_scale=params.guidance_scale,
                )
                for i in range(sample.shape[0])
            ]
        )

    def trainable_module(self) -> torch.nn.Module:
        """Return the transformer the diffusion forward runs on."""
        return self.model.transformer


__all__ = ["Cosmos3DiffusionStage", "Cosmos3DiffusionStep"]
