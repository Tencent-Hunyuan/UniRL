"""Alternating generator/fake-score stack for WAN Self-Forcing DMD."""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type

import torch

from unirl.algorithms.base import typed_conditions
from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.distributed.group.remote import Remote
from unirl.distributed.tensor.batch import _move_value
from unirl.types.sample import Part
from unirl.utils.misc import aggregate_numeric_metrics


@dataclass(frozen=True)
class SelfForcingStepResult:
    """Metrics from one generator update followed by N fake-score updates."""

    generator_loss: float
    generator_grad_norm: float
    generator_lr: float
    fake_score_loss: float
    fake_score_grad_norm: float
    fake_score_lr: float
    fake_score_updates: int
    metrics: Mapping[str, object]


class SelfForcingDMDStack(Remote):
    """Own the two-optimizer update schedule while reusing two FSDP backends."""

    def __init__(
        self,
        *,
        generator_pipeline: Any,
        generator_backend: Any,
        fake_score_pipeline: Any,
        fake_score_backend: Any,
        real_score_pipeline: Any,
        real_score_backend: Any,
        params: Any,
        conditions_cls: Optional[Type[Any]] = None,
        micro_batch_size: int = 1,
        generator_max_grad_norm: float = 1.0,
        fake_score_max_grad_norm: float = 1.0,
        fake_score_updates_per_generator: int = 5,
        denoising_sigmas: Sequence[float] = (1.0, 0.75, 0.5, 0.25),
        same_exit_step_across_blocks: bool = True,
        frames_per_block: int = 1,
        context_sigma: float = 0.0,
        generator_guidance_scale: float = 1.0,
        real_guidance_scale: float = 3.0,
        fake_guidance_scale: float = 0.0,
        score_sigma_min: float = 0.02,
        score_sigma_max: float = 0.98,
        score_timestep_shift: float = 5.0,
        normalization_eps: float = 1e-6,
        dmd_grad_clip: float = 1.0,
        latent_channels: int = 16,
        latent_frames: int = 5,
        latent_height: int = 32,
        latent_width: int = 56,
    ) -> None:
        super().__init__()
        if int(micro_batch_size) < 1:
            raise ValueError("SelfForcingDMDStack.micro_batch_size must be >= 1.")
        if int(fake_score_updates_per_generator) < 1:
            raise ValueError("fake_score_updates_per_generator must be >= 1.")
        if not 0.0 < float(score_sigma_min) < float(score_sigma_max) < 1.0:
            raise ValueError("score sigma bounds must satisfy 0 < min < max < 1.")
        if not float(score_timestep_shift) > 0.0:
            raise ValueError("score_timestep_shift must be > 0.")
        if not float(dmd_grad_clip) > 0.0:
            raise ValueError("dmd_grad_clip must be > 0.")
        if any(int(v) < 1 for v in (latent_channels, latent_frames, latent_height, latent_width)):
            raise ValueError("all latent shape dimensions must be >= 1.")

        self.generator_pipeline = generator_pipeline
        self.generator_backend = generator_backend
        self.fake_score_pipeline = fake_score_pipeline
        self.fake_score_backend = fake_score_backend
        self.real_score_pipeline = real_score_pipeline
        self.real_score_backend = real_score_backend
        self.params = params
        self.conditions_cls = conditions_cls
        self.micro_batch_size = int(micro_batch_size)
        self.generator_max_grad_norm = float(generator_max_grad_norm)
        self.fake_score_max_grad_norm = float(fake_score_max_grad_norm)
        self.fake_score_updates_per_generator = int(fake_score_updates_per_generator)
        self.same_exit_step_across_blocks = bool(same_exit_step_across_blocks)
        self.score_sigma_min = float(score_sigma_min)
        self.score_sigma_max = float(score_sigma_max)
        self.score_timestep_shift = float(score_timestep_shift)
        self.normalization_eps = float(normalization_eps)
        self.dmd_grad_clip = float(dmd_grad_clip)
        self.latent_shape = (
            int(latent_channels),
            int(latent_frames),
            int(latent_height),
            int(latent_width),
        )

        from unirl.models.wan21.self_forcing import WAN21SelfForcingStage

        self.rollout_stage = WAN21SelfForcingStage(
            diffusion=generator_pipeline.diffusion,
            frames_per_block=int(frames_per_block),
            denoising_sigmas=tuple(float(value) for value in denoising_sigmas),
            context_sigma=float(context_sigma),
            guidance_scale=float(generator_guidance_scale),
        )
        self.generator_score = generator_pipeline.diffusion
        self.fake_score = fake_score_pipeline.diffusion
        self.real_score = real_score_pipeline.diffusion
        self.real_guidance_scale = float(real_guidance_scale)
        self.fake_guidance_scale = float(fake_guidance_scale)

        if getattr(generator_pipeline.bundle.transformer, "_wan_block_causal_enabled", False) is not True:
            raise ValueError("SelfForcingDMDStack requires a block-causal generator transformer.")
        if getattr(fake_score_pipeline.bundle.transformer, "_wan_block_causal_enabled", False):
            raise ValueError("SelfForcingDMDStack fake score must be a bidirectional WAN transformer.")
        if getattr(real_score_pipeline.bundle.transformer, "_wan_block_causal_enabled", False):
            raise ValueError("SelfForcingDMDStack real score must be a bidirectional WAN transformer.")
        if any(parameter.requires_grad for parameter in real_score_backend.model.parameters()):
            raise ValueError("SelfForcingDMDStack real score must be fully frozen.")

    # ------------------------------------------------------------------
    # Tensor/metric helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _current_lr(backend: Any) -> float:
        groups = getattr(backend.optimizer, "param_groups", None)
        if isinstance(groups, list) and groups:
            return float(groups[0]["lr"])
        return 0.0

    @staticmethod
    def _mean_metrics(items: List[Mapping[str, object]]) -> Mapping[str, object]:
        return aggregate_numeric_metrics([dict(item) for item in items if item])

    @staticmethod
    def _noise(shape: Tuple[int, ...], *, device: torch.device) -> torch.Tensor:
        return torch.randn(shape, device=device, dtype=torch.float32)

    def _score_sigma(self, batch: int, *, device: torch.device) -> torch.Tensor:
        sigma = torch.rand(batch, device=device, dtype=torch.float32)
        shift = self.score_timestep_shift
        sigma = shift * sigma / (1.0 + (shift - 1.0) * sigma)
        return sigma.clamp(min=self.score_sigma_min, max=self.score_sigma_max)

    def _synced_exit_steps(self, num_samples: int, *, device: torch.device) -> Tuple[int, ...]:
        """Sample denoising exits and synchronize them across ranks.

        Rank-local choices would make FSDP ranks execute different numbers of
        forwards and deadlock, so rank 0 samples the vector and broadcasts it
        before the rollout.
        """
        if num_samples < 1:
            raise ValueError(f"num_samples must be >= 1; got {num_samples}.")
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
            exits = torch.randint(
                len(self.rollout_stage.denoising_sigmas),
                (num_samples,),
                device=device,
                dtype=torch.long,
            )
        else:
            exits = torch.empty(num_samples, device=device, dtype=torch.long)
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(exits, src=0)
        return tuple(int(value) for value in exits.tolist())

    def _rollout_exit_steps(self, *, total_frames: int, device: torch.device) -> Tuple[int, ...]:
        frames_per_block = int(self.rollout_stage.frames_per_block)
        if total_frames % frames_per_block:
            raise ValueError(f"latent frames {total_frames} must be divisible by frames_per_block={frames_per_block}.")
        num_blocks = total_frames // frames_per_block
        sampled = self._synced_exit_steps(
            1 if self.same_exit_step_across_blocks else num_blocks,
            device=device,
        )
        return sampled * num_blocks if self.same_exit_step_across_blocks else sampled

    def _typed_conditions(self, part: Part) -> Any:
        return typed_conditions(part.conditions, self.conditions_cls)

    def _micro_parts(self, part: Part) -> List[Part]:
        return [
            part.slice(start, min(start + self.micro_batch_size, part.batch_size))
            for start in range(0, part.batch_size, self.micro_batch_size)
        ]

    def _loss_scale(self, micro: Part, full: Part) -> float:
        return float(micro.batch_size) / float(full.batch_size)

    def _all_reduce_mean(self, backend: Any, value: float) -> float:
        total = backend.all_reduce_loss_sums([float(value)])[0]
        return total / float(backend.gradient_average_world_size())

    def _align_part(self, part: Part) -> None:
        device = next(self.generator_backend.trainable_module().parameters()).device
        part.conditions = {key: _move_value(value, device) for key, value in part.conditions.items()}

    def _predict_score(
        self,
        stage: Any,
        conditions: Any,
        *,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        guidance_scale: float,
    ) -> torch.Tensor:
        device = sample.device
        autocast_dtype = getattr(stage, "autocast_dtype", None)
        autocast = (
            torch.autocast("cuda", dtype=autocast_dtype)
            if device.type == "cuda" and autocast_dtype in (torch.float16, torch.bfloat16)
            else nullcontext()
        )
        with autocast:
            return stage.step.predict_noise(
                stage.model,
                sample,
                sigma,
                conditions,
                guidance_scale=float(guidance_scale),
            )

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------

    def _generator_loss(self, micro: Part) -> Tuple[torch.Tensor, Dict[str, float]]:
        conditions = self._typed_conditions(micro)
        device = conditions.text.embeds.device
        shape = (micro.batch_size, *self.latent_shape)
        exit_steps = self._rollout_exit_steps(total_frames=shape[2], device=device)
        rollout = self.rollout_stage.rollout(
            conditions,
            initial_noise=self._noise(shape, device=device),
            exit_steps=exit_steps,
        )
        generated = rollout.latents
        batch = generated.shape[0]
        with torch.no_grad():
            sigma = self._score_sigma(batch, device=device)
            s = sigma.view(batch, *([1] * (generated.ndim - 1)))
            score_noise = torch.randn_like(generated, dtype=torch.float32)
            xt = (1.0 - s) * generated.detach() + s * score_noise
            sigma_arg = sigma if batch > 1 else sigma.reshape(())
            fake_v = self._predict_score(
                self.fake_score,
                conditions,
                sample=xt,
                sigma=sigma_arg,
                guidance_scale=self.fake_guidance_scale,
            )
            real_v = self._predict_score(
                self.real_score,
                conditions,
                sample=xt,
                sigma=sigma_arg,
                guidance_scale=self.real_guidance_scale,
            )
            fake_x0 = xt - s * fake_v
            real_x0 = xt - s * real_v
            grad = fake_x0 - real_x0
            normalizer = (
                (generated.detach() - real_x0)
                .abs()
                .mean(
                    dim=tuple(range(1, generated.ndim)),
                    keepdim=True,
                )
            )
            grad = torch.nan_to_num(grad / normalizer.clamp_min(self.normalization_eps))
            grad = grad.clamp(min=-self.dmd_grad_clip, max=self.dmd_grad_clip)

        target = (generated - grad).detach()
        selected = (generated.float() - target.float()).pow(2)[rollout.gradient_mask]
        loss = 0.5 * selected.mean()
        value = float(loss.detach().item())
        if not math.isfinite(value):
            raise RuntimeError(f"SelfForcingDMDStack: non-finite generator loss {value!r}.")
        return loss, {
            "dmd_grad_abs_mean": float(grad.abs().mean().item()),
            "real_fake_x0_gap": float((fake_x0 - real_x0).abs().mean().item()),
            "score_sigma_mean": float(sigma.mean().item()),
            "rollout_exit_step": float(rollout.exit_step),
            "rollout_exit_step_min": float(min(rollout.exit_steps)),
            "rollout_exit_step_max": float(max(rollout.exit_steps)),
            "rollout_exit_step_mean": float(sum(rollout.exit_steps) / len(rollout.exit_steps)),
            "generated_x0_norm": float(generated.detach().float().pow(2).mean().item()),
        }

    def _fake_score_loss(self, micro: Part) -> Tuple[torch.Tensor, Dict[str, float]]:
        conditions = self._typed_conditions(micro)
        device = conditions.text.embeds.device
        shape = (micro.batch_size, *self.latent_shape)
        with torch.no_grad():
            exit_steps = self._rollout_exit_steps(total_frames=shape[2], device=device)
            rollout = self.rollout_stage.rollout(
                conditions,
                initial_noise=self._noise(shape, device=device),
                exit_steps=exit_steps,
            )
            generated = rollout.latents.detach()
            batch = generated.shape[0]
            sigma = self._score_sigma(batch, device=device)
            s = sigma.view(batch, *([1] * (generated.ndim - 1)))
            noise = torch.randn_like(generated, dtype=torch.float32)
            xt = (1.0 - s) * generated + s * noise
            target_v = noise - generated
        sigma_arg = sigma if batch > 1 else sigma.reshape(())
        fake_v = self._predict_score(
            self.fake_score,
            conditions,
            sample=xt,
            sigma=sigma_arg,
            guidance_scale=self.fake_guidance_scale,
        )
        loss = (fake_v.float() - target_v).pow(2).mean()
        value = float(loss.detach().item())
        if not math.isfinite(value):
            raise RuntimeError(f"SelfForcingDMDStack: non-finite fake-score loss {value!r}.")
        return loss, {
            "fake_score_fm_mse": value,
            "fake_score_sigma_mean": float(sigma.mean().item()),
            "fake_score_target_abs_mean": float(target_v.abs().mean().item()),
            "fake_score_generated_x0_norm": float(generated.float().pow(2).mean().item()),
        }

    # ------------------------------------------------------------------
    # Alternating updates
    # ------------------------------------------------------------------

    def _generator_update(self, part: Part) -> Tuple[float, float, Mapping[str, object]]:
        # Cached causal rollout deliberately stays in eval mode. Besides
        # disabling dropout, this prevents Diffusers' model-level gradient
        # checkpointing gate from recomputing forwards that read mutable cache
        # state. Gradients are still enabled and flow normally.
        self.generator_backend.trainable_module().eval()
        self.fake_score_backend.trainable_module().eval()
        self.real_score_backend.trainable_module().eval()
        micros = self._micro_parts(part)
        self.generator_backend.zero_grad()
        losses: List[float] = []
        metrics: List[Mapping[str, object]] = []
        for index, micro in enumerate(micros):
            self.generator_backend.set_grad_sync(index == len(micros) - 1)
            loss, item_metrics = self._generator_loss(micro)
            (loss * self._loss_scale(micro, part)).backward()
            losses.append(float(loss.detach().item()) * self._loss_scale(micro, part))
            metrics.append(item_metrics)
        grad_norm = self.generator_backend.optimizer_step(max_grad_norm=self.generator_max_grad_norm)
        self.generator_backend.on_rollout_end()
        loss = self._all_reduce_mean(self.generator_backend, sum(losses))
        return loss, float(grad_norm), self._mean_metrics(metrics)

    def _fake_score_update(self, part: Part) -> Tuple[float, float, Mapping[str, object]]:
        self.generator_backend.trainable_module().eval()
        self.fake_score_backend.trainable_module().train()
        self.real_score_backend.trainable_module().eval()
        micros = self._micro_parts(part)
        self.fake_score_backend.zero_grad()
        losses: List[float] = []
        metrics: List[Mapping[str, object]] = []
        for index, micro in enumerate(micros):
            self.fake_score_backend.set_grad_sync(index == len(micros) - 1)
            loss, item_metrics = self._fake_score_loss(micro)
            (loss * self._loss_scale(micro, part)).backward()
            losses.append(float(loss.detach().item()) * self._loss_scale(micro, part))
            metrics.append(item_metrics)
        grad_norm = self.fake_score_backend.optimizer_step(max_grad_norm=self.fake_score_max_grad_norm)
        self.fake_score_backend.on_rollout_end()
        loss = self._all_reduce_mean(self.fake_score_backend, sum(losses))
        return loss, float(grad_norm), self._mean_metrics(metrics)

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def train_track(self, part: Part, *, training_progress: float = 0.0) -> List[Dict[str, object]]:
        del training_progress
        if part.batch_size < 1:
            raise ValueError("SelfForcingDMDStack.train_track: empty Part.")
        self._align_part(part)
        generator_loss, generator_grad_norm, generator_metrics = self._generator_update(part)
        fake_losses: List[float] = []
        fake_grad_norms: List[float] = []
        fake_metrics: List[Mapping[str, object]] = []
        for _ in range(self.fake_score_updates_per_generator):
            fake_loss, fake_grad_norm, metrics = self._fake_score_update(part)
            fake_losses.append(fake_loss)
            fake_grad_norms.append(fake_grad_norm)
            fake_metrics.append(metrics)
        metrics = {
            **{f"generator/{key}": value for key, value in generator_metrics.items()},
            **{f"fake_score/{key}": value for key, value in self._mean_metrics(fake_metrics).items()},
        }
        # Plain dict is intentionally used on the distributed boundary:
        # DP_SCATTER's pytree collector understands mappings/scalars, while a
        # dataclass would be treated as an opaque scalar and keep rank 0 only.
        return [
            {
                "generator_loss": generator_loss,
                "generator_grad_norm": generator_grad_norm,
                "generator_lr": self._current_lr(self.generator_backend),
                "fake_score_loss": sum(fake_losses) / len(fake_losses),
                "fake_score_grad_norm": sum(fake_grad_norms) / len(fake_grad_norms),
                "fake_score_lr": self._current_lr(self.fake_score_backend),
                "fake_score_updates": self.fake_score_updates_per_generator,
                "metrics": metrics,
            }
            for _ in range(part.batch_size)
        ]


__all__ = ["SelfForcingDMDStack", "SelfForcingStepResult"]
