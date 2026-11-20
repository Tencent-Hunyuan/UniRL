"""Aggregation-first, frozen-target diffusion regression; see README.md for the sign and mask contracts."""

from __future__ import annotations

import math
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Type

import torch
import torch.distributed as dist

from unirl.sde.kernels import FlowSDEStrategy
from unirl.types.sample import Part
from unirl.types.sample_id import ancestor_id
from unirl.types.segments.latent import LatentSegment

from .base import (
    AlgorithmStepResult,
    BaseAlgorithmConfig,
    StageAlgorithm,
    require_timestep_chunk_size,
    typed_conditions,
)


@dataclass
class DiffusionARTConfig(BaseAlgorithmConfig):
    """Scalar-reward ART hyperparameters."""

    reference_nu: float = 0.8
    kl_mask_threshold: float = 1e-5
    timestep_chunk_size: int = 1
    group_layer: Optional[int] = None


def art_velocity_loss(new, old, guidance, *, sigma, sigma_next, reference_nu, kl_mask_threshold):
    """Return loss and detached diagnostics per row ``[B]`` from velocities ``[B, *latent]``."""
    new = new.float()
    old, guidance = old.detach().to(new), guidance.detach().to(new)
    residual = old - guidance - new
    drift = new - old
    drift_mse = drift.detach().square().flatten(1).mean(1)
    alignment = (drift.detach() * residual.detach()).flatten(1).mean(1)
    h = sigma - sigma_next
    initial = torch.isclose(sigma, torch.ones_like(sigma), atol=1e-5, rtol=0)
    variance = torch.where(initial, reference_nu**2, reference_nu**2 * sigma / (1 - sigma).clamp_min(1e-5) * h)
    coefficient = torch.where(initial, h, h * (1 + reference_nu**2 / 2))
    kl = coefficient.square() * drift_mse / (2 * variance.clamp_min(1e-10))
    active = guidance.flatten(1).ne(0).any(1)
    masked = active & (kl >= kl_mask_threshold) & (alignment > 0)
    loss = sigma * residual.square().flatten(1).mean(1)
    return torch.where(active & ~masked, loss, 0.0), {
        "mask_fraction": masked.float().mean(),
        "zero_signal_fraction": (~active).float().mean(),
        "prediction_drift_mse": drift_mse.mean(),
        "reference_kl_mean": kl.mean(),
        "reference_kl_max": kl.max(),
    }


class DiffusionART(StageAlgorithm):
    """Regress every ODE state toward ``old_prediction - full_group_guidance`` without EMA."""

    supports_multi_update = True
    required_num_updates = 2
    requires_backend = True
    recomputes_anchor = True
    anchor_fields = ("art_old_predictions",)

    def __init__(
        self,
        *,
        params: Any,
        stage: Any = None,
        pipeline: Any = None,
        stage_attr: str = "diffusion",
        backend: Any = None,
        conditions_cls: Optional[Type[Any]] = None,
        reference_nu: float = 0.8,
        kl_mask_threshold: float = 1e-5,
        timestep_chunk_size: int = 1,
        group_layer: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.stage = stage if stage is not None else getattr(pipeline, stage_attr, None)
        if self.stage is None or not callable(getattr(self.stage, "predict_noise_at_step", None)):
            raise ValueError("DiffusionART requires a stage with predict_noise_at_step.")
        if not isinstance(getattr(self.stage, "strategy", None), FlowSDEStrategy):
            raise ValueError("DiffusionART requires FlowSDEStrategy for its scheduler-reference mask.")
        if params.eta != 0 or params.guidance_scale != 1:
            raise ValueError("DiffusionART requires ODE sampling eta=0 and guidance_scale=1.")
        if params.samples_per_prompt < 2:
            raise ValueError("DiffusionART requires samples_per_prompt >= 2 for full-group covariance.")
        if not math.isfinite(reference_nu) or reference_nu <= 0:
            raise ValueError("DiffusionART reference_nu must be finite and positive.")
        if not math.isfinite(kl_mask_threshold) or kl_mask_threshold < 0:
            raise ValueError("DiffusionART kl_mask_threshold must be finite and nonnegative.")
        require_timestep_chunk_size(timestep_chunk_size, algo="DiffusionART")
        if timestep_chunk_size is None:
            raise ValueError("DiffusionART requires a finite timestep_chunk_size.")
        if group_layer is not None and (
            isinstance(group_layer, bool) or not isinstance(group_layer, int) or group_layer < 0
        ):
            raise ValueError("DiffusionART group_layer must be a nonnegative integer or None.")
        if backend is not None and getattr(backend, "ema", None) is not None:
            raise ValueError("DiffusionART is EMA-free; remove ema_cfg and ema_lora_cfg from the backend.")
        self.params, self.backend, self.conditions_cls = params, backend, conditions_cls
        self.config = DiffusionARTConfig(reference_nu, kl_mask_threshold, timestep_chunk_size, group_layer)
        self._has_global_guidance = None

    def _validate_segment_shape(self, segment):
        """Check dense ``[B, T+1, *latent]`` metadata without reading device tensor values."""
        if not isinstance(segment, LatentSegment) or segment.latents is None or segment.sigmas is None:
            raise ValueError("DiffusionART requires LatentSegment.latents and sigmas.")
        if segment.aux_latents is not None or segment.cu_seqlens is not None:
            raise ValueError(
                "DiffusionART supports dense single-modality image/video latents, not packed or joint latents."
            )
        sigmas = segment.sigmas
        if sigmas.ndim != 1 or sigmas.numel() < 2:
            raise ValueError("DiffusionART requires a one-dimensional sigma schedule with at least two positions.")
        if segment.indices is None or segment.indices.ndim != 1 or segment.indices.numel() != sigmas.numel():
            raise ValueError(
                "DiffusionART requires all T+1 trajectory states; configure dense trajectory storage at eta=0."
            )
        if segment.latents.ndim < 3 or segment.latents.shape[1] != sigmas.numel() or not segment.latents.shape[0]:
            raise ValueError("DiffusionART latents must have nonempty shape [B, T+1, *latent].")
        return sigmas.numel() - 1

    def validate_part(self, part: Part) -> None:
        """Validate schedule values once before any anchor forward; see README.md for standalone use."""
        error = None
        try:
            self._validate_local_part(part)
        except (ValueError, TypeError) as exc:
            error = str(exc)
        errors = [error]
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            errors = [None] * dist.get_world_size()
            dist.all_gather_object(errors, error)
        for rank, message in enumerate(errors):
            if message is not None:
                raise ValueError(f"DiffusionART shard {rank}: {message}")

    def _validate_local_part(self, part: Part) -> None:
        """Collect local contract failures before the rank-synchronous preflight boundary."""
        self._validate_segment_shape(part.segment)
        sigmas = part.segment.sigmas
        if not torch.isfinite(sigmas).all():
            raise ValueError("DiffusionART requires a finite sigma schedule.")
        if not ((sigmas[:-1] > sigmas[1:]).all() and sigmas[0] <= 1 and sigmas[-1] == 0):
            raise ValueError("DiffusionART requires strictly decreasing sigmas in [0, 1], ending at 0.")
        indices = part.segment.indices
        expected = torch.arange(sigmas.numel(), device=indices.device)
        if indices.dtype != torch.long or not torch.equal(indices, expected):
            raise ValueError(
                "DiffusionART requires all T+1 trajectory states; configure dense trajectory storage at eta=0."
            )
        x = part.segment.latents[:, -1]
        if part.advantages is None or part.advantages.shape != (x.shape[0],) or len(part.sample_ids) != x.shape[0]:
            raise ValueError("DiffusionART requires scalar advantages and sample_ids aligned with every endpoint.")
        if not torch.isfinite(x).all() or not torch.isfinite(part.advantages).all():
            raise ValueError("DiffusionART requires finite endpoints and advantages.")
        if any(not isinstance(sid, str) or not sid for sid in part.sample_ids):
            raise ValueError("DiffusionART requires nonempty string sample_ids.")
        layer = self.config.group_layer
        if layer is None:
            layer = max(part.sample_ids[0].count("/") - 1, 0)
        for sid in part.sample_ids:
            ancestor_id(sid, layer)
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            ri = getattr(self.backend, "rank_info", None)
            if ri is None or ri.world_size != dist.get_world_size():
                raise ValueError("DiffusionART covariance requires backend RankInfo for the train process group.")

    @contextmanager
    def _prediction_mode(self):
        """Match anchor/replay mode while retaining gradients; see README.md."""
        model = self.backend.model if self.backend is not None else self.stage.trainable_module()
        modes = [(module, module.training) for module in model.modules()]
        if any(isinstance(module, torch.nn.modules.batchnorm._BatchNorm) for module, _ in modes):
            raise ValueError("DiffusionART does not support stateful BatchNorm prediction stages.")
        model.train()
        for module, _ in modes:
            if isinstance(module, torch.nn.modules.dropout._DropoutNd):
                module.eval()
        try:
            yield
        finally:
            for module, was_training in modes:
                module.training = was_training

    def _predict(self, conditions, segment, step):
        """Predict one stored state at its original micro geometry, with stage autocast."""
        sample = segment.latents[:, step]
        dtype = getattr(self.stage, "autocast_dtype", None)
        ctx = (
            torch.autocast("cuda", dtype)
            if sample.is_cuda and dtype in (torch.bfloat16, torch.float16)
            else nullcontext()
        )
        with ctx:
            prediction = self.stage.predict_noise_at_step(
                conditions,
                sample=sample,
                sigma=segment.sigmas[step].float().expand(sample.shape[0]),
                params=self.params,
            )
        if not isinstance(prediction, torch.Tensor) or prediction.shape != sample.shape:
            raise ValueError(
                "DiffusionART predict_noise_at_step must return one velocity tensor matching the latent shape."
            )
        return prediction

    @torch.no_grad()
    def prepare_segment(self, *, conditions: Mapping, segment: LatentSegment) -> None:
        steps = self._validate_segment_shape(segment)
        conditions = typed_conditions(conditions, self.conditions_cls)
        with self._prediction_mode():
            segment.art_old_predictions = torch.stack(
                [self._predict(conditions, segment, step).detach() for step in range(steps)], dim=1
            ).cpu()

    @torch.no_grad()
    def prepare_part(self, part: Part) -> Part:
        self._validate_segment_shape(part.segment)
        x = part.segment.latents[:, -1].detach().float()
        adv = part.advantages.detach().to(x)
        layer = self.config.group_layer
        if layer is None:
            layer = max(part.sample_ids[0].count("/") - 1, 0)
        labels = [ancestor_id(sid, layer) for sid in part.sample_ids]
        distributed = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        rank = dist.get_rank() if distributed else 0
        ri = getattr(self.backend, "rank_info", None)
        local = (part.sample_ids, labels, tuple(x.shape[1:]), ri.dp_rank if ri is not None else 0)
        metadata = [local]
        if distributed:
            metadata = [None] * dist.get_world_size()
            dist.all_gather_object(metadata, local)
        owners, groups = {}, {}
        for owner, (ids, names, shape, dp_rank) in enumerate(metadata):
            if shape != tuple(x.shape[1:]):
                raise ValueError("DiffusionART DP reduction requires the same dense latent shape on all train ranks.")
            if len(set(ids)) != len(ids):
                raise ValueError("DiffusionART sample_ids must be unique within a train shard.")
            for sid, label in zip(ids, names):
                if sid in owners:
                    if owners[sid][1:] != (dp_rank, label):
                        raise ValueError("DiffusionART found a repeated sample on different DP shards or groups.")
                else:
                    owners[sid] = (owner, dp_rank, label)
                    groups.setdefault(label, []).append(sid)
        expected = self.params.samples_per_prompt
        if any(len(ids) != expected for ids in groups.values()):
            raise ValueError(f"DiffusionART needs complete groups of samples_per_prompt={expected} across train ranks.")
        guidance = torch.zeros_like(x)
        active = x.new_zeros((), dtype=torch.bool)
        for label in sorted(groups):
            rows = [i for i, name in enumerate(labels) if name == label]
            group_owners = {owners[sid][0] for sid in groups[label]}
            split = len(group_owners) > 1
            if not split and not rows:
                continue
            owned = [i for i in rows if owners[part.sample_ids[i]][0] == rank] if split else rows
            reference = x.new_zeros(x.shape[1:])
            if split:
                first = groups[label][0]
                source = owners[first][0]
                if rank == source:
                    reference.copy_(x[part.sample_ids.index(first)])
                dist.broadcast(reference, src=source)
            else:
                reference = x[rows[0]]
            moments = x.new_zeros((2, x[0].numel() + 2))
            extrema = x.new_full((2,), -torch.inf)
            if owned:
                values, rewards = (x[owned] - reference).flatten(1), adv[owned]
                moments[0, :-2] = (values * rewards[:, None]).sum(0)
                moments[1, :-2] = values.sum(0)
                moments[0, -2:] = torch.stack((rewards.sum(), rewards.new_tensor(len(owned))))
                extrema = torch.stack((rewards.max(), -rewards.min()))
            if split:
                dist.all_reduce(moments)
                dist.all_reduce(extrema, op=dist.ReduceOp.MAX)
            count = moments[0, -1]
            covariance = (moments[0, :-2] - moments[0, -2] / count * moments[1, :-2]) / count
            covariance = torch.where(extrema[0] == -extrema[1], 0.0, covariance)
            rms = covariance.double().square().mean().sqrt()
            active |= rms > 0
            direction = (covariance.double() / torch.where(rms > 0, rms, 1)).float().reshape(x.shape[1:])
            if rows:
                guidance[rows] = direction
        part.segment.art_guidance = guidance
        if distributed:
            dist.all_reduce(active, op=dist.ReduceOp.MAX)
        self._has_global_guidance = bool(active)
        return part

    def compute_loss_and_backward(self, *, conditions, segment, advantages, training_progress, loss_scale):
        steps = self._validate_segment_shape(segment)
        if segment.art_old_predictions is None or segment.art_guidance is None:
            raise ValueError("DiffusionART requires prepare_segment and full-shard prepare_part before training.")
        expected = (segment.latents.shape[0], steps, *segment.latents.shape[2:])
        if segment.art_old_predictions.shape != expected or segment.art_guidance.shape != segment.latents[:, -1].shape:
            raise ValueError("DiffusionART guidance/anchor shapes must match [B, *latent] / [B, T, *latent].")
        if self._has_global_guidance is None:
            raise ValueError("DiffusionART requires full-shard prepare_part before training.")
        if not self._has_global_guidance:
            return AlgorithmStepResult(
                loss=0.0,
                metrics={"policy_loss": 0.0, "zero_signal_fraction": 1.0},
                num_steps_or_tokens=steps,
                has_backward=False,
            )
        conditions = typed_conditions(conditions, self.conditions_cls)
        old_predictions = segment.art_old_predictions.to(device=segment.latents.device, dtype=torch.float32)
        metrics, total = {}, segment.latents.new_zeros((), dtype=torch.float32)
        with self._prediction_mode():
            for start in range(0, steps, self.config.timestep_chunk_size):
                losses = []
                for step in range(start, min(start + self.config.timestep_chunk_size, steps)):
                    new = self._predict(conditions, segment, step)
                    loss, diagnostics = art_velocity_loss(
                        new,
                        old_predictions[:, step],
                        segment.art_guidance,
                        sigma=segment.sigmas[step].to(new.device).float(),
                        sigma_next=segment.sigmas[step + 1].to(new.device).float(),
                        reference_nu=self.config.reference_nu,
                        kl_mask_threshold=self.config.kl_mask_threshold,
                    )
                    losses.append(loss.mean())
                    for name, value in diagnostics.items():
                        metrics[name] = (
                            torch.maximum(metrics.get(name, value), value)
                            if name.endswith("_max")
                            else metrics.get(name, 0.0) + value / steps
                        )
                chunk_loss = torch.stack(losses).sum() / steps
                total += chunk_loss.detach()
                (chunk_loss * loss_scale).backward()
        total, *values = torch.stack([total, *metrics.values()]).tolist()
        return AlgorithmStepResult(
            loss=total,
            metrics={"policy_loss": total, **dict(zip(metrics, values))},
            num_steps_or_tokens=steps,
            has_backward=True,
        )


__all__ = ["DiffusionART", "DiffusionARTConfig"]
