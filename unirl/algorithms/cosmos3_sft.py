"""Joint video/action flow-matching SFT for Cosmos3."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Type

import torch
import torch.nn.functional as F

from unirl.types.conditions import Condition
from unirl.types.segments.latent import LatentSegment

from .base import AlgorithmStepResult, StageAlgorithm
from .sft import draw_shifted_sigma, sample_eval_seed

# Cosmos3SFTConfig.time_dist -> draw_shifted_sigma timestep_sampling names.
_TIME_DIST_TO_SAMPLING = {"logitnormal": "logit_normal", "uniform": "uniform"}


class Cosmos3JointFlowMatchSFT(StageAlgorithm):
    """Per-sample packed velocity MSE over video and optional action targets."""

    supports_multi_update = True
    requires_advantages = False
    loss_weighting = "sample"

    def __init__(
        self,
        *,
        stage: Any = None,
        pipeline: Any = None,
        stage_attr: str = "joint",
        conditions_cls: Optional[Type[Any]] = None,
        eval_seed: int = 42,
    ) -> None:
        super().__init__()
        if stage is None and pipeline is None:
            raise ValueError("Cosmos3JointFlowMatchSFT: either `stage` or `pipeline` must be provided.")
        self.stage = stage if stage is not None else getattr(pipeline, stage_attr)
        self.conditions_cls = conditions_cls
        self.eval_seed = int(eval_seed)

    def compute_loss_and_backward(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: LatentSegment,
        advantages: Optional[torch.Tensor],
        training_progress: float,
        loss_scale: float,
    ) -> AlgorithmStepResult:
        del advantages, training_progress
        total, vision, action, sigmas = self._per_sample_losses(conditions, segment, sample_ids=None)
        if total.numel() == 0:
            return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False)
        mask = self._loss_mask(segment, total)
        weight = mask.sum()
        loss = (total * mask).sum() / weight.clamp_min(1.0)
        (loss * loss_scale).backward()

        denom = weight.clamp_min(1.0)
        metrics: Dict[str, float] = {
            "cosmos3_total": float(loss.detach().item()),
            "cosmos3_vision_mse": float((vision.detach() * mask).sum().div(denom).item()),
            "sigma_mean": float((sigmas.detach() * mask).sum().div(denom).item()),
        }
        if action is not None:
            metrics["cosmos3_action_mse"] = float((action.detach() * mask).sum().div(denom).item())
        if not all(math.isfinite(value) for value in metrics.values()):
            raise RuntimeError(f"Cosmos3JointFlowMatchSFT: non-finite metrics {metrics!r}.")
        return AlgorithmStepResult(
            loss=float(loss.detach().item()),
            metrics=metrics,
            num_steps_or_tokens=round(float(weight.item())),
            has_backward=True,
        )

    @torch.no_grad()
    def evaluate_loss(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: LatentSegment,
        sample_ids: Optional[Sequence[str]] = None,
    ) -> Tuple[float, float]:
        total, _vision, _action, _sigmas = self._per_sample_losses(
            conditions,
            segment,
            sample_ids=sample_ids,
        )
        if total.numel() == 0:
            return 0.0, 0.0
        mask = self._loss_mask(segment, total)
        return float((total * mask).sum().item()), float(mask.sum().item())

    def _per_sample_losses(
        self,
        conditions: Mapping[str, Condition],
        segment: LatentSegment,
        *,
        sample_ids: Optional[Sequence[str]],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        condition = conditions.get("cosmos3")
        if condition is None:
            raise TypeError("Cosmos3JointFlowMatchSFT requires a conditions['cosmos3'] slot.")
        if self.conditions_cls is not None and not isinstance(condition, self.conditions_cls):
            raise TypeError(
                f"Cosmos3JointFlowMatchSFT requires conditions['cosmos3'] to be "
                f"{self.conditions_cls.__name__}, got {type(condition).__name__}."
            )
        if condition.input_ids is None or condition.cu_seqlens is None:
            raise ValueError("Cosmos3JointFlowMatchSFT: packed input_ids are missing.")
        if condition.fps is None or condition.flow_shifts is None:
            raise ValueError("Cosmos3JointFlowMatchSFT: fps/flow_shifts are missing.")
        if segment is None or segment.latents is None:
            raise ValueError("Cosmos3JointFlowMatchSFT requires an x0-only LatentSegment.")
        x0 = segment.latents[:, -1].float()
        if x0.numel() == 0:
            empty = x0.new_empty((0,))
            return empty, empty, None, empty
        batch = x0.shape[0]
        if condition.batch_size != batch:
            raise ValueError(
                f"Cosmos3JointFlowMatchSFT: condition batch {condition.batch_size} != latent batch {batch}."
            )
        if condition.actions is not None and condition.actions.shape[0] != batch:
            raise ValueError(
                f"Cosmos3JointFlowMatchSFT: action batch {condition.actions.shape[0]} != latent batch {batch}."
            )

        total_rows: list[torch.Tensor] = []
        vision_rows: list[torch.Tensor] = []
        action_rows: list[torch.Tensor] = []
        sigma_rows: list[torch.Tensor] = []
        cfg = self.stage.config
        sampling = _TIME_DIST_TO_SAMPLING.get(cfg.time_dist)
        if sampling is None:
            raise ValueError(f"Unknown time_dist={cfg.time_dist!r}; expected 'logitnormal' or 'uniform'.")
        # One .item() sync per field, not per sample.
        cu = condition.cu_seqlens.tolist()
        fps_rows = condition.fps.tolist()
        shift_rows = condition.flow_shifts.tolist()
        for i in range(batch):
            start, end = int(cu[i]), int(cu[i + 1])
            input_ids = condition.input_ids[start:end].tolist()
            generator = self._eval_generator(x0.device, sample_ids, i) if sample_ids is not None else None
            sigma = draw_shifted_sigma(
                1,
                timestep_sampling=sampling,
                logit_mean=cfg.logitnormal_mean,
                logit_std=cfg.logitnormal_std,
                shift=float(shift_rows[i]),
                sigma_min=1e-4,
                device=x0.device,
                generator=generator,
            )
            prediction = self.stage.predict_velocity(
                input_ids=input_ids,
                x0=x0[i],
                fps=float(fps_rows[i]),
                sigma=sigma,
                actions=condition.actions[i] if condition.actions is not None else None,
                generator=generator,
            )
            vision_loss = F.mse_loss(prediction.vision_pred.float(), prediction.vision_target.float())
            total_loss = float(cfg.vision_loss_weight) * vision_loss
            vision_rows.append(vision_loss)
            if prediction.action_pred is not None:
                if prediction.action_target is None:
                    raise RuntimeError("Cosmos3JointFlowMatchSFT: action prediction has no target.")
                action_loss = F.mse_loss(prediction.action_pred.float(), prediction.action_target.float())
                total_loss = total_loss + float(cfg.action_loss_weight) * action_loss
                action_rows.append(action_loss)
            total_rows.append(total_loss)
            sigma_rows.append(prediction.sigma.reshape(()))

        action_tensor: Optional[torch.Tensor] = None
        if action_rows:
            if len(action_rows) != batch:
                raise RuntimeError("Cosmos3JointFlowMatchSFT: a batch may not mix action and video-only rows.")
            action_tensor = torch.stack(action_rows)
        return (
            torch.stack(total_rows),
            torch.stack(vision_rows),
            action_tensor,
            torch.stack(sigma_rows),
        )

    @staticmethod
    def _loss_mask(segment: LatentSegment, values: torch.Tensor) -> torch.Tensor:
        mask = segment.loss_mask
        if mask is None:
            return torch.ones_like(values, dtype=torch.float32)
        mask = mask.to(device=values.device, dtype=torch.float32).flatten()
        if mask.shape != values.shape:
            raise ValueError(
                f"Cosmos3JointFlowMatchSFT: loss_mask shape {tuple(mask.shape)} != loss shape {tuple(values.shape)}."
            )
        return mask

    def _eval_generator(
        self,
        device: torch.device,
        sample_ids: Optional[Sequence[str]],
        row: int,
    ) -> torch.Generator:
        key = str(sample_ids[row]) if sample_ids is not None and row < len(sample_ids) else str(row)
        generator = torch.Generator(device=device)
        generator.manual_seed(sample_eval_seed(self.eval_seed, key))
        return generator


__all__ = ["Cosmos3JointFlowMatchSFT"]
