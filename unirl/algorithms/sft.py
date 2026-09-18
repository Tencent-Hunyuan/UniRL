"""Supervised finetuning losses — the anchor-free algorithms the base class anticipates."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Type

import torch

from unirl.types.conditions import Condition
from unirl.types.segments.latent import LatentSegment
from unirl.types.segments.text import TextSegment

from .base import AlgorithmStepResult, StageAlgorithm, typed_conditions

_LOSS_AGG_MODES = ("token-mean", "seq-mean-token-mean", "seq-mean-token-sum-norm")


class SFT(StageAlgorithm):
    """Masked next-token cross-entropy over an AR ``TextSegment``."""

    supports_multi_update = True
    requires_advantages = False

    def __init__(
        self,
        *,
        stage: Any = None,
        pipeline: Any = None,
        stage_attr: str = "ar",
        loss_agg_mode: str = "token-mean",
        horizon: int = 8192,
        conditions_cls: Optional[Type[Any]] = None,
    ) -> None:
        super().__init__()
        if stage is None and pipeline is None:
            raise ValueError("SFT: either `stage` or `pipeline` must be provided")
        if stage is None:
            stage = getattr(pipeline, stage_attr)
        if loss_agg_mode not in _LOSS_AGG_MODES:
            raise ValueError(f"SFT: loss_agg_mode must be one of {_LOSS_AGG_MODES}; got {loss_agg_mode!r}.")
        self.stage = stage
        self.loss_agg_mode = loss_agg_mode
        self.horizon = horizon
        self.conditions_cls = conditions_cls
        self.loss_weighting = "token" if self.loss_agg_mode == "token-mean" else "sample"

    def compute_loss_and_backward(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: "TextSegment",
        advantages: Optional[torch.Tensor],
        training_progress: float,
        loss_scale: float,
    ) -> AlgorithmStepResult:
        del advantages, training_progress
        if segment is None or segment.tokens is None or segment.lengths is None:
            return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False)
        if segment.tokens.shape[0] == 0:
            return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False)

        loss, aux = self._masked_ce(conditions, segment)
        (loss * loss_scale).backward()

        metrics: Dict[str, Any] = {
            "sft_ce": aux["token_mean"],
            "sft_ppl": math.exp(min(aux["token_mean"], 20.0)),
            "sft_tokens": aux["tokens"],
            "response_len_mean": float(segment.lengths.float().mean().item()),
        }
        return AlgorithmStepResult(
            loss=float(loss.detach().item()),
            metrics=metrics,
            num_steps_or_tokens=round(aux["tokens"]),
            has_backward=True,
        )

    @torch.no_grad()
    def evaluate_loss(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: "TextSegment",
        sample_ids: Optional[Sequence[str]] = None,
    ) -> Tuple[float, float]:
        """Forward-only ``(objective_sum, objective_weight)`` for validation."""
        del sample_ids
        if segment is None or segment.tokens is None or segment.tokens.shape[0] == 0:
            return 0.0, 0.0
        _, aux = self._masked_ce(conditions, segment)
        return aux["objective_sum"], aux["objective_weight"]

    def _masked_ce(
        self,
        conditions: Mapping[str, Condition],
        segment: "TextSegment",
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """One teacher-forced forward → (aggregated loss, reduction stats)."""
        typed_conds = typed_conditions(conditions, self.conditions_cls)
        new_logp = self.stage.replay(typed_conds, segment=segment, temperature=1.0)
        nll = -new_logp

        mask = segment.loss_mask
        if mask is not None:
            mask = mask.to(dtype=nll.dtype, device=nll.device)
            nll = nll * mask
            tokens = float(mask.sum().item())
        else:
            tokens = float(nll.numel())
        ce_sum = nll.sum()

        if self.loss_agg_mode == "token-mean":
            objective_sum = ce_sum
            objective_weight = tokens
        else:
            parts = torch.split(nll, segment.lengths.tolist())
            if mask is not None:
                mask_parts = torch.split(mask, segment.lengths.tolist())
                token_weights = [float(m.sum().item()) for m in mask_parts]
            else:
                token_weights = [float(p.numel()) for p in parts]

            valid_parts = [(p, weight) for p, weight in zip(parts, token_weights) if weight > 0.0]
            if self.loss_agg_mode == "seq-mean-token-sum-norm":
                per_seq = [p.sum() / self.horizon for p, _ in valid_parts]
            else:
                per_seq = [p.sum() / weight for p, weight in valid_parts]
            objective_sum = torch.stack(per_seq).sum() if per_seq else ce_sum * 0.0
            objective_weight = float(len(per_seq))

        loss = objective_sum / max(objective_weight, 1.0)

        token_mean = float((ce_sum / max(tokens, 1.0)).detach().item())
        if not math.isfinite(token_mean):
            raise RuntimeError(f"SFT: non-finite CE (token_mean={token_mean!r}, tokens={tokens}).")
        return loss, {
            "ce_sum": float(ce_sum.detach().item()),
            "tokens": tokens,
            "token_mean": token_mean,
            "objective_sum": float(objective_sum.detach().item()),
            "objective_weight": objective_weight,
        }


class FlowMatchSFT(StageAlgorithm):
    """Flow-matching velocity MSE over a dataset x0-only ``LatentSegment``."""

    supports_multi_update = True
    requires_advantages = False
    loss_weighting = "sample"

    def __init__(
        self,
        *,
        params: Any,
        stage: Any = None,
        pipeline: Any = None,
        stage_attr: str = "diffusion",
        conditions_cls: Optional[Type[Any]] = None,
        timestep_sampling: str = "logit_normal",
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        timestep_shift: float = 1.0,
        sigma_min: float = 1e-4,
        eval_seed: int = 42,
    ) -> None:
        super().__init__()
        if stage is None and pipeline is None:
            raise ValueError("FlowMatchSFT: either `stage` or `pipeline` must be provided")
        if stage is None:
            stage = getattr(pipeline, stage_attr)
        if timestep_sampling not in ("uniform", "logit_normal"):
            raise ValueError(
                f"FlowMatchSFT: timestep_sampling must be 'uniform' or 'logit_normal'; got {timestep_sampling!r}."
            )
        if not timestep_shift > 0.0:
            raise ValueError(f"FlowMatchSFT: timestep_shift must be > 0; got {timestep_shift!r}.")
        if not 0.0 < sigma_min < 0.5:
            raise ValueError(f"FlowMatchSFT: sigma_min must lie in (0, 0.5); got {sigma_min!r}.")
        self.stage = stage
        self.params = params
        self.conditions_cls = conditions_cls
        self.timestep_sampling = timestep_sampling
        self.logit_mean = logit_mean
        self.logit_std = logit_std
        self.timestep_shift = timestep_shift
        self.sigma_min = sigma_min
        self.eval_seed = eval_seed

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
        x0 = self._clean_latents(segment)
        if x0 is None:
            return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False)
        loss, aux = self._velocity_mse(conditions, x0, generator=None)
        (loss * loss_scale).backward()
        metrics: Dict[str, Any] = {
            "fm_mse": float(loss.detach().item()),
            "sigma_mean": aux["sigma_mean"],
            "x0_norm": aux["x0_norm"],
        }
        if not math.isfinite(metrics["fm_mse"]):
            raise RuntimeError(f"FlowMatchSFT: non-finite loss {metrics['fm_mse']!r}.")
        return AlgorithmStepResult(
            loss=float(loss.detach().item()),
            metrics=metrics,
            num_steps_or_tokens=x0.shape[0],
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
        """Forward-only ``(mse_sum, sample_count)`` at a FIXED (σ, ε) draw."""
        x0 = self._clean_latents(segment)
        if x0 is None:
            return 0.0, 0.0
        sigma, noise = self._eval_draws(x0, sample_ids)
        _, aux = self._velocity_mse(conditions, x0, generator=None, sigma=sigma, noise=noise)
        per_sample = aux["per_sample_mse"]
        mask = getattr(segment, "loss_mask", None)
        if mask is not None:
            mask = mask.to(dtype=per_sample.dtype, device=per_sample.device).flatten()
            if mask.shape[0] != per_sample.shape[0]:
                raise ValueError(
                    f"FlowMatchSFT.evaluate_loss: loss_mask length {mask.shape[0]} != "
                    f"batch {per_sample.shape[0]} (expected one weight per sample)."
                )
            return float((per_sample * mask).sum().item()), float(mask.sum().item())
        return float(per_sample.sum().item()), float(per_sample.shape[0])

    @staticmethod
    def _clean_latents(segment: LatentSegment) -> Optional[torch.Tensor]:
        if segment is None or segment.latents is None:
            raise ValueError(
                "FlowMatchSFT requires segment.latents with the clean target latent at "
                "the last trajectory position (a SupervisedTrackBuilder-built x0-only segment)."
            )
        x0 = segment.latents[:, -1]
        if x0.numel() == 0:
            return None
        return x0.float()

    def _draw_sigma(self, batch: int, device: torch.device, generator: Optional[torch.Generator]) -> torch.Tensor:
        return draw_shifted_sigma(
            batch,
            timestep_sampling=self.timestep_sampling,
            logit_mean=self.logit_mean,
            logit_std=self.logit_std,
            shift=self.timestep_shift,
            sigma_min=self.sigma_min,
            device=device,
            generator=generator,
        )

    def _eval_draws(self, x0: torch.Tensor, sample_ids: Optional[Sequence[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-sample deterministic ``(σ, ε)`` for eval, keyed on sample id so noising is batch-independent."""
        batch = x0.shape[0]
        device = x0.device
        sigmas: list[torch.Tensor] = []
        noises: list[torch.Tensor] = []
        for i in range(batch):
            key = str(sample_ids[i]) if sample_ids is not None and i < len(sample_ids) else str(i)
            generator = torch.Generator(device=device)
            generator.manual_seed(sample_eval_seed(self.eval_seed, key))
            sigmas.append(self._draw_sigma(1, device, generator))
            noises.append(torch.randn(x0[i].shape, device=device, dtype=torch.float32, generator=generator))
        return torch.cat(sigmas, dim=0), torch.stack(noises, dim=0)

    def _velocity_mse(
        self,
        conditions: Mapping[str, Condition],
        x0: torch.Tensor,
        *,
        generator: Optional[torch.Generator],
        sigma: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        batch = x0.shape[0]
        device = x0.device
        typed_conds = typed_conditions(conditions, self.conditions_cls)

        if sigma is None:
            sigma = self._draw_sigma(batch, device, generator)
        if noise is None:
            noise = torch.randn(x0.shape, device=device, dtype=torch.float32, generator=generator)
        s = sigma.view(batch, *([1] * (x0.ndim - 1)))
        xt = (1.0 - s) * x0 + s * noise
        v_target = noise - x0

        sigma_arg = sigma if batch > 1 else sigma.reshape(())
        v_pred = self.stage.predict_noise_at_step(typed_conds, sample=xt, sigma=sigma_arg, params=self.params)
        if v_pred.ndim == x0.ndim - 1:
            v_pred = v_pred.unsqueeze(0)
        per_sample = (v_pred.float() - v_target).pow(2).mean(dim=tuple(range(1, x0.ndim)))
        loss = per_sample.mean()
        aux = {
            "per_sample_mse": per_sample.detach(),
            "sigma_mean": float(sigma.mean().item()),
            "x0_norm": float(x0.pow(2).mean().detach().item()),
        }
        return loss, aux


def flow_shift_sigma(u: torch.Tensor, shift: float) -> torch.Tensor:
    """The FlowMatch static time-shift warp ``σ = s·u / (1 + (s-1)·u)`` (test hook)."""
    return (shift * u) / (1.0 + (shift - 1.0) * u)


def draw_shifted_sigma(
    batch: int,
    *,
    timestep_sampling: str,
    logit_mean: float,
    logit_std: float,
    shift: float,
    sigma_min: float,
    device: torch.device,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    """Draw ``[batch]`` fp32 training sigmas: base draw -> ``flow_shift_sigma`` warp -> clamp."""
    if timestep_sampling == "logit_normal":
        z = torch.randn(batch, device=device, dtype=torch.float32, generator=generator)
        u = torch.sigmoid(z * logit_std + logit_mean)
    elif timestep_sampling == "uniform":
        u = torch.rand(batch, device=device, dtype=torch.float32, generator=generator)
    else:
        raise ValueError(
            f"draw_shifted_sigma: timestep_sampling must be 'uniform' or 'logit_normal'; got {timestep_sampling!r}."
        )
    sigma = flow_shift_sigma(u, shift)
    return sigma.clamp(min=sigma_min, max=1.0 - sigma_min)


def sample_eval_seed(eval_seed: int, key: str) -> int:
    """Stable int64 eval seed from ``eval_seed`` + per-sample key; shared so eval draws compare across algorithms."""
    digest = hashlib.sha256(f"{eval_seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFF_FFFF_FFFF_FFFF


__all__ = ["SFT", "FlowMatchSFT", "draw_shifted_sigma", "flow_shift_sigma", "sample_eval_seed"]
