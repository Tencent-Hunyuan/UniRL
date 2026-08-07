"""Self-forcing objectives for causal video diffusion."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Type

import torch

from unirl.algorithms.base import AlgorithmStepResult, StageAlgorithm, typed_conditions
from unirl.types.conditions import Condition
from unirl.types.segments.latent import LatentSegment


class SelfForcingDMD(StageAlgorithm):
    """Generator-side DMD update over a block-causal self rollout.

    ``real_score`` is the frozen target distribution and ``fake_score`` tracks
    the generator distribution. The dedicated trainer owns their lifecycle and
    alternating optimizer updates; this algorithm implements the exact
    generator-side surrogate gradient.
    """

    supports_multi_update = False
    requires_advantages = False
    loss_weighting = "sample"

    def __init__(
        self,
        *,
        pipeline: Any,
        params: Any,
        conditions_cls: Optional[Type[Any]] = None,
        real_score: Any = None,
        fake_score: Any = None,
        score_sigma_min: float = 0.02,
        score_sigma_max: float = 0.98,
        normalization_eps: float = 1e-6,
        eval_seed: int = 42,
    ) -> None:
        self.rollout_stage = pipeline.self_forcing
        if self.rollout_stage is None:
            raise ValueError("SelfForcingDMD requires pipeline.self_forcing.")
        self.params = params
        self.conditions_cls = conditions_cls
        self.real_score = real_score if real_score is not None else pipeline.diffusion
        self.fake_score = fake_score
        self.score_sigma_min = float(score_sigma_min)
        self.score_sigma_max = float(score_sigma_max)
        self.normalization_eps = float(normalization_eps)
        self.eval_seed = int(eval_seed)
        if self.fake_score is None:
            raise ValueError(
                "SelfForcingDMD requires a separately maintained fake_score stage; "
                "use the dedicated Self-Forcing trainer."
            )
        if not 0.0 < self.score_sigma_min < self.score_sigma_max < 1.0:
            raise ValueError("score sigma bounds must satisfy 0 < min < max < 1.")

    @staticmethod
    def _clean_latents(segment: LatentSegment) -> Optional[torch.Tensor]:
        latents = getattr(segment, "latents", None)
        if latents is None:
            return None
        return latents[:, -1] if latents.ndim == 6 else latents

    @staticmethod
    def _noise(x: torch.Tensor, generator: Optional[torch.Generator]) -> torch.Tensor:
        return torch.randn(x.shape, device=x.device, dtype=torch.float32, generator=generator)

    def _dmd_loss(
        self,
        conditions: Any,
        *,
        shape_like: torch.Tensor,
        generator: Optional[torch.Generator],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        rollout = self.rollout_stage.rollout(
            conditions,
            initial_noise=self._noise(shape_like, generator),
            generator=generator,
        )
        generated = rollout.latents
        batch = int(generated.shape[0])
        with torch.no_grad():
            sigma = torch.rand(batch, device=generated.device, generator=generator)
            sigma = self.score_sigma_min + sigma * (self.score_sigma_max - self.score_sigma_min)
            s = sigma.view(batch, *([1] * (generated.ndim - 1)))
            xt = (1.0 - s) * generated.detach() + s * self._noise(generated, generator)
            sigma_arg = sigma if batch > 1 else sigma.reshape(())
            fake_v = self.fake_score.predict_noise_at_step(
                conditions, sample=xt, sigma=sigma_arg, params=self.params
            )
            real_v = self.real_score.predict_noise_at_step(
                conditions, sample=xt, sigma=sigma_arg, params=self.params
            )
            fake_x0 = xt - s * fake_v
            real_x0 = xt - s * real_v
            grad = fake_x0 - real_x0
            normalizer = (generated.detach() - real_x0).abs().mean(
                dim=tuple(range(1, generated.ndim)), keepdim=True
            )
            grad = torch.nan_to_num(grad / normalizer.clamp_min(self.normalization_eps))

        target = (generated - grad).detach()
        loss = 0.5 * (generated.double() - target.double()).pow(2)[rollout.gradient_mask].mean()
        return loss, {
            "dmd_grad_abs_mean": float(grad.abs().mean().item()),
            "score_sigma_mean": float(sigma.mean().item()),
            "rollout_exit_step": float(rollout.exit_step),
            "generated_x0_norm": float(generated.detach().float().pow(2).mean().item()),
        }

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
        clean = self._clean_latents(segment)
        if clean is None:
            return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False)
        loss, metrics = self._dmd_loss(
            typed_conditions(conditions, self.conditions_cls),
            shape_like=clean,
            generator=None,
        )
        (loss * loss_scale).backward()
        value = float(loss.detach().item())
        if not math.isfinite(value):
            raise RuntimeError(f"SelfForcingDMD: non-finite generator loss {value!r}.")
        metrics["dmd_loss"] = value
        return AlgorithmStepResult(
            loss=value,
            metrics=metrics,
            num_steps_or_tokens=int(clean.shape[0]),
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
        clean = self._clean_latents(segment)
        if clean is None:
            return 0.0, 0.0
        key = "|".join(str(value) for value in (sample_ids or range(clean.shape[0])))
        digest = hashlib.sha256(f"{self.eval_seed}:{key}".encode()).digest()
        seed = int.from_bytes(digest[:8], "little") & 0x7FFF_FFFF_FFFF_FFFF
        generator = torch.Generator(device=clean.device).manual_seed(seed)
        loss, _ = self._dmd_loss(
            typed_conditions(conditions, self.conditions_cls),
            shape_like=clean,
            generator=generator,
        )
        mask = getattr(segment, "loss_mask", None)
        count = float(mask.sum().item()) if mask is not None else float(clean.shape[0])
        return float(loss.item()) * count, count


__all__ = ["SelfForcingDMD"]
