"""BagelFlowUniGRPO — FlowGRPO + velocity-MSE regularization (UniGRPO image side)."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Mapping, Optional, Type

import torch

from unirl.types.conditions import Condition
from unirl.types.segments.latent import LatentSegment

from .base import (
    AlgorithmStepResult,
    _grpo_clip_loss,
    _resolve_clip_range_from_schedule,
    gather_sde_field,
    typed_conditions,
)
from .flowgrpo import FlowGRPO


@contextmanager
def _disable_lora(module: Any) -> Iterator[bool]:
    """Temporarily disable LoRA adapters so a forward runs the base model."""
    try:
        from peft.tuners.lora import LoraLayer
    except Exception:
        yield False
        return
    layers = [m for m in module.modules() if isinstance(m, LoraLayer)]
    if not layers:
        yield False
        return
    for layer in layers:
        layer.enable_adapters(False)
    try:
        yield True
    finally:
        for layer in layers:
            layer.enable_adapters(True)


class BagelFlowUniGRPO(FlowGRPO):
    """FlowGRPO with UniGRPO's velocity-MSE regularization (BAGEL image side)."""

    def __init__(
        self,
        *,
        params: Any,
        stage: Any = None,
        pipeline: Any = None,
        stage_attr: str = "diffusion",
        clip_range: float = 1e-4,
        clip_schedule: str = "constant",
        old_logp_source: str = "rollout",
        conditions_cls: Optional[Type[Any]] = None,
        mse_weight: float = 0.0,
        ratio_norm: bool = False,
        grad_reweight: bool = False,
    ) -> None:
        super().__init__(
            params=params,
            stage=stage,
            pipeline=pipeline,
            stage_attr=stage_attr,
            clip_range=clip_range,
            clip_schedule=clip_schedule,
            old_logp_source=old_logp_source,
            conditions_cls=conditions_cls,
        )
        self.mse_weight = float(mse_weight)
        self.ratio_norm = bool(ratio_norm)
        self.grad_reweight = bool(grad_reweight)
        self.anchor_fields = ("sde_logp", "sde_means") if self.ratio_norm else ("sde_logp",)
        self._ref_snapshot: Optional[Dict[int, torch.Tensor]] = None

    @staticmethod
    def _has_lora(transformer: Any) -> bool:
        """True if the transformer carries peft LoRA layers (LoRA training)."""
        try:
            from peft.tuners.lora import LoraLayer
        except Exception:
            return False
        return any(isinstance(m, LoraLayer) for m in transformer.modules())

    @contextmanager
    def _reference_weights(self, transformer: Any) -> Iterator[None]:
        """Swap the frozen base weights into the trainable params for a v_ref forward."""
        from unirl.distributed.local import local_view

        live = [p for p in transformer.parameters() if p.requires_grad]
        if not live:
            raise RuntimeError(
                "BagelFlowUniGRPO: mse_weight > 0 with no LoRA and no trainable params to snapshot "
                "as the v_ref base — the transformer is fully frozen. Enable full fine-tuning "
                "(use_lora=false unfreezes the decoder blocks) or set mse_weight=0."
            )
        if self._ref_snapshot is None:
            self._ref_snapshot = {id(p): local_view(p).detach().to(dtype=torch.bfloat16).clone() for p in live}

        stash: List[torch.Tensor] = []
        for p in live:
            lv = local_view(p)
            stash.append(lv.detach().clone())
            lv.copy_(self._ref_snapshot[id(p)])
        try:
            yield
        finally:
            for p, saved in zip(live, stash):
                local_view(p).copy_(saved)

    def prepare_segment(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: "LatentSegment",
    ) -> None:
        """Freeze the π_old anchor; with ``old_logp_source='replay'`` + RatioNorm also refresh μ_old in one replay."""
        if not (self.ratio_norm and self.old_logp_source == "replay"):
            super().prepare_segment(conditions=conditions, segment=segment)
            return
        if segment.sde_indices is None:
            return
        target_steps = self._resolve_target_steps(segment)
        if not target_steps:
            return
        typed_conds = typed_conditions(conditions, self.conditions_cls)
        with torch.no_grad():
            result = self.stage.replay(typed_conds, segment=segment, params=self.params, step_indices=target_steps)
        segment.sde_logp = result.log_probs.detach().cpu()
        segment.sde_means = result.prev_sample_means.detach().cpu()

    def compute_loss_and_backward(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: "LatentSegment",
        advantages: torch.Tensor,
        training_progress: float,
        loss_scale: float,
    ) -> AlgorithmStepResult:
        if self.ratio_norm:
            result = self._ratio_norm_surrogate(
                conditions=conditions,
                segment=segment,
                advantages=advantages,
                training_progress=training_progress,
                loss_scale=loss_scale,
            )
        else:
            result = super().compute_loss_and_backward(
                conditions=conditions,
                segment=segment,
                advantages=advantages,
                training_progress=training_progress,
                loss_scale=loss_scale,
            )
        if self.mse_weight <= 0.0 or not result.has_backward:
            return result
        target_steps = self._resolve_target_steps(segment)
        if not target_steps or segment.sigmas is None:
            return result

        typed_conds = typed_conditions(conditions, self.conditions_cls)
        device = next(self.stage.model.transformer.parameters()).device
        schedule = segment.sigmas.to(device)
        transformer = self.stage.model.transformer
        full_ft_ref = not self._has_lora(transformer)
        # Compute reference velocities before retaining trainable graphs to reduce peak memory.
        with torch.no_grad():
            if full_ft_ref:
                ref_ctx = self._reference_weights(transformer)
            else:
                ref_ctx = _disable_lora(transformer)
            with ref_ctx as disabled:
                if not full_ft_ref and not disabled:
                    raise RuntimeError(
                        "BagelFlowUniGRPO: mse_weight > 0 but found neither peft LoRA layers "
                        "to disable nor trainable params to snapshot as v_ref on "
                        "stage.model.transformer. Train with a lora_cfg or full fine-tuning, "
                        "or set mse_weight=0."
                    )
                # Rebuild reference conditioning only after policy weights are disabled.
                ref_forward_kwargs = self.stage.build_forward_kwargs(
                    typed_conds,
                    params=self.params,
                    device=device,
                    force_rebuild=bool(typed_conds.input_images),
                )
                v_refs = [
                    self.stage.predict_velocity_at(
                        ref_forward_kwargs,
                        sample=segment.latents_at(s)[0].to(device),
                        sigma=schedule[s],
                        params=self.params,
                    ).detach()
                    for s in target_steps
                ]
                del ref_forward_kwargs
        if full_ft_ref and torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Build policy contexts after freeing the reference contexts.
        forward_kwargs = self.stage.build_forward_kwargs(typed_conds, params=self.params, device=device)

        mse_terms: List[torch.Tensor] = []
        for step_idx, v_ref in zip(target_steps, v_refs):
            x_t = segment.latents_at(step_idx)[0].to(device)
            sigma = schedule[step_idx]
            v_theta = self.stage.predict_velocity_at(forward_kwargs, sample=x_t, sigma=sigma, params=self.params)
            mse_terms.append(((v_theta - v_ref) ** 2).mean())

        mse = torch.stack(mse_terms).mean()
        (self.mse_weight * mse * loss_scale).backward()

        mse_val = float(mse.detach().item())
        return AlgorithmStepResult(
            loss=result.loss + self.mse_weight * mse_val,
            metrics={**dict(result.metrics), "velocity_mse": mse_val, "mse_weight": self.mse_weight},
            num_steps_or_tokens=result.num_steps_or_tokens,
            has_backward=True,
        )

    def _ratio_norm_surrogate(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: "LatentSegment",
        advantages: torch.Tensor,
        training_progress: float,
        loss_scale: float,
    ) -> AlgorithmStepResult:
        """FlowGRPO clipped surrogate with GRPO-Guard RatioNorm."""
        target_steps = self._resolve_target_steps(segment)
        if not target_steps:
            return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False)
        if segment.sde_means is None:
            raise RuntimeError(
                "BagelFlowUniGRPO(ratio_norm=True): segment.sde_means is None. RatioNorm needs the rollout "
                "to store per-SDE-step μ_old; ensure BagelDiffusionStage.diffuse records sde_means."
            )
        typed_conds = typed_conditions(conditions, self.conditions_cls)
        replay = self.stage.replay(typed_conds, segment=segment, params=self.params, step_indices=target_steps)
        new_logp = replay.log_probs
        mu_theta = replay.prev_sample_means
        if mu_theta is None:
            raise RuntimeError("BagelFlowUniGRPO(ratio_norm=True): stage.replay returned no prev_sample_means (μ_θ).")
        old_logp = gather_sde_field(segment.sde_logp, segment.sde_indices, target_steps, field_name="sde_logp").to(
            dtype=new_logp.dtype, device=new_logp.device
        )
        mu_old = gather_sde_field(segment.sde_means, segment.sde_indices, target_steps, field_name="sde_means").to(
            dtype=mu_theta.dtype, device=mu_theta.device
        )
        sde_sigma_max = float(segment.sigmas[1]) if int(segment.sigmas.shape[0]) > 1 else float(segment.sigmas[0])
        std_var = self._sde_std_var(
            segment.sigmas,
            target_steps,
            eta=float(self.params.eta),
            device=new_logp.device,
            dtype=new_logp.dtype,
            sigma_max=sde_sigma_max,
        )

        log_r = new_logp - old_logp
        delta_mu = mu_old - mu_theta
        mean_dmu2 = (delta_mu**2).mean(dim=tuple(range(2, delta_mu.ndim)))
        log_r_hat = std_var * (log_r + mean_dmu2 / (2.0 * std_var**2))

        clip_range = _resolve_clip_range_from_schedule(self.clip_range, self.clip_schedule, training_progress)
        adv_b = advantages.detach().to(dtype=new_logp.dtype, device=new_logp.device).reshape(-1, 1).expand_as(new_logp)
        loss_per_elem, ratio_metrics = _grpo_clip_loss(
            new_logp=old_logp + log_r_hat, old_logp=old_logp, advantages=adv_b, clip_range=clip_range
        )
        if self.grad_reweight:
            inv_dt = self._sde_inv_dt(segment.sigmas, target_steps, device=new_logp.device, dtype=new_logp.dtype)
            weight = inv_dt / inv_dt.mean().clamp_min(1e-12)
            loss = (loss_per_elem * weight).mean()
        else:
            loss = loss_per_elem.mean()
        (loss * loss_scale).backward()

        with torch.no_grad():
            raw_ratio_mean = float(torch.exp(log_r).mean().item())
        metrics: Dict[str, Any] = {
            "policy_loss": float(loss.detach().item()),
            "clip_range": float(clip_range),
            **{k: float(v.item()) for k, v in ratio_metrics.items()},
            "rn_raw_ratio_mean": raw_ratio_mean,
            "rn_delta_mu_sq_mean": float(mean_dmu2.mean().item()),
            "ratio_norm": 1.0,
            "grad_reweight": float(bool(self.grad_reweight)),
        }
        return AlgorithmStepResult(
            loss=float(loss.detach().item()),
            metrics=metrics,
            num_steps_or_tokens=len(target_steps),
            has_backward=True,
        )

    @staticmethod
    def _sde_std_var(
        sigmas: torch.Tensor,
        target_steps: List[int],
        *,
        eta: float,
        device: Any,
        dtype: Any,
        sigma_max: float = 0.99,
    ) -> torch.Tensor:
        """Per-SDE-step ``std_var = σ_t·√(-dt)`` as ``[1, len(target_steps)]``, broadcasting against ``[1, S']``."""
        sig = sigmas.to(device=device, dtype=torch.float32)
        vals: List[torch.Tensor] = []
        for s in target_steps:
            sigma = sig[s]
            sigma_next = sig[s + 1]
            dt = sigma_next - sigma
            denom = 1.0 - (sigma_max if float(sigma) == 1.0 else float(sigma))
            std_dev_t = torch.sqrt(sigma / denom) * float(eta)
            vals.append(std_dev_t * torch.sqrt(-dt))
        return torch.stack(vals).to(dtype=dtype).reshape(1, -1)

    @staticmethod
    def _sde_inv_dt(
        sigmas: torch.Tensor,
        target_steps: List[int],
        *,
        device: Any,
        dtype: Any,
    ) -> torch.Tensor:
        """Per-SDE-step ``1/|dt| = 1/(σ − σ_next)`` for the GRPO-Guard gradient reweight."""
        sig = sigmas.to(device=device, dtype=torch.float32)
        vals = [1.0 / float(sig[s] - sig[s + 1]) for s in target_steps]
        return torch.tensor(vals, device=device, dtype=dtype).reshape(1, -1)


__all__ = ["BagelFlowUniGRPO"]
