"""BagelFlowUniGRPO — FlowGRPO + velocity-MSE regularization (UniGRPO image side).

UniGRPO replaces FlowGRPO's latent-KL penalty with an unweighted MSE on the
predicted velocity field::

    L_MSE(theta) = || v_theta(x_t, t, y) - v_ref(x_t, t, y) ||^2

evaluated at the SDE-trained timesteps, where ``v_ref`` is the frozen pre-trained
base reference: under LoRA the policy with adapters disabled, under full
fine-tuning a frozen snapshot of the base weights swapped in for the v_ref
forward. This pulls the RL-tuned vector field back toward the base across all
noise levels, which mitigates reward hacking better than the timestep-weighted KL.

Subclasses :class:`FlowGRPO`: the clipped surrogate is inherited; the MSE term
adds its own backward into the same optimizer step. GRPO-Guard RatioNorm
(per-SDE-step ratio normalization) is optional via ``ratio_norm=True``.

Compute note: the MSE runs two extra velocity forwards per SDE step (``v_theta``
with grad, ``v_ref`` with adapters off / base snapshot), separate from the
inherited GRPO log-prob replay; fusing them is a follow-up.
"""

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
    """Temporarily disable LoRA adapters so a forward runs the base model.

    FSDPBackend injects LoRA via ``inject_adapter_in_model`` (not ``get_peft_model``),
    so target modules are ``peft.tuners.lora.LoraLayer``s exposing
    ``enable_adapters(bool)``. Walk the tree, flip every LoRA layer off for the
    scope, restore on exit. Yields ``True`` when at least one LoRA layer was found
    (so the forward really is the base = v_ref), ``False`` otherwise (no-op) so the
    caller can refuse rather than use the policy as its own reference.

    Self-contained here so the BAGEL UniGRPO task stays independent of any other
    algorithm module.
    """
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
        # ratio_norm (GRPO-Guard): normalize the flow importance ratio per SDE step
        # so PPO clipping actually engages (the ratio is otherwise left-shifted,
        # mean<1). grad_reweight (×1/|dt|) is the optional 2nd component, off by default.
        self.ratio_norm = bool(ratio_norm)
        self.grad_reweight = bool(grad_reweight)
        # Under old_logp_source="replay" the train stack recomputes these per 1-sample
        # micro-slice and cats them back (UnifiedModelTrainStack.prepare_segment). RatioNorm
        # needs μ_old (sde_means) refreshed at the SAME replay geometry as π_old (sde_logp)
        # — base FlowGRPO only refreshes sde_logp — so declare both when ratio_norm is on.
        self.anchor_fields = ("sde_logp", "sde_means") if self.ratio_norm else ("sde_logp",)
        # Full-FT v_ref: a frozen bf16 snapshot of the base (pre-training) weights, captured
        # lazily on the first v_ref swap (before the first optimizer step) from each trainable
        # param's local shard, keyed by param id, and swapped in per step via in-place copy.
        # Stays None under LoRA (v_ref = adapters off) or mse_weight=0 (no MSE). See
        # _reference_weights.
        self._ref_snapshot: Optional[Dict[int, torch.Tensor]] = None

    @staticmethod
    def _has_lora(transformer: Any) -> bool:
        """True if the transformer carries peft LoRA layers (LoRA training)."""
        try:
            from peft.tuners.lora import LoraLayer
        except Exception:
            return False
        return any(isinstance(m, LoraLayer) for m in transformer.modules())

    def _snapshot_reference(self, transformer: Any) -> None:
        """Deprecated shim — the v_ref base snapshot is now captured lazily inside
        :meth:`_reference_weights` (at the swap site, so the shard state matches every
        step). Kept as a no-op for any external caller; safe to remove once none remain.
        """
        return None

    @contextmanager
    def _reference_weights(self, transformer: Any) -> Iterator[None]:
        """Swap the frozen base weights into the trainable params for a v_ref forward.

        Full-FT analog of :func:`_disable_lora`. Swaps by **in-place copy of the local
        shard**, NOT a ``.data`` pointer swap: under ``fully_shard`` the forward's
        all-gather reads FSDP2's captured shard storage, so reassigning ``param.data`` is
        silently ignored (verified on a 2-GPU repro — the swapped forward equalled the
        live one), whereas ``local_view(p).copy_(...)`` writes that storage and IS honored.

        On the FIRST call it captures the base snapshot (the pre-trained weights, before
        the first optimizer step) **at this swap site** — the same shard state every
        subsequent step sees, so the copy sizes always match (a pre-loop snapshot would be
        sharded while the swap site, right after the v_theta forward, is unsharded → size
        mismatch). Stored as bf16 (the forward computes in bf16; halves the ~3.5→1.75
        GiB/GPU footprint) keyed by param id.

        Per step: stash each live local shard, copy the base in (cast to the live fp32
        master dtype), run the (no_grad) v_ref forward, then copy the trained weights back
        before any backward — so v_theta's autograd graph (recomputed under activation
        checkpointing at the post-loop backward) reads the trained weights. In-place
        copy+restore is autograd-safe here (verified on a 2-GPU backward repro).
        """
        from unirl.train.ema import local_view

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
        """Freeze the π_old anchor; under ``old_logp_source="replay"`` + RatioNorm refresh
        μ_old (``sde_means``) alongside π_old (``sde_logp``) from ONE replay.

        Base FlowGRPO recomputes only ``sde_logp`` from the pre-update replay. RatioNorm
        also reads ``segment.sde_means`` as μ_old; leaving it at the rollout (pack-B,
        bf16-packing) geometry while ``sde_logp`` is recomputed at the bs=1 replay geometry
        makes Δμ ≠ 0 at update 0 → ratio ≠ 1. So do one replay and write BOTH. The train
        stack calls this per 1-sample micro-slice (so a bs=1 replay suffices) and cats the
        declared ``anchor_fields`` back. Other modes defer to FlowGRPO unchanged.
        """
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
        # 1. Clipped surrogate (own backward). RatioNorm (GRPO-Guard) replaces the
        #    plain FlowGRPO ratio with the per-step normalized one when enabled;
        #    otherwise the inherited FlowGRPO surrogate.
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

        # 2. Velocity-MSE regularizer toward the LoRA-disabled base, at the SDE
        #    steps. Separate backward -> grads accumulate into the same step.
        typed_conds = typed_conditions(conditions, self.conditions_cls)
        device = next(self.stage.model.transformer.parameters()).device
        schedule = segment.sigmas.to(device)
        # Rebuild the conditioning KV contexts from text ONCE (the und-path prefill)
        # and reuse the resulting forward kwargs across every SDE step and both
        # v_theta / v_ref. The conditions now carry only text (see
        # BagelDiffusionConditions), and the context is a detached constant, so one
        # build serves all steps. Built here (outside the _disable_lora scope) it is
        # the LoRA-on context; v_ref then runs the velocity forward with LoRA disabled
        # over that same context — matching the prior behavior, where v_ref reused the
        # rollout (LoRA-on) context with a base velocity forward.
        forward_kwargs = self.stage.build_forward_kwargs(typed_conds, params=self.params, device=device)
        transformer = self.stage.model.transformer
        # v_ref source: LoRA -> adapters off (cheap); full FT -> a frozen bf16 snapshot of
        # the base weights, captured lazily on the first _reference_weights swap (= the
        # pre-trained base, before the first optimizer step). Both yield the pre-trained
        # reference velocity over the prebuilt context.
        full_ft_ref = not self._has_lora(transformer)
        # Compute ALL v_ref FIRST, under a SINGLE base-weight swap, storing only the
        # detached velocity tensors (tiny [seq,C] each — no autograd graphs). Then run the
        # v_theta forwards (grad-on) against those constants. This keeps the expensive
        # base-weight swap (a full fp32 master-sized stash under full FT) OUT of the window
        # where the N retained v_theta graphs + activations are live — the peak that OOM'd
        # a per-step swap. v_ref is a detached constant either way (it is `.detach()`ed into
        # the MSE), so hoisting it changes nothing numerically.
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
                v_refs = [
                    self.stage.predict_velocity_at(
                        forward_kwargs,
                        sample=segment.latents_at(s)[0].to(device),
                        sigma=schedule[s],
                        params=self.params,
                    ).detach()
                    for s in target_steps
                ]
        # Return the freed stash + v_ref activation blocks to the driver before the v_theta
        # graphs build, so this step's peak does not carry both (mirrors the train stack's
        # post-churn defrag under num_updates_per_batch>1). Full-FT only — the LoRA path's
        # v_ref leaves no stash to reclaim.
        if full_ft_ref and torch.cuda.is_available():
            torch.cuda.empty_cache()

        mse_terms: List[torch.Tensor] = []
        for step_idx, v_ref in zip(target_steps, v_refs):
            x_t = segment.latents_at(step_idx)[0].to(device)  # [seq, C] (navit bs=1)
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

    # ------------------------------------------------------------------
    # GRPO-Guard RatioNorm surrogate
    # ------------------------------------------------------------------

    def _ratio_norm_surrogate(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: "LatentSegment",
        advantages: torch.Tensor,
        training_progress: float,
        loss_scale: float,
    ) -> AlgorithmStepResult:
        """FlowGRPO clipped surrogate with GRPO-Guard RatioNorm.

        The flow importance ratio is left-shifted (mean < 1) and step-inconsistent,
        so plain clipping fails. RatioNorm normalizes the per-SDE-step log-ratio::

            log r̂ = std_var · ( log r + mean(Δμ²) / (2·std_var²) )

        where ``std_var = σ_t·√(-dt)`` is exactly FlowSDEStrategy's per-step SDE std,
        ``Δμ = μ_old − μ_θ`` (``μ_old`` = rollout SDE mean ``segment.sde_means``;
        ``μ_θ`` = replay mean), and ``mean(Δμ²)`` is over elements to match the
        mean-reduced ``log r``. The additive term cancels the ``−‖Δμ‖²/(2σ²dt)`` bias
        (mean → 0, i.e. ``r̂`` mean → 1); the ``σ_t√dt`` factor unifies variance
        across steps. The clip then runs on ``r̂``. With ``grad_reweight`` each step's
        loss is also scaled by the normalized ``1/|dt|`` (GRPO-Guard gradient
        balancing). Mirrors :meth:`FlowGRPO.compute_loss_and_backward` otherwise.

        Logs ``rn_raw_ratio_mean`` (the PRE-RatioNorm ratio): on an off-policy update
        it should be < 1 while ``ratio_mean`` (post-RatioNorm) ≈ 1 — the smoke check
        that RatioNorm is centering correctly. (On the on-policy update both ≈ 1.)
        """
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
        new_logp = replay.log_probs  # [1, S']
        mu_theta = replay.prev_sample_means  # [1, S', seq, C]
        if mu_theta is None:
            raise RuntimeError("BagelFlowUniGRPO(ratio_norm=True): stage.replay returned no prev_sample_means (μ_θ).")
        old_logp = gather_sde_field(segment.sde_logp, segment.sde_indices, target_steps, field_name="sde_logp").to(
            dtype=new_logp.dtype, device=new_logp.device
        )
        mu_old = gather_sde_field(segment.sde_means, segment.sde_indices, target_steps, field_name="sde_means").to(
            dtype=mu_theta.dtype, device=mu_theta.device
        )
        # std_var must use the same sigma_max as the SDE step that produced old/new log_probs
        # (diffuse/replay pass schedule[1]); otherwise the two disagree at the σ=1 step.
        sde_sigma_max = float(segment.sigmas[1]) if int(segment.sigmas.shape[0]) > 1 else float(segment.sigmas[0])
        std_var = self._sde_std_var(
            segment.sigmas,
            target_steps,
            eta=float(self.params.eta),
            device=new_logp.device,
            dtype=new_logp.dtype,
            sigma_max=sde_sigma_max,
        )  # [1, S']

        log_r = new_logp - old_logp  # [1, S']
        delta_mu = mu_old - mu_theta  # [1, S', seq, C]
        mean_dmu2 = (delta_mu**2).mean(dim=tuple(range(2, delta_mu.ndim)))  # [1, S'] mean over elements
        log_r_hat = std_var * (log_r + mean_dmu2 / (2.0 * std_var**2))  # [1, S']

        clip_range = _resolve_clip_range_from_schedule(self.clip_range, self.clip_schedule, training_progress)
        adv_b = advantages.detach().to(dtype=new_logp.dtype, device=new_logp.device).reshape(-1, 1).expand_as(new_logp)
        # Feed the RatioNorm'd ratio: new' − old = log r̂, so _grpo_clip_loss uses exp(log r̂) = r̂.
        loss_per_elem, ratio_metrics = _grpo_clip_loss(
            new_logp=old_logp + log_r_hat, old_logp=old_logp, advantages=adv_b, clip_range=clip_range
        )
        if self.grad_reweight:
            inv_dt = self._sde_inv_dt(segment.sigmas, target_steps, device=new_logp.device, dtype=new_logp.dtype)
            weight = inv_dt / inv_dt.mean().clamp_min(1e-12)  # normalize to mean 1 (keep loss scale)
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
            "rn_raw_ratio_mean": raw_ratio_mean,  # pre-RatioNorm (off-policy: <1); ratio_mean is post (≈1)
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
        """Per-SDE-step ``std_var = σ_t·√(-dt)`` — byte-matches ``FlowSDEStrategy.step``.

        ``σ_t = η·√(σ/(1-σ))`` (σ=1 clamped via ``sigma_max``), ``dt = σ_next − σ`` (<0).
        Returns ``[1, len(target_steps)]`` so it broadcasts against the ``[1, S']`` ratios.
        """
        sig = sigmas.to(device=device, dtype=torch.float32)
        vals: List[torch.Tensor] = []
        for s in target_steps:
            sigma = sig[s]
            sigma_next = sig[s + 1]
            dt = sigma_next - sigma  # negative (sigma decreases)
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
