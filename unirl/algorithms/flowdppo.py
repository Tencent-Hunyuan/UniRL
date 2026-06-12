"""FlowDPPO: KL-divergence-based masking for diffusion RL.

Implements :class:`FlowDPPO` — a :class:`StageAlgorithm` that replaces
PPO-style ratio clipping with a KL-ADV masking criterion. Uses
``prev_sample_means`` from replay to compute Gaussian KL between old and
new policy, then masks updates where KL is high AND the ratio direction is
aligned with advantage (i.e. overly aggressive policy updates).

Module-level helpers ``_gaussian_kl_div`` and ``_flowdppo_kl_adv_loss`` contain
the core math; the class wires them into the stage-driven training contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, Dict, List, Mapping, Optional, Tuple, Type

import torch

from unirl.config.require import require
from unirl.types.conditions import Condition
from unirl.types.segments.latent import LatentSegment

from .base import AlgorithmStepResult, BaseAlgorithmConfig, StageAlgorithm, gather_sde_field, typed_conditions


@dataclass
class FlowDPPOConfig(BaseAlgorithmConfig):
    stage_attr: str = "diffusion"
    conditions_cls: str = ""
    kl_mask_threshold: float = 1e-5
    add_kl_coefficient: bool = True
    old_logp_source: str = "rollout"
    params: Any = dc_field(default=None)


# ---------------------------------------------------------------------------
# Loss helpers
# ---------------------------------------------------------------------------


def _gaussian_kl_div(p: torch.Tensor, q: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """KL-style squared error between Gaussian means scaled by variance (x-space).

    For two Gaussians N(p, sigma^2) and N(q, sigma^2) with shared variance,
    KL(N(p,...) || N(q,...)) = (p - q)^2 / (2 * sigma^2).

    Returns per-element KL; caller reduces over spatial dims.
    """
    return (p - q) ** 2 / (2 * sigma**2)


def _flowdppo_kl_adv_loss(
    *,
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    new_means: torch.Tensor,
    old_means: torch.Tensor,
    advantages: torch.Tensor,
    sigma_t: torch.Tensor,
    kl_mask_threshold: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """FlowDPPO KL-ADV masking loss.

    Instead of PPO's ratio clipping, this function:
    1. Computes per-sample KL between new and old policy means
    2. Creates a KL mask (keep if KL < threshold)
    3. Among high-KL samples, masks out those where the ratio direction is
       aligned with advantage (overly aggressive updates that push the policy
       too far in the reward-improving direction)

    Args:
        new_logp: New policy log-probs at current weights. ``[B, S']``.
        old_logp: Old policy log-probs frozen at pre-update weights. ``[B, S']``.
        new_means: New policy prev_sample_means. ``[B, S', *latent_shape]``.
        old_means: Old policy prev_sample_means. ``[B, S', *latent_shape]``.
        advantages: Per-element advantages (broadcast). ``[B, S']``.
        sigma_t: Per-step noise scale for KL normalization. Broadcastable
            to ``new_means`` shape (typically ``[1, S', 1, 1, 1]``).
        kl_mask_threshold: KL threshold below which updates pass freely.

    Returns:
        ``(loss_per_element, metrics_dict)``. Reduction is the caller's job.
    """
    log_diff = new_logp - old_logp
    ratio = torch.exp(log_diff)
    adv = advantages.detach()
    unclipped_loss = -adv * ratio

    # Compute per-sample KL between new and old policy means
    kl_per_elem = _gaussian_kl_div(new_means, old_means, sigma_t)  # [B, S', C, H, W]
    kl_per_sample = kl_per_elem.mean(dim=tuple(range(2, kl_per_elem.ndim)))  # [B, S']

    # KL mask: keep samples where KL < threshold (low divergence → safe to update)
    kl_mask = kl_per_sample < kl_mask_threshold

    # Advantage-aware masking: among high-KL samples, remove those whose
    # update direction conflicts with the advantage signal.
    # - pos_rm: KL high AND ratio > 1 (increasing prob) AND adv > 0
    #   → policy is already moving in reward direction but too aggressively
    # - neg_rm: KL high AND ratio < 1 (decreasing prob) AND adv < 0
    #   → policy is already moving away from bad actions but too aggressively
    pos_rm_mask = (~kl_mask) & (ratio > 1.0) & (adv > 0)
    neg_rm_mask = (~kl_mask) & (ratio < 1.0) & (adv < 0)
    rm_mask = pos_rm_mask | neg_rm_mask
    keep_adv_mask = (~rm_mask).detach()

    # Use torch.where for numerical safety: avoids inf * 0 = nan when ratio overflows
    zero = torch.zeros((), dtype=unclipped_loss.dtype, device=unclipped_loss.device)
    loss_per_elem = torch.where(keep_adv_mask, unclipped_loss, zero)

    # Metrics for logging
    if ratio.numel() > 1:
        ratio_std = ratio.std()
    else:
        ratio_std = torch.zeros((), dtype=ratio.dtype, device=ratio.device)
    # Mask breakdown:
    # - kl_mask_fraction: fraction of elements where KL >= threshold (high divergence)
    # - pos_rm_fraction: fraction masked by positive-direction conflict
    # - neg_rm_fraction: fraction masked by negative-direction conflict
    # - masked_fraction: total fraction of elements zeroed out (the key metric)
    # - unmasked_fraction: fraction of elements that contribute to gradient
    metrics = {
        "ratio_mean": ratio.mean().detach(),
        "ratio_std": ratio_std.detach(),
        "ratio_min": ratio.min().detach(),
        "ratio_max": ratio.max().detach(),
        "approx_kl": (0.5 * log_diff.pow(2)).mean().detach(),
        "kl_new_old_mean": kl_per_sample.mean().detach(),
        "kl_new_old_max": kl_per_sample.max().detach(),
        "kl_mask_fraction": (~kl_mask).float().mean().detach(),
        "pos_rm_fraction": pos_rm_mask.float().mean().detach(),
        "neg_rm_fraction": neg_rm_mask.float().mean().detach(),
        "masked_fraction": rm_mask.float().mean().detach(),
        "unmasked_fraction": keep_adv_mask.float().mean().detach(),
    }
    return loss_per_elem, metrics


# ---------------------------------------------------------------------------
# Algorithm class
# ---------------------------------------------------------------------------


class FlowDPPO(StageAlgorithm):
    """FlowDPPO: KL-divergence-based masking for diffusion RL.

    Replaces PPO's ratio clipping (``FlowGRPO``) with a KL-ADV masking
    criterion:

    1. Computes KL(current || old) from ``prev_sample_means`` (the Gaussian
       mean of the SDE transition) at each replayed step.
    2. Creates a two-stage mask:
       - **KL mask**: updates with KL < ``kl_mask_threshold`` pass freely.
       - **ADV mask**: among high-KL updates, masks out those whose ratio
         direction is aligned with advantage (overly aggressive moves).
    3. Loss = ``(-advantage * ratio) * keep_mask``

    This allows aggressive policy updates when KL is small (unlike PPO which
    clips uniformly), and only constrains updates that both diverge far from
    the old policy AND push too aggressively in the reward-improving direction.

    Args:
        stage: The :class:`DiffusionStage` whose ``replay`` produces new
            log-probs and prev_sample_means.
        params: Per-call params (e.g. ``SD3DiffusionParams``).
        kl_mask_threshold: KL divergence threshold for masking. Updates
            with per-sample KL below this pass without masking.
        add_kl_coefficient: If True, normalize KL by
            ``sigma_t = std_dev_t * sqrt(-dt)`` (flow-matching noise scale).
            If False, use unnormalized squared error.
        old_logp_source: ``"rollout"`` (default) trusts the rollout engine's
            emitted ``segment.sde_logp``; ``"replay"`` uses the replayed
            log-probs. ``sde_means`` is always replayed regardless. See
            :meth:`prepare_segment`.
        conditions_cls: Stage-typed conditions container.
    """

    # prepare_segment freezes segment.sde_logp + sde_means once, so the ratio
    # and KL anchor stay fixed across every num_updates_per_batch optimizer step.
    supports_multi_update = True
    anchor_fields = ("sde_logp", "sde_means")

    def recomputes_anchor(self) -> bool:
        # FlowDPPO always replays sde_means for the KL term (regardless of
        # old_logp_source), so the anchor always needs train-time geometry.
        return True

    def __init__(
        self,
        *,
        params: Any,
        stage: Any = None,
        pipeline: Any = None,
        stage_attr: str = "diffusion",
        kl_mask_threshold: float = 1e-5,
        add_kl_coefficient: bool = True,
        old_logp_source: str = "rollout",
        conditions_cls: Optional[Type[Any]] = None,
    ) -> None:
        # v1 (track_builder) passes `stage`; v2 (DiffusionTrainer) passes the
        # `pipeline` sibling and the stage is resolved off it (mirrors FlowGRPO).
        if stage is None and pipeline is not None:
            stage = getattr(pipeline, stage_attr)
        if stage is None:
            raise ValueError("FlowDPPO: either `stage` or `pipeline` must be provided")
        self.stage = stage
        self.params = params
        self.kl_mask_threshold = float(kl_mask_threshold)
        self.add_kl_coefficient = bool(add_kl_coefficient)
        self.old_logp_source = str(old_logp_source).strip().lower()
        require(
            self.old_logp_source in ("rollout", "replay"),
            f"FlowDPPO: old_logp_source must be 'rollout' or 'replay'; got {old_logp_source!r}",
        )
        self.conditions_cls = conditions_cls

    def prepare_segment(
        self,
        *,
        conditions: Mapping[str, "Condition"],
        segment: "LatentSegment",
    ) -> None:
        """Establish the frozen π_old anchor (``segment.sde_logp``) and means
        (``segment.sde_means``) at pre-update weights, before the
        ``num_updates_per_batch`` loop.

        ``stage.replay`` always runs under ``torch.no_grad`` — FlowDPPO needs the
        old policy's ``prev_sample_means`` for the KL term — so ``sde_means``
        is always written from this pre-update replay. The log-prob anchor is
        chosen by ``old_logp_source``:

        - ``"rollout"`` (default): keep the rollout engine's emitted
          ``sde_logp``; raises if it is ``None`` (pin an emitting rollout
          build, or set ``old_logp_source='replay'``).
        - ``"replay"``: use the replayed log-probs, overwriting any engine value.

        No-op if the segment has no SDE-gated steps to train on.
        """
        if segment.sde_indices is None:
            return
        target_steps = self._resolve_target_steps(segment)
        if not target_steps:
            return
        if self.old_logp_source == "rollout" and segment.sde_logp is None:
            raise RuntimeError(
                "FlowDPPO.prepare_segment: old_logp_source='rollout' but the "
                "rollout engine emitted no per-step log-probs (segment.sde_logp is "
                "None). Pin a rollout build that emits trajectory log-probs, or set "
                "old_logp_source='replay'."
            )
        typed_conds = typed_conditions(conditions, self.conditions_cls)
        with torch.no_grad():
            result = self.stage.replay(typed_conds, segment=segment, params=self.params, step_indices=target_steps)
        # Log-prob anchor: replay overwrites; rollout keeps the engine's emission.
        if self.old_logp_source == "replay":
            segment.sde_logp = result.log_probs.detach().cpu()
        # Always populate old means (core of FlowDPPO)
        if result.prev_sample_means is None:
            raise RuntimeError(
                "FlowDPPO.prepare_segment: stage.replay() returned "
                "prev_sample_means=None. Ensure the stage's replay method "
                "produces means (required for KL-ADV masking)."
            )
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
        target_steps = self._resolve_target_steps(segment)
        if not target_steps:
            return AlgorithmStepResult(loss=0.0, metrics={}, num_steps_or_tokens=0, has_backward=False)

        typed_conds = typed_conditions(conditions, self.conditions_cls)
        replay_result = self.stage.replay(
            typed_conds,
            segment=segment,
            params=self.params,
            step_indices=target_steps,
        )
        new_logp = replay_result.log_probs  # [B, S']
        new_means = replay_result.prev_sample_means  # [B, S', C, H, W]

        if new_means is None:
            raise RuntimeError(
                "FlowDPPO requires stage.replay() to return prev_sample_means, "
                "but got None. Ensure the stage's replay method produces means."
            )

        old_logp = gather_sde_field(segment.sde_logp, segment.sde_indices, target_steps, field_name="sde_logp").to(
            dtype=new_logp.dtype, device=new_logp.device
        )
        old_means = gather_sde_field(segment.sde_means, segment.sde_indices, target_steps, field_name="sde_means").to(
            dtype=new_means.dtype, device=new_means.device
        )

        # Compute sigma_t for KL normalization
        sigma_t = self._compute_sigma_t(segment, target_steps, device=new_logp.device)

        adv_b = advantages.detach().to(dtype=new_logp.dtype, device=new_logp.device).reshape(-1, 1).expand_as(new_logp)

        loss_per_elem, ratio_metrics = _flowdppo_kl_adv_loss(
            new_logp=new_logp,
            old_logp=old_logp,
            new_means=new_means,
            old_means=old_means,
            advantages=adv_b,
            sigma_t=sigma_t,
            kl_mask_threshold=self.kl_mask_threshold,
        )

        loss = loss_per_elem.mean()
        (loss * loss_scale).backward()

        metrics: Dict[str, Any] = {
            "policy_loss": float(loss.detach().item()),
            "kl_mask_threshold": float(self.kl_mask_threshold),
            **{k: float(v.item()) for k, v in ratio_metrics.items()},
        }
        return AlgorithmStepResult(
            loss=float(loss.detach().item()),
            metrics=metrics,
            num_steps_or_tokens=len(target_steps),
            has_backward=True,
        )

    # -- helpers --------------------------------------------------------

    def _resolve_target_steps(self, segment: "LatentSegment") -> List[int]:
        """All SDE-recorded step indices on the segment."""
        if segment.sde_indices is None:
            return []
        return [int(i) for i in segment.sde_indices.tolist()]

    def _compute_sigma_t(
        self,
        segment: "LatentSegment",
        target_steps: List[int],
        device: torch.device,
    ) -> torch.Tensor:
        """Compute sigma_t = std_dev_t * sqrt(-dt) for KL normalization.

        When ``add_kl_coefficient=False``, returns ones (unnormalized MSE).
        Returns shape ``[1, S', 1, 1, 1]`` for broadcasting with means tensors.
        """
        if not self.add_kl_coefficient:
            return torch.ones(1, len(target_steps), 1, 1, 1, device=device)

        if segment.sigmas is None:
            raise ValueError("FlowDPPO with add_kl_coefficient=True requires segment.sigmas.")
        sigmas = segment.sigmas.to(device=device, dtype=torch.float32)
        eta = float(self.params.eta)

        # Vectorized: index into sigmas with target_steps tensor
        idx = torch.tensor(target_steps, dtype=torch.long, device=device)
        s = sigmas[idx]
        s_next = sigmas[idx + 1]
        dt = s_next - s  # negative for denoising
        # Match FlowSDEStrategy.step at sigma==1: denominator uses sigma_max=sigmas[1],
        # not a 0.99 clamp, so sigma_t equals the transition's std_var.
        sigma_max = sigmas[1] if int(sigmas.shape[0]) > 1 else torch.tensor(0.99, device=device, dtype=sigmas.dtype)
        std_dev_t = torch.sqrt(s / (1.0 - torch.where(s == 1.0, sigma_max, s))) * eta
        sigma_t = std_dev_t * torch.sqrt(-dt)
        return sigma_t.reshape(1, -1, 1, 1, 1)


__all__ = ["FlowDPPO", "FlowDPPOConfig"]
