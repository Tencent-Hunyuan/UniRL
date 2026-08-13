"""Stage-driven algorithm base class."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple, Type

import torch

from unirl.distributed.group.remote import Remote

if TYPE_CHECKING:
    from unirl.types.conditions import Condition
    from unirl.types.sample import Part
    from unirl.types.segments.base import Segment


def typed_conditions(
    conditions: Mapping[str, "Condition"],
    conditions_cls: Optional[Type[Any]],
) -> Any:
    """Reconstruct the stage's typed conditions container from the dict shape."""
    if conditions_cls is None:
        return conditions
    return conditions_cls.from_dict(dict(conditions))


def gather_sde_field(
    tensor: Optional[torch.Tensor],
    sde_indices: Optional[torch.Tensor],
    target_steps: List[int],
    *,
    field_name: str = "field",
) -> torch.Tensor:
    """Gather slices from a segment's SDE-aligned tensor by step index."""
    if tensor is None or sde_indices is None:
        raise ValueError(
            f"gather_sde_field: {field_name} or sde_indices is None "
            f"(ensure prepare_segment ran before compute_loss_and_backward)."
        )
    target_t = torch.tensor(target_steps, dtype=sde_indices.dtype, device=sde_indices.device)
    sort_order = sde_indices.argsort()
    sde_indices = sde_indices[sort_order]
    tensor = tensor[:, sort_order.tolist()]
    positions = torch.searchsorted(sde_indices, target_t)
    positions = positions.clamp(max=sde_indices.shape[0] - 1)
    if (sde_indices[positions] != target_t).any():
        bad = [int(t) for t, p in zip(target_steps, positions) if sde_indices[p] != t]
        raise ValueError(
            f"gather_sde_field({field_name}): target steps {bad} not in sde_indices={sde_indices.tolist()}"
        )
    return tensor[:, positions.tolist()]


def rollout_replay_logp_absdiff(new_logp: torch.Tensor, old_logp: torch.Tensor) -> Dict[str, float]:
    """Per-token |Δlogp| between rollout and replay — AR train-rollout drift gauge."""
    with torch.no_grad():
        absdiff = (new_logp - old_logp).abs()
    return {
        "rollout_replay_logp_absdiff_mean": float(absdiff.mean()),
        "rollout_replay_logp_absdiff_max": float(absdiff.max()),
    }


def rollout_replay_k3(new_logp: torch.Tensor, old_logp: torch.Tensor) -> Dict[str, float]:
    """Per-token K3 KL estimator between rollout and replay log-probs."""
    with torch.no_grad():
        log_r = (new_logp.float() - old_logp.float()).clamp(min=-20.0, max=20.0)
        k3 = torch.expm1(log_r) - log_r
        out = {
            "k3_mean": float(k3.mean()),
            "k3_max": float(k3.max()),
        }
        if k3.numel() >= 2:
            q = torch.quantile(k3.float(), torch.tensor([0.90, 0.99], device=k3.device))
            out["k3_p90"] = float(q[0])
            out["k3_p99"] = float(q[1])
    return out


def _resolve_clip_range_from_schedule(clip_range: float, schedule: str, progress: float) -> float:
    """Schedule-aware clip range. Mirrors ``GRPOAlgorithm.get_clip_range``."""
    if schedule == "linear_decay":
        return clip_range * (1.0 - 0.5 * float(progress))
    if schedule == "cosine_decay":
        return clip_range * (0.5 * (1.0 + math.cos(math.pi * float(progress))))
    return clip_range


def _grpo_clip_loss(
    *,
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    clip_range: float,
    clip_range_high: Optional[float] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """PPO-style clipped objective. Element-wise; reduction is the caller's job."""
    high = clip_range if clip_range_high is None else clip_range_high
    log_diff = new_logp - old_logp
    ratio = torch.exp(log_diff)
    adv = advantages.detach()
    unclipped = -adv * ratio
    clipped = -adv * torch.clamp(ratio, 1.0 - clip_range, 1.0 + high)
    loss_per_elem = torch.maximum(unclipped, clipped)

    if ratio.numel() > 1:
        ratio_std = ratio.std()
    else:
        ratio_std = torch.zeros((), dtype=ratio.dtype, device=ratio.device)
    gt = (ratio - 1.0 > high).float()
    lt = (1.0 - ratio > clip_range).float()
    metrics = {
        "ratio_mean": ratio.mean().detach(),
        "ratio_std": ratio_std.detach(),
        "ratio_min": ratio.min().detach(),
        "ratio_max": ratio.max().detach(),
        "clip_fraction": torch.maximum(gt, lt).mean().detach(),
        "clipfrac_gt_one": gt.mean().detach(),
        "clipfrac_lt_one": lt.mean().detach(),
        "approx_kl": (0.5 * log_diff.pow(2)).mean().detach(),
    }
    return loss_per_elem, metrics


def _gaussian_kl_div(p: torch.Tensor, q: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Per-element Gaussian KL between means at shared variance: ``(p-q)^2 / (2 sigma^2)``."""
    return (p - q) ** 2 / (2 * sigma**2)


def _transition_sigma(
    stage: Any,
    *,
    segment: "Segment",
    target_steps: List[int],
    eta: float,
    device: torch.device,
    add_coefficient: bool = True,
) -> torch.Tensor:
    """Per-step SDE transition std ``sigma_t`` for KL normalization, shape ``[1, S', 1, 1, 1]``."""
    if not add_coefficient:
        return torch.ones(1, len(target_steps), 1, 1, 1, device=device)
    if segment.sigmas is None:
        raise ValueError("_transition_sigma requires segment.sigmas (add_coefficient=True).")
    sigmas = segment.sigmas.to(device=device, dtype=torch.float32)
    idx = torch.tensor(target_steps, dtype=torch.long, device=device)
    s = sigmas[idx]
    s_next = sigmas[idx + 1]
    sigma_max = sigmas[1] if int(sigmas.shape[0]) > 1 else torch.tensor(0.99, device=device, dtype=sigmas.dtype)
    sigma_t = stage.strategy.transition_std(sigma=s, sigma_next=s_next, eta=float(eta), sigma_max=sigma_max)
    return sigma_t.reshape(1, -1, 1, 1, 1)


def _reference_replay_means(
    stage: Any,
    ref_model: Any,
    *,
    conditions: Any,
    segment: "Segment",
    params: Any,
    target_steps: List[int],
) -> torch.Tensor:
    """Replay the reference policy (LoRA adapter disabled) → detached ``prev_sample_means``."""
    from unirl.train.lora import adapters_disabled

    with torch.no_grad(), adapters_disabled(ref_model):
        result = stage.replay(conditions, segment=segment, params=params, step_indices=target_steps)
    if result.prev_sample_means is None:
        raise RuntimeError(
            "_reference_replay_means: stage.replay() returned prev_sample_means=None "
            "for the adapter-disabled reference forward."
        )
    return result.prev_sample_means.detach()


def _reference_kl_loss(
    new_means: torch.Tensor,
    ref_means: torch.Tensor,
    sigma_t: torch.Tensor,
) -> torch.Tensor:
    """Mean Gaussian KL(pi_theta || pi_ref) over ``[B, S']`` per-step means; grad flows through ``new_means`` only."""
    kl_per_elem = _gaussian_kl_div(new_means, ref_means, sigma_t)
    kl_per_sample = kl_per_elem.mean(dim=tuple(range(2, kl_per_elem.ndim)))
    return kl_per_sample.mean()


def _resolve_reference_model(backend: Any, *, beta: float, algo: str) -> Any:
    """Resolve the trainable model for the adapter-disabled reference replay, or None."""
    if float(beta) < 0.0:
        raise ValueError(f"{algo}: beta must be >= 0; got {beta!r}.")
    if float(beta) == 0.0:
        return None
    model = getattr(backend, "model", None) if backend is not None else None
    if model is None:
        raise ValueError(
            f"{algo}: beta>0 needs the trainable model to define the reference policy, but "
            f"no `backend` was injected. The v2 DiffusionTrainer injects it when the "
            f"algorithm declares requires_backend=True."
        )
    if not any("lora_" in name for name, _ in model.named_parameters()):
        raise ValueError(
            f"{algo}: beta>0 computes KL against the LoRA-disabled base model (reference "
            f"policy), which requires a LoRA adapter, but the trainable model has none. Use "
            f"a LoRA recipe, or set beta=0."
        )
    return model


def _require_replay_anchor_for_batched_replay(stage: Any, old_logp_source: str, *, algo: str) -> None:
    """Reject ``batch_replay_steps`` paired with a rollout-sourced π_old anchor."""
    if not getattr(stage, "batch_replay_steps", False):
        return
    if old_logp_source != "replay":
        raise ValueError(
            f"{algo}: pipeline.batch_replay_steps=True requires old_logp_source='replay', "
            f"got {old_logp_source!r}. The batched replay path is numerically equivalent to "
            f"the serial one but not bit-identical to the rollout forward, so a "
            f"rollout-sourced anchor puts the PPO ratio outside clip_range. Set "
            f"old_logp_source='replay', or disable pipeline.batch_replay_steps."
        )


@dataclass(frozen=True)
class AlgorithmStepResult:
    """Result of one micro-step under the stage-driven contract."""

    loss: float
    metrics: Mapping[str, Any]
    num_steps_or_tokens: int
    has_backward: bool


class BaseAlgorithmConfig(ABC):
    """Marker base for all algorithm config dataclasses."""


class StageAlgorithm(Remote, ABC):
    """Pure (conditions, segment, advantages) → loss; holds its stage."""

    requires_ema_rollout: bool = False
    supports_multi_update: bool = False
    requires_backend: bool = False
    requires_advantages: bool = True
    loss_weighting: str = "sample"
    anchor_fields: Tuple[str, ...] = ()

    def recomputes_anchor(self) -> bool:
        """Whether the anchor must be recomputed at the exact ``(mini, micro)`` geometry training uses."""
        return False

    def prepare_segment(
        self,
        *,
        conditions: Mapping[str, "Condition"],
        segment: "Segment",
    ) -> None:
        """Optional pre-step hook called once before the multi-update loop."""
        return None

    def prepare_part(self, part: "Part") -> "Part":
        """Optional post-anchor hook over the complete arranged worker shard."""
        return part

    @abstractmethod
    def compute_loss_and_backward(
        self,
        *,
        conditions: Mapping[str, "Condition"],
        segment: "Segment",
        advantages: Optional[torch.Tensor],
        training_progress: float,
        loss_scale: float,
    ) -> AlgorithmStepResult:
        """Loss for one micro-batch, then ``.backward()``; ``advantages [B]``, ``training_progress`` in ``[0, 1]``."""
        ...


__all__ = [
    "AlgorithmStepResult",
    "StageAlgorithm",
    "gather_sde_field",
    "rollout_replay_logp_absdiff",
    "rollout_replay_k3",
    "typed_conditions",
]
