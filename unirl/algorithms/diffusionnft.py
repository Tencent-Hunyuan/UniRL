"""Stage-driven DiffusionNFT (Negative Fine-Tuning): forward-process diffusion RL.

DiffusionNFT trains the policy across the diffusion noise spectrum without
running an SDE rollout: each micro-step iterates over a set of
timesteps :math:`\\{t_1, \\dots, t_K\\}` and, at every :math:`t_k`,

1. constructs ``xt = (1 - t_k) * x0 + t_k * noise`` (flow-matching
   forward diffusion of the rollout's clean final latent);
2. asks the trainable adapter (``"default"``) for a noise prediction
   :math:`new_pred` and the EMA-tracked frozen adapter (``"old"``) for
   a reference prediction :math:`old_pred`;
3. blends them into a dual positive / negative pair, reconstructs
   :math:`x_0` from each, and weights the two MSE terms by a
   reward-derived scalar :math:`r \\in [0, 1]`.

Sweeping K timesteps per micro-step (with the gradient scaled by
``loss_scale / K``) ensures the policy is updated across every noise
level the rollout actually visited; a single random ``t`` would leave
each rollout's update concentrated on one slice of the schedule.

Timestep set source:

* ``train_timestep_mode='all'`` — read directly from ``segment.sigmas``
  (the rollout's sampled schedule), drop the terminal zero (training
  on ``t=0`` collapses ``xt`` to ``x0`` and yields no signal), then
  apply ``training_timestep_fraction`` as a slice;
* ``train_timestep_mode='random'`` — synthesize ``B`` random scalars
  in ``(0, training_timestep_fraction]``.

In both modes each iteration broadcasts one scalar ``t`` to the whole
batch.

The dual-adapter mechanics (install / EMA / switch) live in the EMA
policy owned by the FSDP backend. The algorithm receives the policy
via constructor injection (``nft_lora_policy=...``, resolved off
``backend.ema`` by the v2 trainer) and calls its ``with_old_adapter()``
context manager to obtain :math:`old_pred`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple, Type

import torch

from unirl.types.conditions import Condition
from unirl.types.segments.latent import LatentSegment
from unirl.utils.misc import aggregate_numeric_metrics
from unirl.utils.scheduler_utils import normalize_timestep_fraction

from .base import AlgorithmStepResult, BaseAlgorithmConfig, StageAlgorithm


@dataclass
class DiffusionNFTConfig(BaseAlgorithmConfig):
    """Per-call DiffusionNFT loss hyperparameters.

    Only the configuration surface exercised by current recipes is
    accepted; unsupported values fail fast in :meth:`DiffusionNFT.__init__`
    rather than silently behaving differently.
    """

    # Dual-prediction blend coefficient. ``positive = beta*new + (1-beta)*old``;
    # ``negative = (1+beta)*old - beta*new``.
    beta: float = 1.0
    # Clip the rollout advantages to ``[-adv_clip_max, +adv_clip_max]``
    # before the linear remap into ``r in [0, 1]``.
    adv_clip_max: float = 5.0
    # Only ``"raw"`` is wired. Other modes (sign / binary / one_only /
    # ranked / per-timestep) can be reintroduced if a recipe needs them.
    adv_mode: str = "raw"
    # When True: divide each per-sample MSE by its mean-abs-error to
    # equalize scales across timesteps (matches the original DiffusionNFT recipe).
    use_adaptive_weight: bool = True
    # ``"all"`` reads timesteps from ``segment.sigmas``; ``"random"`` draws
    # ``B`` fresh uniforms per micro-step. Both then run the K-iteration
    # loop (K = len of resolved timesteps).
    train_timestep_mode: str = "all"
    shuffle_train_timesteps: bool = True
    # Reserved for Sigma-Schedule shift terms; not implemented.
    apply_time_shift_in_loss: bool = False
    # Slice of the resolved timestep set kept after dropping terminal zero,
    # expressed as a fraction of the schedule length.
    training_timestep_fraction: float = 0.99
    # KL penalty against the un-adapted base model. Not implemented in
    # this revision; ``> 0`` raises so recipes can't silently drop the term.
    kl_coef: float = 0.0


class DiffusionNFT(StageAlgorithm):
    """Forward-process DiffusionNFT over a diffusion ``LatentSegment``.

    DiffusionNFT is off-policy: the rollout uses EMA-smoothed weights to produce
    high-quality trajectories, and the dual-adapter loss trains against
    them (``requires_ema_rollout = True``).

    Args:
        stage: A :class:`DiffusionStage` exposing
            :meth:`predict_noise_at_step(conditions, *, sample, sigma, params)`.
            All DiffusionNFT-supported recipes today target SD3 (``SD3DiffusionStage``);
            the API works model-agnostically across the six NEW stages.
        params: Per-call params object the stage's predictor consumes
            (e.g. ``SD3DiffusionParams``). Held as algorithm state so the
            dispatcher doesn't need to know it. Read fields:
            ``guidance_scale`` (always), and any model-specific extras
            that ``predict_noise_at_step`` forwards.
        config: :class:`DiffusionNFTConfig` — loss hyperparameters.
        nft_lora_policy: The :class:`NFTLoRAPolicy` instance owning the
            ``default`` / ``old`` adapter pair. Injected by
            ``train_actor`` at algorithm-construction time (the
            algorithm cannot walk to it from ``stage`` alone — the chain
            goes outward from stage, not inward to policies).
        conditions_cls: Optional stage-typed conditions container with
            a ``from_dict(Mapping[str, Condition])`` classmethod.
    """

    requires_ema_rollout: bool = True

    def __init__(
        self,
        *,
        params: Any,
        stage: Any = None,
        pipeline: Any = None,
        stage_attr: str = "diffusion",
        nft_lora_policy: Any = None,
        backend: Any = None,
        beta: float = 1.0,
        adv_clip_max: float = 5.0,
        adv_mode: str = "raw",
        use_adaptive_weight: bool = True,
        train_timestep_mode: str = "all",
        shuffle_train_timesteps: bool = True,
        apply_time_shift_in_loss: bool = False,
        training_timestep_fraction: float = 0.99,
        kl_coef: float = 0.0,
        conditions_cls: Optional[Type[Any]] = None,
    ) -> None:
        # Flat-kwarg constructor: each field matches a ``DiffusionNFTConfig``
        # attribute. The dataclass exists for typing / documentation; the
        # runtime accepts the fields directly so YAML recipes don't need
        # to nest a ``config:`` block.
        #
        # Two wiring paths converge here:
        #   - v1 (track_builder) passes ``stage`` + ``nft_lora_policy=<EMA>``.
        #   - v2 (DiffusionTrainer) passes ``pipeline`` + ``backend`` siblings;
        #     resolve ``stage`` off the pipeline (mirrors FlowGRPO) and the
        #     EMA off ``backend.ema`` (the FSDPBackend owns the dual-adapter EMA).
        if stage is None and pipeline is not None:
            stage = getattr(pipeline, stage_attr)
        if stage is None:
            raise ValueError("DiffusionNFT: either `stage` or `pipeline` must be provided")
        if nft_lora_policy is None and backend is not None:
            nft_lora_policy = getattr(backend, "ema", None)
        if adv_mode != "raw":
            raise ValueError(f"DiffusionNFT: adv_mode={adv_mode!r} not supported (only 'raw' is wired).")
        if train_timestep_mode not in ("all", "random"):
            raise ValueError(
                f"DiffusionNFT: train_timestep_mode={train_timestep_mode!r} not supported (use 'all' or 'random')."
            )
        if apply_time_shift_in_loss:
            raise ValueError("DiffusionNFT: apply_time_shift_in_loss=True is not implemented.")
        if not (0.0 < float(training_timestep_fraction) <= 1.0):
            raise ValueError(
                f"DiffusionNFT: training_timestep_fraction must lie in (0, 1]; got {training_timestep_fraction!r}."
            )
        if float(kl_coef) > 0:
            raise ValueError(
                "DiffusionNFT: kl_coef > 0 not supported (KL penalty against base "
                "model not implemented in this revision)."
            )
        if not (0.0 < float(beta)):
            raise ValueError(f"DiffusionNFT: beta must be > 0; got {beta!r}.")
        if not (0.0 < float(adv_clip_max)):
            raise ValueError(f"DiffusionNFT: adv_clip_max must be > 0; got {adv_clip_max!r}.")

        if not callable(getattr(nft_lora_policy, "use_shadow", None)):
            raise TypeError(
                f"DiffusionNFT: nft_lora_policy={type(nft_lora_policy).__name__} "
                f"is missing required method 'use_shadow'; expected an "
                f"EMA handle (or compatible)."
            )

        self.stage = stage
        self.params = params
        self.nft_lora_policy = nft_lora_policy
        self.conditions_cls = conditions_cls
        self.config = DiffusionNFTConfig(
            beta=float(beta),
            adv_clip_max=float(adv_clip_max),
            adv_mode=str(adv_mode),
            use_adaptive_weight=bool(use_adaptive_weight),
            train_timestep_mode=str(train_timestep_mode),
            shuffle_train_timesteps=bool(shuffle_train_timesteps),
            apply_time_shift_in_loss=bool(apply_time_shift_in_loss),
            training_timestep_fraction=float(training_timestep_fraction),
            kl_coef=float(kl_coef),
        )

    # ------------------------------------------------------------------
    # StageAlgorithm contract
    # ------------------------------------------------------------------

    def compute_loss_and_backward(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: LatentSegment,
        advantages: torch.Tensor,
        training_progress: float,
        loss_scale: float,
    ) -> AlgorithmStepResult:
        if segment.latents is None:
            raise ValueError(
                "DiffusionNFT requires segment.latents (clean final latent at "
                "the last trajectory position); got None. Forward-process "
                "rollout must populate the dense latents path."
            )
        x0 = segment.latents[:, -1]  # [B, ...latent_shape]
        if x0.numel() == 0:
            return AlgorithmStepResult(
                loss=0.0,
                metrics={},
                num_steps_or_tokens=0,
                has_backward=False,
            )
        B = int(x0.shape[0])
        device = x0.device
        compute_dtype = self._compute_dtype(x0)

        if int(advantages.shape[0]) != B:
            raise ValueError(
                f"DiffusionNFT: advantages batch size ({int(advantages.shape[0])}) "
                f"does not match clean-latents batch size ({B})."
            )

        # K timesteps from the configured source (see ``_resolve_timesteps``).
        timesteps = self._resolve_timesteps(segment, B, device, compute_dtype)
        K = int(timesteps.numel())
        if K == 0:
            return AlgorithmStepResult(
                loss=0.0,
                metrics={},
                num_steps_or_tokens=0,
                has_backward=False,
            )

        typed_conds = _typed_conditions(conditions, self.conditions_cls)
        # Reward → r in [0, 1]. Independent of t, so computed once.
        adv = advantages.detach().to(dtype=compute_dtype, device=device)
        adv_clipped = torch.clamp(adv, -self.config.adv_clip_max, self.config.adv_clip_max)
        r = (adv_clipped / self.config.adv_clip_max) / 2.0 + 0.5
        r = torch.clamp(r, 0.0, 1.0)

        # Each iteration runs one (forward, backward) at a scalar ``t``
        # broadcast to the whole batch; the gradient is scaled by 1/K
        # so the K iterations cumulatively produce one optimizer-step's
        # worth of signal.
        per_iter_metrics: List[Dict[str, float]] = []
        total_loss = 0.0
        has_backward = False
        iter_scale = float(loss_scale) / float(K)

        for k in range(K):
            t_scalar = timesteps[k]
            loss_k, metrics_k = self._compute_loss_at_t(
                conditions=typed_conds,
                x0=x0,
                t_scalar=t_scalar,
                r=r,
                adv=adv,
                B=B,
                compute_dtype=compute_dtype,
            )
            (loss_k * iter_scale).backward()
            total_loss += float(loss_k.detach().item())
            per_iter_metrics.append(metrics_k)
            has_backward = True

        agg = aggregate_numeric_metrics(per_iter_metrics)
        agg["num_timesteps"] = float(K)
        agg["loss_per_iter"] = float(total_loss / K)
        agg["total_loss"] = float(total_loss)
        if not math.isfinite(agg["total_loss"]):
            raise RuntimeError(
                f"DiffusionNFT: non-finite total_loss={agg['total_loss']!r}. Per-iter metrics: {per_iter_metrics}"
            )

        return AlgorithmStepResult(
            loss=float(total_loss),
            metrics=agg,
            num_steps_or_tokens=K,
            has_backward=has_backward,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_loss_at_t(
        self,
        *,
        conditions: Any,
        x0: torch.Tensor,
        t_scalar: torch.Tensor,
        r: torch.Tensor,
        adv: torch.Tensor,
        B: int,
        compute_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Single-timestep DiffusionNFT loss. ``t_scalar`` is a 0-dim tensor that
        is broadcast to the whole batch — every sample sees the same
        noise level in this iteration of the outer K-loop.
        """
        device = x0.device
        t_batch = t_scalar.detach().to(device=device, dtype=compute_dtype).expand(B)
        t_exp = t_batch.view(B, *([1] * (x0.ndim - 1)))

        noise = torch.randn_like(x0)
        xt = (1.0 - t_exp) * x0 + t_exp * noise

        # Trainable adapter forward; gradient flows back into "default".
        new_pred = self.stage.predict_noise_at_step(
            conditions,
            sample=xt,
            sigma=t_batch,
            params=self.params,
        )
        # Reference adapter forward, detached. ``with_old_adapter``
        # temporarily activates the EMA-tracked adapter; the surrounding
        # ``no_grad`` keeps autograd off this branch.
        with torch.no_grad(), self.nft_lora_policy.use_shadow():
            old_pred = self.stage.predict_noise_at_step(
                conditions,
                sample=xt,
                sigma=t_batch,
                params=self.params,
            )
        old_pred = old_pred.detach()

        beta = float(self.config.beta)
        positive_pred = beta * new_pred + (1.0 - beta) * old_pred
        negative_pred = (1.0 + beta) * old_pred - beta * new_pred
        x0_pos = xt - t_exp * positive_pred
        x0_neg = xt - t_exp * negative_pred

        reduce_dims = tuple(range(1, x0.ndim))
        x0_for_mse = x0.to(dtype=new_pred.dtype)
        if self.config.use_adaptive_weight:
            # Per-sample mean-abs-error in float64 (low-precision can
            # underflow when ``xt`` is close to ``x0``). The clamp floor
            # prevents division by zero when prediction matches target
            # exactly.
            with torch.no_grad():
                weight_pos = (
                    (x0_pos.detach().double() - x0_for_mse.double())
                    .abs()
                    .mean(dim=reduce_dims, keepdim=True)
                    .clamp(min=1e-5)
                ).to(dtype=new_pred.dtype)
                weight_neg = (
                    (x0_neg.detach().double() - x0_for_mse.double())
                    .abs()
                    .mean(dim=reduce_dims, keepdim=True)
                    .clamp(min=1e-5)
                ).to(dtype=new_pred.dtype)
            pos_loss = ((x0_pos - x0_for_mse) ** 2 / weight_pos).mean(dim=reduce_dims)
            neg_loss = ((x0_neg - x0_for_mse) ** 2 / weight_neg).mean(dim=reduce_dims)
        else:
            pos_loss = ((x0_pos - x0_for_mse) ** 2).mean(dim=reduce_dims)
            neg_loss = ((x0_neg - x0_for_mse) ** 2).mean(dim=reduce_dims)

        # ``adv_clip_max`` factors back in so the gradient magnitude is
        # invariant to the choice of clip range (the ``r`` remap divides
        # by it; multiplying outside restores the original scale).
        policy_loss = (r * pos_loss / beta + (1.0 - r) * neg_loss / beta).mean()
        total = policy_loss * float(self.config.adv_clip_max)

        metrics = {
            "policy_loss": float(policy_loss.detach().item()),
            "pos_loss_mean": float(pos_loss.mean().detach().item()),
            "neg_loss_mean": float(neg_loss.mean().detach().item()),
            "r_mean": float(r.mean().detach().item()),
            "advantage_mean": float(adv.mean().detach().item()),
            "advantage_std": float(adv.std().detach().item()) if B > 1 else 0.0,
            "prediction_deviation": float(((new_pred - old_pred) ** 2).mean().detach().item()),
            "x0_norm": float((x0**2).mean().detach().item()),
            "t_value": float(t_scalar.detach().item()),
        }
        return total, metrics

    def _resolve_timesteps(
        self,
        segment: LatentSegment,
        B: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Resolve the K scalar timesteps for the outer K-iteration loop.

        ``mode='all'`` reads the rollout's actual sampled schedule from
        ``segment.sigmas`` (dropping the terminal zero, then applying
        ``training_timestep_fraction`` as a slice). ``mode='random'``
        draws ``B`` fresh uniforms in ``(0, training_timestep_fraction]``.
        Either output is optionally shuffled before the caller iterates.
        """
        mode = self.config.train_timestep_mode
        frac = float(self.config.training_timestep_fraction)
        if mode == "all":
            if segment.sigmas is None:
                raise ValueError(
                    "DiffusionNFT(train_timestep_mode='all') requires "
                    "segment.sigmas; the rollout did not capture a schedule. "
                    "Set train_timestep_mode='random' instead."
                )
            ts = segment.sigmas.detach().to(device=device, dtype=dtype).flatten()
            # ``sigma == 0`` collapses ``xt`` to ``x0`` (no noise) and
            # yields no gradient, so the terminal entry is excluded.
            if (
                ts.numel() > 1
                and torch.isclose(
                    ts[-1],
                    torch.zeros((), device=device, dtype=dtype),
                    atol=1e-8,
                ).item()
            ):
                ts = ts[:-1]
            if ts.numel() > 0 and frac != 1.0:
                start, end = normalize_timestep_fraction(frac)
                n = int(ts.numel())
                eff_start = int(n * start)
                eff_end = min(int(n * end), n)
                ts = ts[eff_start:eff_end] if eff_start < eff_end else ts[:0]
            if ts.numel() == 0:
                # The slice can be empty when the schedule is short and
                # the fraction tight; fall back to a single random ``t``
                # so the train step still produces a gradient.
                ts = torch.rand(1, device=device, dtype=dtype) * frac
        elif mode == "random":
            ts = torch.rand(B, device=device, dtype=dtype) * frac
        else:
            raise ValueError(f"DiffusionNFT: unsupported train_timestep_mode={mode!r}")

        if bool(self.config.shuffle_train_timesteps):
            perm = torch.randperm(int(ts.numel()), device=device)
            ts = ts[perm]
        return ts

    @staticmethod
    def _compute_dtype(x0: torch.Tensor) -> torch.dtype:
        """fp32 for the timestep tensor — the forward-diffusion arithmetic
        loses too much precision in bf16 when ``t`` is close to either
        endpoint.
        """
        del x0
        return torch.float32


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _typed_conditions(
    conditions: Mapping[str, Condition],
    conditions_cls: Optional[Type[Any]],
) -> Any:
    """Wrap the conditions dict into the stage's typed container, or pass
    through unchanged if no container class is given (e.g. unit tests).
    """
    if conditions_cls is None:
        return conditions
    return conditions_cls.from_dict(dict(conditions))


__all__ = ["DiffusionNFT", "DiffusionNFTConfig"]
