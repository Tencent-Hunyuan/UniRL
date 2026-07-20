"""Supervised finetuning losses — the anchor-free algorithms the base class anticipates.

Two :class:`StageAlgorithm` siblings of GRPO / FlowGRPO / DiffusionNFT, sharing
their stage surface so RL and SFT run the SAME model forwards (no parity drift):

- :class:`SFT` — autoregressive masked cross-entropy over a dataset-built
  ``TextSegment`` via ``ARStage.replay`` (the teacher-forced per-token-logp
  kernel GRPO already uses; structural prompt masking, chunked lm_head,
  packed-varlen — all inherited). CE is just ``-logp`` at ``temperature=1.0``.
- :class:`FlowMatchSFT` — diffusion flow-matching velocity MSE over a
  dataset-built x0-only ``LatentSegment`` via ``predict_noise_at_step`` (the
  single-``(x_t, σ)`` forward DiffusionNFT already uses). The forward-noising
  recipe ``x_t = (1-σ)·x0 + σ·ε`` and the velocity convention
  ``target = ε - x0`` (⇔ ``x0_hat = x_t - σ·pred``) match DiffusionNFT and
  ``FlowSDEStrategy`` exactly.

Neither reads ``advantages`` / ``old_logp`` / SDE trajectories — both declare
``requires_advantages = False`` and leave :meth:`prepare_segment` as the no-op
the base docstring promises for anchor-free algorithms. Both are driven by the
regular :class:`~unirl.train.stack.TrainStack`; tracks come from a
``SupervisedTrackBuilder`` (``unirl/train/sft/track_builder.py``) instead of a
rollout engine.

Loss-normalization contract (the cross-framework lesson): token-level CE must
be normalized by the GLOBAL valid-token count of one optimizer step — across
micro-batches and DP ranks — or gradients silently depend on packing and DP
layout. :class:`SFT` therefore declares ``loss_weighting = "token"`` under
``loss_agg_mode="token-mean"`` and returns a micro-token-MEAN loss; the stack
supplies ``loss_scale = micro_tokens · dp_world / global_tokens`` so
``(loss · loss_scale).backward()`` accumulates the exact full-batch token-mean
gradient. Seq-mean agg modes weigh each sequence equally instead
(``loss_weighting = "sample"``; equal DP shards make rank-mean exact).
"""

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
    """Masked next-token cross-entropy over an AR ``TextSegment``.

    ``segment.tokens`` are the GROUND-TRUTH response tokens (dataset targets,
    EOS included — the same layout rollout engines emit, so ``ARStage.replay``
    is reused verbatim); ``segment.loss_mask`` optionally down-weights tokens
    (1 = train, 0 = ignore). ``segment.log_probs`` may be ``None`` — SFT has no
    behavior policy.

    Args:
        stage / pipeline / stage_attr: standard stage resolution (the trainer
            injects ``pipeline``; the stage is ``getattr(pipeline, stage_attr)``).
        loss_agg_mode: ``"token-mean"`` (default — global token-mean via the
            stack's ``loss_weighting='token'`` contract), ``"seq-mean-token-mean"``
            (each sequence weighs equally), or ``"seq-mean-token-sum-norm"``
            (Dr.GRPO-style per-seq sum / ``horizon``). The seq-mean modes are
            what :class:`~unirl.train.stack.TokenBudgetPlanner` packing requires.
        horizon: normalizer for ``seq-mean-token-sum-norm``.
        conditions_cls: stage-typed conditions container with ``from_dict``.

    Deliberately NOT exposed: ``temperature`` (pinned 1.0 — the ``logits/T``
    rescale exists only to match a sampling engine's distribution; inheriting a
    rollout temperature would silently rescale CE), clip ranges, old-logp
    sources, schedules — those are rollout-anchoring semantics with no meaning
    under supervision.
    """

    supports_multi_update = True  # anchor-free: each disjoint update is plain SGD on its slice
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
        # token-mean pairs with the stack's global-token weighting; the seq-mean
        # modes weigh sequences equally, i.e. sample-share weighting.
        self.loss_weighting = "token" if self.loss_agg_mode == "token-mean" else "sample"

    # ------------------------------------------------------------------
    # StageAlgorithm contract
    # ------------------------------------------------------------------

    def compute_loss_and_backward(
        self,
        *,
        conditions: Mapping[str, Condition],
        segment: "TextSegment",
        advantages: Optional[torch.Tensor],
        training_progress: float,
        loss_scale: float,
    ) -> AlgorithmStepResult:
        del advantages, training_progress  # supervised: no advantage signal, no schedules
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
        """Forward-only ``(objective_sum, objective_weight)`` for validation.

        ``token-mean`` returns raw CE and valid-token count. Sequence-mean modes
        return the sum of their per-sequence objectives and the number of valid
        sequences. The caller can therefore aggregate the exact training
        objective across micros, DP ranks, and eval batches. ``sample_ids`` is
        unused here (CE is deterministic) — accepted for a uniform signature.
        """
        del sample_ids
        if segment is None or segment.tokens is None or segment.tokens.shape[0] == 0:
            return 0.0, 0.0
        _, aux = self._masked_ce(conditions, segment)
        return aux["objective_sum"], aux["objective_weight"]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _masked_ce(
        self,
        conditions: Mapping[str, Condition],
        segment: "TextSegment",
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """One teacher-forced forward → (aggregated loss, reduction stats).

        ``temperature=1.0`` — plain CE, never a sampling-matched rescale.
        """
        typed_conds = typed_conditions(conditions, self.conditions_cls)
        new_logp = self.stage.replay(typed_conds, segment=segment, temperature=1.0)  # [total_tokens] fp32
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
            else:  # seq-mean-token-mean
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
    """Flow-matching velocity MSE over a dataset x0-only ``LatentSegment``.

    Per micro-step, for each sample: draw ``σ`` from the configured
    distribution, noise ``x_t = (1-σ)·x0 + σ·ε``, forward once via
    ``stage.predict_noise_at_step`` and regress the predicted velocity onto
    ``ε - x0`` in fp32. ``segment.latents[:, -1]`` is the clean VAE-encoded
    target latent (the same slot DiffusionNFT reads).

    Timestep sampling (the train/inference-schedule parity knob):
        ``timestep_sampling="logit_normal"`` draws ``u = sigmoid(N(logit_mean,
        logit_std))`` (Esser et al. 2024 — the SD3 paper's weighting);
        ``"uniform"`` draws ``u ~ U(0,1)``. Either is then warped through the
        model's inference time-shift ``σ = s·u / (1 + (s-1)·u)`` — set
        ``timestep_shift`` to the family's inference shift (SD3/Bagel 3.0,
        WAN 5.0, Flux 1.0) so training visits the noise band inference actually
        uses; an unshifted draw under-trains it (the kohya/diffusers-issue
        class of silent quality loss).

    Args:
        params: per-call params the stage's predictor consumes (recipes bind
            ``${sampling}``); only ``guidance_scale`` is read here — keep it
            1.0 so SFT runs the pure conditional branch.
        timestep_shift: inference-schedule shift to warp draws through
            (default 1.0 = no warp; set per family).
        eval_seed: seed for the DETERMINISTIC eval draw — validation re-noises
            the same (sample, σ, ε) every time, so the eval loss is comparable
            across steps instead of a fresh MC estimate (the training loss only
            oscillates; a fixed-noise val loss is the signal that shows
            learning).
    """

    supports_multi_update = True  # anchor-free
    requires_advantages = False
    loss_weighting = "sample"  # each image/video sample weighs equally

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

    # ------------------------------------------------------------------
    # StageAlgorithm contract
    # ------------------------------------------------------------------

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
        """Forward-only ``(mse_sum, sample_count)`` at a FIXED (σ, ε) draw.

        The (σ, ε) draw is pinned PER SAMPLE (seeded from ``eval_seed`` + the
        sample's own id), not per batch: a single batch-level seed would tie a
        sample's noising to its position in the batch, so the eval loss would
        shift if the eval batch size, order, or DP sharding changed. Per-sample
        seeding makes the number comparable across steps AND invariant to
        batching (random-σ losses only oscillate). ``segment.loss_mask``
        (``[B]``) excludes padded eval rows.
        """
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

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

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
        # fp32 endpoint math — bf16 loses precision when σ approaches 0 or 1
        # (same rationale as DiffusionNFT's fp32 timestep path).
        return x0.float()

    def _draw_sigma(self, batch: int, device: torch.device, generator: Optional[torch.Generator]) -> torch.Tensor:
        if self.timestep_sampling == "logit_normal":
            z = torch.randn(batch, device=device, dtype=torch.float32, generator=generator)
            u = torch.sigmoid(z * self.logit_std + self.logit_mean)
        else:
            u = torch.rand(batch, device=device, dtype=torch.float32, generator=generator)
        s = self.timestep_shift
        sigma = (s * u) / (1.0 + (s - 1.0) * u)
        return sigma.clamp(min=self.sigma_min, max=1.0 - self.sigma_min)

    def _sample_eval_seed(self, key: str) -> int:
        """Stable int64 seed from ``eval_seed`` + a per-sample key (id).

        ``hashlib`` (not Python's salted ``hash``) so the seed is identical
        across processes / ranks / runs — the whole point of a comparable eval.
        """
        digest = hashlib.sha256(f"{self.eval_seed}:{key}".encode()).digest()
        return int.from_bytes(digest[:8], "little") & 0x7FFF_FFFF_FFFF_FFFF

    def _eval_draws(self, x0: torch.Tensor, sample_ids: Optional[Sequence[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-sample deterministic ``(σ, ε)`` for eval — one seeded generator
        per sample keyed on its id, so a sample's noising is independent of the
        batch it lands in. Falls back to the row index when ids are absent."""
        batch = x0.shape[0]
        device = x0.device
        sigmas: list[torch.Tensor] = []
        noises: list[torch.Tensor] = []
        for i in range(batch):
            key = str(sample_ids[i]) if sample_ids is not None and i < len(sample_ids) else str(i)
            generator = torch.Generator(device=device)
            generator.manual_seed(self._sample_eval_seed(key))
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

        # Single-sample micros pass a 0-dim σ — accepted by every stage
        # (SD3 broadcasts; Bagel's packed forward requires the scalar form).
        sigma_arg = sigma if batch > 1 else sigma.reshape(())
        v_pred = self.stage.predict_noise_at_step(typed_conds, sample=xt, sigma=sigma_arg, params=self.params)
        if v_pred.ndim == x0.ndim - 1:  # unit-batch stages may squeeze the batch dim
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


__all__ = ["SFT", "FlowMatchSFT", "flow_shift_sigma"]
