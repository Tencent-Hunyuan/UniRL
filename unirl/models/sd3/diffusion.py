"""SD3 diffusion: typed params + per-step kernel + rollout-level stage.

Three classes:

- ``SD3DiffusionParams`` — typed request-shape knobs (steps / guidance /
  size / seed / sde_indices / eta / init_same_noise / samples_per_prompt /
  noise_group_ids / max_sequence_length).
- ``SD3DiffusionStep`` — stateless per-step kernel. ``step`` /
  ``step_with_logp`` take the model + conditions + strategy and run both
  CFG noise prediction and the SDE transition (via
  ``StepStrategy.denoise``). ``forward`` is a lower-level helper that
  takes a precomputed ``noise_pred``.
- ``SD3DiffusionStage`` — implements ``DiffusionStage[SD3Conditions]``.
  Owns the SDE ``strategy`` and the loop bookkeeping; delegates the
  per-step model+SDE work to the kernel. Also exposes ``replay`` for
  single-step log-prob replay during training.

CFG math copied from ``samplers/fsdp/sd3_sampler.py:158-207`` (do NOT
import legacy code).
"""

from __future__ import annotations

from typing import Any, ClassVar, List, Optional, Tuple

import torch

from unirl.models.diffusion import (
    ReplayResult,
    SingleStreamDiffusionRunner,
    SingleStreamDiffusionStep,
    SingleStreamLatentSpec,
)
from unirl.sde.kernels import SDEStrategy, StepStrategy
from unirl.types.conditions import TextEmbedCondition
from unirl.types.sampling import DiffusionSamplingParams
from unirl.types.segments.latent import LatentSegment

from .bundle import SD3Bundle
from .conditions import SD3Conditions


class SD3DiffusionStep(SingleStreamDiffusionStep[SD3Bundle, SD3Conditions]):
    """Per-step SD3 denoising kernel — stateless.

    ``step`` / ``step_with_logp`` take the model + conditions + an SDE
    ``strategy`` per call, run CFG noise prediction internally, then
    apply the transition via ``strategy.denoise``. ``forward`` is the
    lower-level escape hatch that takes a precomputed ``noise_pred``.
    """

    def predict_noise(
        self,
        model: SD3Bundle,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        conditions: SD3Conditions,
        *,
        guidance_scale: float,
    ) -> torch.Tensor:
        """Run SD3 transformer with CFG batched ``[uncond, cond]`` forward.

        Reads ``conditions.text.embeds`` / ``.pooled`` for the conditional
        branch. For ``guidance_scale > 1`` reads
        ``conditions.negative_text.embeds`` / ``.pooled`` for the
        unconditional branch; falls back to zero embeddings if
        ``negative_text`` is ``None``.
        """
        if conditions.text is None:
            raise ValueError("SD3DiffusionStep.predict_noise: conditions.text is None")
        text = conditions.text
        if text.embeds is None:
            raise ValueError("SD3DiffusionStep.predict_noise: conditions.text.embeds is None")
        # Pin every model input to the transformer's device. Dedicated-engine
        # (vLLM-Omni) replay hands sample/conditions back on CPU; the trainside
        # engine already has them on GPU (these ``.to`` calls are then no-ops).
        dev = model.device
        sample = sample.to(dev)
        sigma = sigma.to(dev)
        prompt_embeds = text.embeds.to(dev)
        pooled_prompt_embeds = text.pooled.to(dev) if text.pooled is not None else None

        # Cast latent/embeds to the transformer's param dtype before the bf16
        # pos_embed conv — autocast doesn't reliably catch the first conv input
        # under FSDP2 wrap (the DiffusionNFT forward-process path hits this; GRPO/FlowDPPO
        # replay feeds already-bf16 latents). Idempotent when dtype matches.
        try:
            model_dtype = next(model.transformer.parameters()).dtype
        except StopIteration:
            model_dtype = sample.dtype
        sample = sample.to(dtype=model_dtype)
        prompt_embeds = prompt_embeds.to(dtype=model_dtype)
        if pooled_prompt_embeds is not None:
            pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=model_dtype)

        batch_size = sample.shape[0]
        timestep = sigma * 1000.0
        if timestep.dim() == 0:
            timestep = timestep.expand(batch_size)
        elif timestep.shape[0] != batch_size:
            timestep = timestep.expand(batch_size)

        if guidance_scale > 1.0:
            neg = conditions.negative_text
            if neg is not None and neg.embeds is not None:
                negative_prompt_embeds = neg.embeds.to(dev)
                negative_pooled_prompt_embeds = neg.pooled.to(dev) if neg.pooled is not None else None
            else:
                negative_prompt_embeds = torch.zeros_like(prompt_embeds)
                negative_pooled_prompt_embeds = (
                    torch.zeros_like(pooled_prompt_embeds) if pooled_prompt_embeds is not None else None
                )

            if pooled_prompt_embeds is not None:
                pooled_batched = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
            else:
                pooled_batched = None

            noise_pred = model.transformer(
                hidden_states=torch.cat([sample, sample], dim=0),
                encoder_hidden_states=torch.cat([negative_prompt_embeds, prompt_embeds], dim=0),
                timestep=torch.cat([timestep, timestep], dim=0),
                pooled_projections=pooled_batched,
                return_dict=False,
            )[0]
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2, dim=0)
            return noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

        return model.transformer(
            hidden_states=sample,
            encoder_hidden_states=prompt_embeds,
            timestep=timestep,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False,
        )[0]

    # ---- Protocol surface ---------------------------------------------------

    def forward(
        self,
        *,
        strategy: StepStrategy,
        noise_pred: torch.Tensor,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        prev_sample: Optional[torch.Tensor] = None,
        sigma_max: float = 0.99,
        eta: float = 1.0,
        step_index: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run one SDE transition given a precomputed ``noise_pred``.

        Returns ``(prev_sample, log_prob, prev_sample_mean)``.
        ``prev_sample=None`` means sampling mode; otherwise log-prob
        replay. ``log_prob`` and ``prev_sample_mean`` are ``None`` for
        deterministic steps (``eta=0`` or DPM2-style ODE).
        """
        return strategy.denoise(
            noise_pred=noise_pred,
            sample=sample,
            sigma=sigma,
            sigma_next=sigma_next,
            eta=eta,
            prev_sample=prev_sample,
            sigma_max=sigma_max,
            step_index=step_index,
        )

    def step(
        self,
        model: SD3Bundle,
        conditions: SD3Conditions,
        *,
        strategy: StepStrategy,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        guidance_scale: float,
        prev_sample: Optional[torch.Tensor] = None,
        sigma_max: float = 0.99,
        eta: float = 1.0,
        step_index: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run model forward + SDE transition. End-to-end one diffusion step.

        Returns ``(prev_sample, log_prob, prev_sample_mean)``.
        """
        noise_pred = self.predict_noise(model, sample, sigma, conditions, guidance_scale=guidance_scale)
        return self.forward(
            strategy=strategy,
            noise_pred=noise_pred,
            sample=sample,
            sigma=sigma,
            sigma_next=sigma_next,
            prev_sample=prev_sample,
            sigma_max=sigma_max,
            eta=eta,
            step_index=step_index,
        )

    def step_with_logp(
        self,
        model: SD3Bundle,
        conditions: SD3Conditions,
        *,
        strategy: StepStrategy,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        sigma_next: torch.Tensor,
        guidance_scale: float,
        prev_sample: Optional[torch.Tensor] = None,
        sigma_max: float = 0.99,
        eta: float = 1.0,
        step_index: int = 0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run model forward + SDE transition.

        Returns ``(prev_sample, log_prob, prev_sample_mean)``. ``log_prob``
        and ``prev_sample_mean`` are ``None`` for deterministic strategies.
        """
        return self.step(
            model,
            conditions,
            strategy=strategy,
            sample=sample,
            sigma=sigma,
            sigma_next=sigma_next,
            guidance_scale=guidance_scale,
            prev_sample=prev_sample,
            sigma_max=sigma_max,
            eta=eta,
            step_index=step_index,
        )


class SD3DiffusionStage(SingleStreamDiffusionRunner[SD3Bundle, SD3Conditions]):
    """SD3 rollout-level diffusion stage.

    Owns the SDE ``strategy`` (stateful strategies like ``DPM2Strategy``
    require a stable instance across the loop), the bundle, the kernel,
    and the precision policy. The kernel is stateless and is invoked
    per-step with the strategy passed in.

    ``diffuse(conditions, *, schedule, params)`` runs the full sampling
    loop and returns a ``LatentSegment`` carrying the trajectory plus
    per-SDE log probs (``sde_logp [N, S]`` + ``sde_indices [S]``).

    ``replay(conditions, *, segment, params, step_indices=None)``
    recomputes log-probs for the SDE transitions in a stored
    ``LatentSegment``. Returns a :class:`ReplayResult` with ``log_probs``
    of shape ``[B, S']`` aligned with ``segment.sde_logp`` (or a slice
    when ``step_indices`` selects a subset) and ``prev_sample_means``
    for KL-penalty consumption. Used by GRPO-style training.

    ``_no_split_modules`` is the model-side fallback used by FSDPPolicy
    when HF auto-discovery (`type(trainable_root).__mro__._no_split_modules`)
    yields nothing — diffusers' ``SD3Transformer2DModel`` doesn't
    follow the HF transformers convention, so we declare it here.
    """

    _no_split_modules: ClassVar[Tuple[str, ...]] = ("JointTransformerBlock",)

    def __init__(
        self,
        *,
        model: SD3Bundle,
        step: SD3DiffusionStep,
        strategy: StepStrategy,
        autocast_precision: str = "bf16",
        trajectory_precision: str = "fp16",
        logprob_precision: str = "fp32",
        vae_scale_factor: int = 8,
        latent_channels: int = 16,
        batch_replay_steps: bool = False,
    ) -> None:
        super().__init__(
            model=model,
            step=step,
            strategy=strategy,
            autocast_precision=autocast_precision,
            trajectory_precision=trajectory_precision,
            logprob_precision=logprob_precision,
        )
        self.vae_scale_factor = vae_scale_factor
        self.latent_channels = latent_channels
        # Batched-step replay: stack all S SDE steps into one [S*B] transformer
        # forward (+ vectorized SDE transition), cutting per-replay forwards /
        # FSDP all-gathers from S to 1. Stateless SDE strategies only
        # (Flow/Dance/CPS). Under old_logp_source='replay' the anchor and train
        # forward share this path, so the on-policy ratio stays exactly 1.
        self.batch_replay_steps = batch_replay_steps

    def _latent_spec(
        self,
        conditions: SD3Conditions,
        params: DiffusionSamplingParams,
    ) -> SingleStreamLatentSpec:
        if conditions.text is None or conditions.text.embeds is None:
            raise ValueError("SD3DiffusionStage: conditions.text.embeds is None")
        embeds = conditions.text.embeds
        latent_h = int(params.height) // int(self.vae_scale_factor)
        latent_w = int(params.width) // int(self.vae_scale_factor)
        return SingleStreamLatentSpec(
            device=embeds.device,
            batch_size=int(embeds.shape[0]),
            shape=(int(self.latent_channels), latent_h, latent_w),
        )

    def _replay_batched(
        self,
        conditions: SD3Conditions,
        *,
        segment: LatentSegment,
        params: DiffusionSamplingParams,
        target: List[int],
        sigmas: torch.Tensor,
        sigma_max: Any,
        device: torch.device,
    ) -> Optional[ReplayResult]:
        if self.batch_replay_steps and len(target) > 1 and isinstance(self.strategy, SDEStrategy):
            return self._replay_batched_steps(
                conditions,
                segment=segment,
                params=params,
                target=target,
                sigmas=sigmas,
                sigma_max=sigma_max,
                device=device,
            )
        return None

    def _replay_batched_steps(
        self,
        conditions: SD3Conditions,
        *,
        segment: LatentSegment,
        params: DiffusionSamplingParams,
        target: List[int],
        sigmas: torch.Tensor,
        sigma_max: torch.Tensor,
        device: torch.device,
    ) -> ReplayResult:
        """Replay all ``target`` SDE steps in a single batched forward.

        Equivalent to the serial loop above but stacks the S steps on the batch
        dim: ``sample``/``prev_sample`` become ``[S*B, C, H, W]`` (step-major),
        the conditioning text embeds are tiled S× to ``[S*B, ...]``, and the
        per-step ``sigma``/``sigma_next`` ride as ``[S*B]`` vectors. One
        ``step_with_logp`` call then does ONE transformer forward (CFG batching,
        noise prediction) and one vectorized SDE transition over the whole stack;
        the per-step log-probs are reshaped back to ``[B, S]``. The SD3
        transformer has no cross-sample interaction, so per-sample results match
        the serial path up to bf16 batch-shape rounding — and because the π_old
        anchor is replayed through this same method, the on-policy ratio is still
        exactly 1.

        ``step_index`` is passed as ``target[0]`` for signature parity; the
        guarded stateless SDE strategies ignore it.
        """
        S = len(target)
        # Step-major stack: rows [k*B:(k+1)*B] are all B samples at step target[k].
        sample_all = torch.cat([segment.latents_at(i).to(device) for i in target], dim=0)
        prev_all = torch.cat([segment.latents_at(i + 1).to(device) for i in target], dim=0)
        B = sample_all.shape[0] // S
        # Per-sample sigma vectors aligned with the step-major stack.
        sigma_all = torch.cat([sigmas[i].to(torch.float32).expand(B) for i in target], dim=0)
        sigma_next_all = torch.cat([sigmas[i + 1].to(torch.float32).expand(B) for i in target], dim=0)
        tiled = self._tile_conditions(conditions, S)

        _, log_prob_all, prev_mean_all = self.step.step_with_logp(
            self.model,
            tiled,
            strategy=self.strategy,
            sample=sample_all,
            prev_sample=prev_all,
            sigma=sigma_all,
            sigma_next=sigma_next_all,
            guidance_scale=float(params.guidance_scale),
            eta=float(params.eta),
            sigma_max=sigma_max,
            step_index=int(target[0]),
        )
        if log_prob_all is None:
            raise RuntimeError(
                "SD3DiffusionStage._replay_batched_steps: strategy returned None log-prob "
                "(deterministic mode); batched replay requires a stochastic SDE strategy."
            )
        # [S*B] -> [S, B] -> [B, S] so slot s aligns with segment.sde_logp ordering.
        log_probs_t = log_prob_all.view(S, B).transpose(0, 1).contiguous().to(dtype=self.logprob_dtype)
        means_t = None
        if prev_mean_all is not None:
            tail = prev_mean_all.shape[1:]
            means_t = prev_mean_all.view(S, B, *tail).transpose(0, 1).contiguous().to(dtype=self.trajectory_dtype)
        return ReplayResult(log_probs=log_probs_t, prev_sample_means=means_t)

    @staticmethod
    def _tile_conditions(conditions: SD3Conditions, repeats: int) -> SD3Conditions:
        """Repeat the text (and CFG-negative) embeds ``repeats``× along the batch
        dim so they align with the step-major ``[S*B, ...]`` sample stack — each
        block of B reuses the same per-sample conditioning, since all S steps
        replay the SAME B trajectories at different timesteps."""

        def _rep(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            return t.repeat(repeats, *([1] * (t.dim() - 1))) if t is not None else None

        def _tile(cond: Optional[TextEmbedCondition]) -> Optional[TextEmbedCondition]:
            if cond is None:
                return None
            # Tile attn_mask too for metadata parity (SD3 predict_noise ignores
            # it today, but keep the condition self-consistent under batching).
            return TextEmbedCondition(
                embeds=_rep(cond.embeds), pooled=_rep(cond.pooled), attn_mask=_rep(cond.attn_mask)
            )

        return SD3Conditions(text=_tile(conditions.text), negative_text=_tile(conditions.negative_text))


__all__ = ["SD3DiffusionStage", "SD3DiffusionStep"]
