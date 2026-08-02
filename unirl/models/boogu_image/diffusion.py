"""Boogu-Image diffusion: per-step kernel + rollout-level stage.

Two classes mirror :mod:`unirl.models.z_image.diffusion`:

- :class:`BooguImageDiffusionStep` — stateless per-step kernel. Wraps
  :meth:`predict_noise` (which adapts the vendored
  ``BooguImageTransformer2DModel`` forward to the framework's batched
  ``[B, C, H, W]`` SDE math) around ``StepStrategy.denoise``. The
  protocol-matching ``forward`` / ``step`` / ``step_with_logp`` ride on top.
- :class:`BooguImageDiffusionStage` — implements
  ``DiffusionStage[BooguImageConditions]``. Owns the SDE strategy, the loop
  bookkeeping, and the cached rotary tables; segment latents stay in spatial
  ``[B, C, H, W]`` so :class:`BooguImageVAEDecodeStage` reads them directly.

Transformer adapter
-------------------
The vendored DiT consumes batched 4D latents directly (it lifts to a
per-sample list internally for variable-length packing). :meth:`predict_noise`:

1. passes ``t = 1 - sigma`` as the timestep in **model dtype** (the reference
   ``predict`` does ``t.expand(B).to(latents.dtype)``; the transformer scales
   by ``timestep_scale=1000`` internally);
2. forwards positionally per the reference
   (``transformer(latents, timestep, instruction_embeds, freqs_cis,
   instruction_attention_mask, ref_image_hidden_states=None)``) — the
   default ``return_dict=False`` returns a **bare tensor**, not a tuple
   (indexing ``[0]`` would take sample 0 of the batch);
3. applies CFG with Boogu's convention — gate ``guidance_scale > 1.0``
   (1.0 == off), a **second sequential forward** on the negative branch
   (the reference runs branches sequentially; positive/negative embeds are
   padded to different lengths), plain linear combine
   ``out + (g - 1) * (out - neg_out)`` with no norm correction;
4. **negates** the result: Boogu integrates ``x += (t_next - t)·v`` in the
   t-convention (t = 1 - σ, toward data), so the σ-convention FlowMatch
   velocity ``FlowSDEStrategy`` expects is ``-v`` (z_image precedent).

Rotary tables (``freqs_cis``) are the reference pipeline's per-call input:
built once from ``(axes_dim_rope, axes_lens, theta=10000)`` — resolution
independent — and cached per device on the stage.

CFG range
---------
The reference gates guidance by step fraction (``cfg_range``,
pipeline_boogu.py:3367-3381; default ``(0, 1)`` == always on). The stage
collapses that into a per-step *effective* guidance scale
(:meth:`BooguImageDiffusionStage._effective_guidance_scale`); scale 1.0 makes
the kernel skip the negative forward exactly like the reference's skipped
branch. ``cfg_range`` rides in ``DiffusionSamplingParams.sampler_kwargs``.

Math mirrors the reference ``BooguImagePipeline.processing`` T2I denoising
loop (pipeline_boogu.py:3243-3688).
"""

from __future__ import annotations

from typing import Any, ClassVar, List, Mapping, Optional, Tuple

import torch

from unirl.models.diffusion import DiffusionLatentSpec, DiffusionRunner, DiffusionStep
from unirl.sde.kernels import StepStrategy
from unirl.types.sampling import DiffusionSamplingParams

from .bundle import BooguImageBundle
from .conditions import BooguImageConditions
from .vendor.rope import BooguImageRotaryPosEmbed


def build_freqs_cis(transformer_config, device: torch.device) -> List[torch.Tensor]:
    """Build the reference pipeline's rotary tables for the vendored DiT.

    Depends only on ``(axes_dim_rope, axes_lens, theta)`` — resolution
    independent; the per-(H, W) position gather happens inside the
    transformer's ``rope_embedder``. Mirrors the reference ``__call__``
    (pipeline_boogu.py:2896-2900).
    """
    tables = BooguImageRotaryPosEmbed.get_freqs_cis(
        list(transformer_config.axes_dim_rope),
        list(transformer_config.axes_lens),
        theta=10000,
    )
    return [t.to(device) for t in tables]


class BooguImageDiffusionStep(DiffusionStep[BooguImageBundle, BooguImageConditions]):
    """Per-step Boogu-Image denoising kernel — stateless."""

    def predict_noise(
        self,
        model: BooguImageBundle,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        conditions: BooguImageConditions,
        *,
        guidance_scale: float,
        freqs_cis: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Run the vendored DiT and return the FlowMatch velocity
        ``[B, C, H, W]`` (negated model output, with Boogu's text CFG applied
        when ``guidance_scale > 1.0``)."""
        if conditions.text is None or conditions.text.embeds is None:
            raise ValueError("BooguImageDiffusionStep.predict_noise: conditions.text is None")

        dev = model.device
        try:
            model_dtype = next(model.transformer.parameters()).dtype
        except StopIteration:
            model_dtype = sample.dtype
        sample = sample.to(device=dev, dtype=model_dtype)
        sigma = sigma.to(dev)

        batch_size = int(sample.shape[0])
        # Boogu timestep input: t = 1 - sigma, in MODEL dtype (the reference
        # predict() does t.expand(B).to(latents.dtype); timestep_scale=1000
        # is applied inside the transformer).
        if sigma.dim() == 0 or sigma.shape[0] != batch_size:
            timestep = (1.0 - sigma).expand(batch_size)
        else:
            timestep = 1.0 - sigma
        timestep = timestep.to(device=dev, dtype=model_dtype)

        if freqs_cis is None:
            freqs_cis = build_freqs_cis(model.transformer.config, dev)

        embeds = conditions.text.embeds.to(device=dev, dtype=model_dtype)
        mask = conditions.text.attn_mask
        if mask is None:
            raise ValueError("BooguImageDiffusionStep.predict_noise: conditions.text.attn_mask is None")
        mask = mask.to(dev)

        # Bare-tensor return: the vendored forward defaults return_dict=False
        # and returns [B, C, H, W] directly — no `[0]` indexing.
        out = model.transformer(
            sample,
            timestep,
            embeds,
            freqs_cis,
            mask,
            ref_image_hidden_states=None,
        )

        # Boogu CFG gate: 1.0 == off (unlike z_image's 0.0). Sequential
        # negative forward — positive/negative embeds are padded to
        # different lengths, and the reference runs branches sequentially.
        use_cfg = guidance_scale > 1.0 and conditions.negative_text is not None
        if use_cfg:
            neg = conditions.negative_text
            if neg.embeds is None or neg.attn_mask is None:
                raise ValueError("BooguImageDiffusionStep.predict_noise: negative_text embeds/attn_mask is None")
            neg_out = model.transformer(
                sample,
                timestep,
                neg.embeds.to(device=dev, dtype=model_dtype),
                freqs_cis,
                neg.attn_mask.to(dev),
                ref_image_hidden_states=None,
            )
            # Reference combine (pipeline_boogu.py:3646-3649), plain linear:
            # model_pred + (g - 1) * (model_pred - model_pred_drop_all).
            out = out + (guidance_scale - 1.0) * (out - neg_out)
        elif guidance_scale > 1.0:
            raise ValueError(
                "BooguImageDiffusionStep.predict_noise: guidance_scale > 1.0 "
                "but conditions.negative_text is None — the pipeline should "
                "have built ''-negatives"
            )

        # t-convention velocity -> sigma-convention FlowMatch velocity.
        return -out

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
        """Run one SDE transition given a precomputed ``noise_pred``."""
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
        model: BooguImageBundle,
        conditions: BooguImageConditions,
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
        freqs_cis: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run model forward + SDE transition. End-to-end one diffusion step."""
        noise_pred = self.predict_noise(
            model,
            sample,
            sigma,
            conditions,
            guidance_scale=guidance_scale,
            freqs_cis=freqs_cis,
        )
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
        model: BooguImageBundle,
        conditions: BooguImageConditions,
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
        freqs_cis: Optional[List[torch.Tensor]] = None,
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
            freqs_cis=freqs_cis,
        )


class BooguImageDiffusionStage(DiffusionRunner[BooguImageBundle, BooguImageConditions]):
    """Boogu-Image rollout-level diffusion stage.

    Owns the SDE ``strategy``, the bundle, the stateless kernel, the
    precision policy, and the per-device rotary-table cache.

    Segment latents stay in spatial ``[B, C, H, W]`` (standard 2D FLUX
    ``AutoencoderKL``), so :class:`BooguImageVAEDecodeStage` reads
    ``segment.latents[:, -1]`` without per-shape handling.

    ``_no_split_modules`` is the model-side fallback used by FSDPPolicy when
    HF auto-discovery yields nothing. NOTE: FSDP wrap matching is by exact
    concrete class name — recipes must list the five INSTANTIATED block
    classes (the bare ``BooguImageTransformerBlock`` is never instantiated,
    and ``BooguImageDoubleStreamTransformerBlock`` is a separate class, not
    a subclass).
    """

    _no_split_modules: ClassVar[Tuple[str, ...]] = (
        "BooguImageTransformerBlock",
        "BooguImageNoiseRefinerTransformerBlock",
        "BooguImageRefImgRefinerTransformerBlock",
        "BooguImageContextRefinerTransformerBlock",
        "BooguImageSingleStreamTransformerBlock",
        "BooguImageDoubleStreamTransformerBlock",
    )

    def __init__(
        self,
        *,
        model: BooguImageBundle,
        step: BooguImageDiffusionStep,
        strategy: StepStrategy,
        autocast_precision: str = "bf16",
        trajectory_precision: str = "fp16",
        logprob_precision: str = "fp32",
        vae_scale_factor: int = 8,
        latent_channels: Optional[int] = None,
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
        if latent_channels is None:
            tx_cfg = getattr(model.transformer, "config", None)
            in_channels = getattr(tx_cfg, "in_channels", 16) if tx_cfg is not None else 16
            latent_channels = int(in_channels)
        self.latent_channels = int(latent_channels)
        self._freqs_cis: Optional[List[torch.Tensor]] = None
        self._freqs_cis_device: Optional[torch.device] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_freqs_cis(self, device: torch.device) -> List[torch.Tensor]:
        """Per-device cache of the resolution-independent rotary tables."""
        if self._freqs_cis is None or self._freqs_cis_device != device:
            self._freqs_cis = build_freqs_cis(self.model.transformer.config, device)
            self._freqs_cis_device = device
        return self._freqs_cis

    @staticmethod
    def _effective_guidance_scale(step_index: int, num_steps: int, params: DiffusionSamplingParams) -> float:
        """Collapse the reference's ``cfg_range`` step-fraction gate into a
        per-step scale (pipeline_boogu.py:3367-3381, inclusive bounds; the
        out-of-range value is 1.0 == CFG off in Boogu's convention).

        ``cfg_range`` rides in ``params.sampler_kwargs``; the default
        ``(0.0, 1.0)`` keeps guidance on at every step. Deterministic in
        ``(step_index, num_steps)``, so replay reproduces rollout exactly.
        """
        lo, hi = params.sampler_kwargs.get("cfg_range", (0.0, 1.0))
        fraction = step_index / num_steps if num_steps > 0 else 0.0
        if float(lo) <= fraction <= float(hi):
            return float(params.guidance_scale)
        return 1.0

    def _latent_spec(
        self,
        conditions: BooguImageConditions,
        params: DiffusionSamplingParams,
    ) -> DiffusionLatentSpec:
        if conditions.text is None or conditions.text.embeds is None:
            raise ValueError("BooguImageDiffusionStage: conditions.text.embeds is None")
        embeds = conditions.text.embeds
        vsf = int(self.vae_scale_factor)
        latent_h = 2 * (int(params.height) // (vsf * 2))
        latent_w = 2 * (int(params.width) // (vsf * 2))
        return DiffusionLatentSpec(
            device=embeds.device,
            batch_size=int(embeds.shape[0]),
            shape=(int(self.latent_channels), latent_h, latent_w),
        )

    def _step_kwargs(
        self,
        conditions: BooguImageConditions,
        params: DiffusionSamplingParams,
        *,
        sample: torch.Tensor,
        step_index: int,
        num_steps: int,
        mode: str,
        state: Any,
    ) -> Mapping[str, Any]:
        del conditions, params, step_index, num_steps, mode, state
        return {"freqs_cis": self._get_freqs_cis(sample.device)}

    def _guidance_scale(
        self,
        params: DiffusionSamplingParams,
        *,
        step_index: int,
        num_steps: int,
        mode: str,
    ) -> float:
        if mode == "predict":
            return float(params.guidance_scale)
        return self._effective_guidance_scale(step_index, num_steps, params)


__all__ = ["BooguImageDiffusionStage", "BooguImageDiffusionStep", "build_freqs_cis"]
