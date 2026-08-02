"""Z-Image diffusion: per-step kernel + rollout-level stage.

Two classes mirror :mod:`unirl.models.sd3.diffusion`:

- :class:`ZImageDiffusionStep` — stateless per-step kernel. Wraps
  :meth:`predict_noise` (which adapts the single-stream
  ``ZImageTransformer2DModel``'s list-based forward to the framework's
  batched ``[B, C, H, W]`` SDE math) around ``StepStrategy.denoise``. The
  protocol-matching ``forward`` / ``step`` / ``step_with_logp`` ride on
  top.
- :class:`ZImageDiffusionStage` — implements
  ``DiffusionStage[ZImageConditions]``. Owns the SDE strategy and loop
  bookkeeping; segment latents stay in spatial ``[B, C, H, W]`` shape so
  :class:`ZImageVAEDecodeStage` can read them directly.

Transformer adapter
-------------------
Z-Image's S3-DiT consumes **lists**: a list of per-sample latents
``[C, F=1, H, W]`` and a list of per-sample caption embeddings
``[t_i, D]`` (variable length). It returns a list of per-sample velocity
predictions ``[C, F=1, H, W]``. :meth:`predict_noise`:

1. lifts the batched latent ``[B, C, H, W]`` → list of ``[C, 1, H, W]``;
2. rebuilds the per-prompt caption list from the padded
   ``conditions.text.embeds`` + ``attn_mask``;
3. passes ``t = 1 - sigma`` as the timestep (the diffusers reference
   feeds ``(1000 - sigma*1000)/1000``);
4. **negates** the model output (the reference does ``noise_pred =
   -model_out`` before the scheduler step) so the result is the
   FlowMatch velocity ``FlowSDEStrategy`` expects;
5. stacks the list back to ``[B, C, H, W]``.

CFG math
--------
Z-Image's CFG is ``pred = pos + scale * (pos - neg)`` (gated on
``guidance_scale > 0``), batched as a single ``[pos; neg]`` forward.
Z-Image-Turbo is distilled to run **without** CFG (``guidance_scale = 0``),
which is the RL-friendly setting; the CFG branch supports the undistilled
base model.

Math mirrors diffusers ``ZImagePipeline.__call__`` denoising loop.
"""

from __future__ import annotations

from typing import ClassVar, List, Optional, Tuple

import torch

from unirl.models.diffusion import DiffusionLatentSpec, DiffusionRunner, DiffusionStep
from unirl.sde.kernels import StepStrategy
from unirl.types.sampling import DiffusionSamplingParams

from .bundle import ZImageBundle
from .conditions import ZImageConditions


def _caption_list(text, dtype: torch.dtype, device: torch.device) -> List[torch.Tensor]:
    """Rebuild the per-prompt variable-length caption list from a padded
    ``TextEmbedCondition`` (``embeds [B, T, D]`` + ``attn_mask [B, T]``).

    Dedicated-engine replay can hand conditions back on CPU; pin both the
    embeds and mask to the transformer's device before splitting.
    """
    if text is None or text.embeds is None:
        raise ValueError("ZImage predict_noise: conditions text/embeds is None")
    embeds = text.embeds.to(device=device, dtype=dtype)
    mask = text.attn_mask
    bsz = int(embeds.shape[0])
    if mask is None:
        return [embeds[i] for i in range(bsz)]
    bool_mask = mask.to(device).bool()
    return [embeds[i][bool_mask[i]] for i in range(bsz)]


class ZImageDiffusionStep(DiffusionStep[ZImageBundle, ZImageConditions]):
    """Per-step Z-Image denoising kernel — stateless."""

    def predict_noise(
        self,
        model: ZImageBundle,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        conditions: ZImageConditions,
        *,
        guidance_scale: float,
    ) -> torch.Tensor:
        """Run the single-stream Z-Image transformer and return the
        FlowMatch velocity ``[B, C, H, W]`` (negated model output, with CFG
        applied when ``guidance_scale > 0`` and a negative is present)."""
        if conditions.text is None:
            raise ValueError("ZImageDiffusionStep.predict_noise: conditions.text is None")

        dev = model.device
        sample = sample.to(dev)
        sigma = sigma.to(dev)
        try:
            model_dtype = next(model.transformer.parameters()).dtype
        except StopIteration:
            model_dtype = sample.dtype
        sample = sample.to(dtype=model_dtype)

        batch_size = int(sample.shape[0])
        # Z-Image timestep input: (1000 - sigma*1000)/1000 == 1 - sigma.
        if sigma.dim() == 0:
            timestep = (1.0 - sigma).expand(batch_size)
        elif sigma.shape[0] != batch_size:
            timestep = (1.0 - sigma).expand(batch_size)
        else:
            timestep = 1.0 - sigma
        timestep = timestep.to(device=dev, dtype=torch.float32)

        cap_list = _caption_list(conditions.text, model_dtype, dev)

        # Lift batched latent [B, C, H, W] -> list of [C, 1, H, W].
        x_5d = sample.unsqueeze(2)  # [B, C, 1, H, W]

        use_cfg = guidance_scale > 0.0 and conditions.negative_text is not None
        if use_cfg:
            neg_list = _caption_list(conditions.negative_text, model_dtype, dev)
            x_list = list(torch.cat([x_5d, x_5d], dim=0).unbind(dim=0))
            cap_all = cap_list + neg_list
            timestep_all = timestep.repeat(2)
            out_list = model.transformer(x_list, timestep_all, cap_all, return_dict=False)[0]
            pos = torch.stack(out_list[:batch_size], dim=0)
            neg = torch.stack(out_list[batch_size:], dim=0)
            combined = pos + guidance_scale * (pos - neg)
            noise_pred = -combined
        else:
            x_list = list(x_5d.unbind(dim=0))
            out_list = model.transformer(x_list, timestep, cap_list, return_dict=False)[0]
            noise_pred = -torch.stack(out_list, dim=0)

        # Drop the temporal dim (Z-Image t2i uses F=1).
        return noise_pred.squeeze(2)

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
        model: ZImageBundle,
        conditions: ZImageConditions,
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
        """Run model forward + SDE transition. End-to-end one diffusion step."""
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
        model: ZImageBundle,
        conditions: ZImageConditions,
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


class ZImageDiffusionStage(DiffusionRunner[ZImageBundle, ZImageConditions]):
    """Z-Image rollout-level diffusion stage.

    Owns the SDE ``strategy`` (stateful strategies like ``DPM2Strategy``
    require a stable instance across the loop), the bundle, the kernel, and
    the precision policy. The kernel is stateless and invoked per-step.

    Segment latents stay in spatial ``[B, C, H, W]`` shape (Z-Image's VAE
    is the standard 2D ``AutoencoderKL``), so :class:`ZImageVAEDecodeStage`
    reads ``segment.latents[:, -1]`` without per-shape handling.

    ``_no_split_modules`` is the model-side fallback used by FSDPPolicy when
    HF auto-discovery yields nothing — diffusers'
    ``ZImageTransformer2DModel`` block class is ``ZImageTransformerBlock``.
    """

    _no_split_modules: ClassVar[Tuple[str, ...]] = ("ZImageTransformerBlock",)

    def __init__(
        self,
        *,
        model: ZImageBundle,
        step: ZImageDiffusionStep,
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

    def _latent_spec(
        self,
        conditions: ZImageConditions,
        params: DiffusionSamplingParams,
    ) -> DiffusionLatentSpec:
        if conditions.text is None or conditions.text.embeds is None:
            raise ValueError("ZImageDiffusionStage: conditions.text.embeds is None")
        embeds = conditions.text.embeds
        vsf = int(self.vae_scale_factor)
        latent_h = 2 * (int(params.height) // (vsf * 2))
        latent_w = 2 * (int(params.width) // (vsf * 2))
        return DiffusionLatentSpec(
            device=embeds.device,
            batch_size=int(embeds.shape[0]),
            shape=(int(self.latent_channels), latent_h, latent_w),
        )


__all__ = ["ZImageDiffusionStage", "ZImageDiffusionStep"]
