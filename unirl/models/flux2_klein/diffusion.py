"""FLUX.2-klein diffusion: typed params + per-step kernel + rollout-level stage.

Mirrors :mod:`unirl.models.sd3.diffusion` and
:mod:`unirl.models.qwen_image.diffusion`. Three classes:

- :class:`Flux2KleinDiffusionParams` — typed request-shape knobs
  (steps / guidance / size / seed / sde_indices / eta /
  init_same_noise / samples_per_prompt / noise_group_ids).
- :class:`Flux2KleinDiffusionStep` — stateless per-step kernel. Packs
  patchified latents ``[B, 128, H_pat, W_pat]`` into the transformer's
  expected ``[B, H_pat*W_pat, 128]`` layout, builds RoPE ``txt_ids`` /
  ``img_ids``, calls the transformer with ``guidance=torch.zeros(B)``
  (Klein has no guidance distillation), and unpacks the noise
  prediction back to patchified spatial form.
- :class:`Flux2KleinDiffusionStage` — implements
  ``DiffusionStage[Flux2KleinConditions]``. Owns the SDE strategy and
  loop bookkeeping; segment latents stay in patchified ``[B, 128,
  H_pat, W_pat]`` shape so :class:`Flux2KleinVAEDecodeStage` can read
  them directly without per-shape special-casing.

Klein vs. dev (FLUX.2-dev) differences:

- **No CFG branch consumed by the transformer**. Klein checkpoints ship
  with ``has_pooled_projections=false`` and ``guidance_embeds=false``,
  so we always feed ``guidance=torch.zeros(B)`` and never pass
  ``pooled_projections``. The Klein training script also runs with
  ``guidance_scale=1.0`` so the CFG combine math is bypassed
  end-to-end.
- **Pre-patchified latent space**. Latents live in the 128-channel
  patchified space ``[B, 128, H_pix/16, W_pix/16]`` throughout the SDE
  loop (vs. dev's 32-channel ``[B, 32, H_pix/8, W_pix/8]`` form). The
  VAE decode stage handles the inverse: unpack → denormalize →
  unpatchify → decode.
- **4-axis RoPE ids**. ``txt_ids`` ``[B, L, 4]`` and ``img_ids``
  ``[B, H_pat*W_pat, 4]`` are built via :func:`prepare_text_ids` /
  :func:`prepare_latent_ids`; passing FLUX.1's 3-axis form crashes
  inside ``FluxPosEmbed`` because Klein's
  ``axes_dims_rope=[32, 32, 32, 32]``.
- **Replay uses eval() mode**. To mirror the legacy
  ``Flux2Sampler.compute_log_prob_for_training`` Klein branch and the
  FLUX.2 PR's safety fence: the transformer stays in ``.eval()``
  inside ``step.predict_noise`` during replay. Caller manages
  ``train()`` / ``eval()`` mode at the outer scope.

Math mirrors ``samplers/fsdp/flux2_sampler.py::Flux2Sampler.sample``
(Klein branch). The new-design path does NOT import legacy code; the
two implementations must stay in spec sync via review and tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, List, Optional, Tuple

import torch

from unirl.models.diffusion import (
    SingleStreamDiffusionRunner,
    SingleStreamDiffusionStep,
    SingleStreamLatentSpec,
    temporary_eval,
)
from unirl.sde.kernels import StepStrategy
from unirl.types.segments.latent import LatentSegment

from .bundle import Flux2KleinBundle
from .conditions import Flux2KleinConditions
from .flux2_klein_utils import (
    pack_latents,
    prepare_latent_ids,
    prepare_text_ids,
    unpack_latents,
)


@dataclass
class Flux2KleinDiffusionParams:
    """Per-request sampling knobs for FLUX.2-klein diffusion.

    Strategy + precision knobs are *not* here — they live at
    :class:`Flux2KleinDiffusionStage` construction since precision is
    operator policy, not request shape. Klein's transformer ignores
    ``guidance_scale`` (no guidance distillation, no CFG-consuming
    pooled projection), but the field is kept for API symmetry with
    SD3 / Qwen-Image. ``guidance_scale > 1.0`` will *also* trigger a
    classical CFG combine if ``conditions.negative_text`` is supplied
    — the canonical Klein recipe runs at ``guidance_scale=1.0`` with
    no negative branch.
    """

    num_inference_steps: int = 10
    guidance_scale: float = 1.0
    height: int = 512
    width: int = 512
    seed: int = 42
    sde_indices: Optional[List[int]] = None
    eta: float = 0.7
    init_same_noise: bool = False
    samples_per_prompt: int = 1
    noise_group_ids: Optional[List[str]] = None


class Flux2KleinDiffusionStep(SingleStreamDiffusionStep[Flux2KleinBundle, Flux2KleinConditions]):
    """Per-step FLUX.2-klein denoising kernel — stateless.

    Operates on patchified ``[B, 128, H_pat, W_pat]`` latents. Packs to
    ``[B, H_pat*W_pat, 128]`` for the transformer forward, then unpacks
    the noise prediction back to spatial form.
    """

    def predict_noise(
        self,
        model: Flux2KleinBundle,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        conditions: Flux2KleinConditions,
        *,
        guidance_scale: float,
    ) -> torch.Tensor:
        """Run Klein transformer forward.

        ``sample`` is the patchified latent ``[B, 128, H_pat, W_pat]``.
        Returns the noise prediction in the same shape.

        Klein's transformer expects ``guidance=torch.zeros(B)`` (no
        guidance distillation) and does **not** accept
        ``pooled_projections``. ``txt_ids`` / ``img_ids`` are 4-axis
        RoPE coordinate tensors.
        """
        if conditions.text is None:
            raise ValueError("Flux2KleinDiffusionStep.predict_noise: conditions.text is None")
        text = conditions.text
        prompt_embeds = text.embeds
        if prompt_embeds is None:
            raise ValueError("Flux2KleinDiffusionStep.predict_noise: conditions.text.embeds is None")

        batch_size = sample.shape[0]
        device = sample.device
        dtype = prompt_embeds.dtype

        packed = pack_latents(sample).to(dtype=dtype)
        noise_seq_len = packed.shape[1]

        if sigma.dim() == 0:
            timestep = sigma.float().expand(batch_size).to(device)
        elif sigma.shape[0] != batch_size:
            timestep = sigma.float().expand(batch_size).to(device)
        else:
            timestep = sigma.float().to(device)

        guidance = torch.zeros(batch_size, device=device, dtype=dtype)
        txt_ids = prepare_text_ids(prompt_embeds).to(device=device)
        img_ids = prepare_latent_ids(sample).to(device=device)

        # Image-edit conditioning: append the source-image condition tokens to
        # the noise token sequence (and their RoPE ids to img_ids), mirroring
        # diffusers' Flux2KleinPipeline reference path
        # (latent_model_input = cat([latents, image_latents], dim=1)). The
        # transformer attends jointly; we slice the prediction back to the
        # noise tokens afterwards. Pure T2I leaves these None → no-op.
        cond_tokens = conditions.image_latent
        if cond_tokens is not None:
            cond_tokens = cond_tokens.to(device=device, dtype=dtype)
            packed = torch.cat([packed, cond_tokens], dim=1)
            cond_ids = conditions.image_latent_ids
            if cond_ids is None:
                raise ValueError(
                    "Flux2KleinDiffusionStep.predict_noise: conditions.image_latent set "
                    "but image_latent_ids is None; both are required for the edit path."
                )
            img_ids = torch.cat([img_ids, cond_ids.to(device=device)], dim=1)

        noise_pred_packed = model.transformer(
            hidden_states=packed,
            encoder_hidden_states=prompt_embeds,
            timestep=timestep,
            guidance=guidance,
            txt_ids=txt_ids,
            img_ids=img_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]
        # Drop the condition-token predictions; keep only the noise tokens.
        noise_pred_packed = noise_pred_packed[:, :noise_seq_len]

        if guidance_scale > 1.0:
            neg = conditions.negative_text
            if neg is not None and neg.embeds is not None:
                negative_prompt_embeds = neg.embeds
                neg_txt_ids = prepare_text_ids(negative_prompt_embeds).to(device=device)
                negative_noise_pred_packed = model.transformer(
                    hidden_states=packed,
                    encoder_hidden_states=negative_prompt_embeds,
                    timestep=timestep,
                    guidance=guidance,
                    txt_ids=neg_txt_ids,
                    img_ids=img_ids,
                    joint_attention_kwargs=None,
                    return_dict=False,
                )[0][:, :noise_seq_len]
                noise_pred_packed = negative_noise_pred_packed + guidance_scale * (
                    noise_pred_packed - negative_noise_pred_packed
                )

        latent_h = int(sample.shape[-2])
        latent_w = int(sample.shape[-1])
        return unpack_latents(noise_pred_packed, latent_h, latent_w)

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
        model: Flux2KleinBundle,
        conditions: Flux2KleinConditions,
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
        model: Flux2KleinBundle,
        conditions: Flux2KleinConditions,
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


class Flux2KleinDiffusionStage(SingleStreamDiffusionRunner[Flux2KleinBundle, Flux2KleinConditions]):
    """FLUX.2-klein rollout-level diffusion stage.

    Owns the SDE ``strategy`` (DanceSDE by default for Klein), the
    bundle, the kernel, and the precision policy. The kernel is
    stateless and is invoked per-step with the strategy passed in.

    Segment latents are stored as **patchified** spatial tensors
    ``[B, K, 128, H_pat, W_pat]`` so :class:`Flux2KleinVAEDecodeStage`
    can read them directly. The pack/unpack at the transformer
    boundary lives in :class:`Flux2KleinDiffusionStep`.

    ``_no_split_modules`` is the model-side fallback used by
    FSDPPolicy: Klein's transformer block classes are
    ``Flux2TransformerBlock`` (dual-stream) plus
    ``Flux2SingleTransformerBlock`` (single-stream). These match the
    installed diffusers ``Flux2Transformer2DModel._no_split_modules``.
    """

    _no_split_modules: ClassVar[Tuple[str, ...]] = (
        "Flux2TransformerBlock",
        "Flux2SingleTransformerBlock",
    )

    # FLUX.2-klein VAE spatial downsample (8×) and patchify factor (2×)
    # → effective patchified downsample 16×. The bundle's
    # ``transformer.config.in_channels`` is the patchified channel count
    # (128 = 32 × 4); we use it to derive ``latent_channels`` (32).
    DEFAULT_VAE_SCALE_FACTOR: ClassVar[int] = 8
    DEFAULT_PATCHIFY_FACTOR: ClassVar[int] = 2

    def __init__(
        self,
        *,
        model: Flux2KleinBundle,
        step: Flux2KleinDiffusionStep,
        strategy: StepStrategy,
        autocast_precision: str = "bf16",
        trajectory_precision: str = "fp16",
        logprob_precision: str = "fp32",
        vae_scale_factor: int = 8,
        patchify_factor: int = 2,
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
        self.vae_scale_factor = int(vae_scale_factor)
        self.patchify_factor = int(patchify_factor)
        if latent_channels is None:
            tx_cfg = getattr(model.transformer, "config", None)
            in_channels = getattr(tx_cfg, "in_channels", 128) if tx_cfg is not None else 128
            latent_channels = int(in_channels)
        self.latent_channels = int(latent_channels)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _patchified_shape(self, height: int, width: int) -> Tuple[int, int, int]:
        """Compute the patchified ``(C, H_pat, W_pat)`` for ``(height, width)`` pixels."""
        downsample = self.vae_scale_factor * self.patchify_factor
        if height % downsample != 0 or width % downsample != 0:
            raise ValueError(
                f"Flux2KleinDiffusionStage: height ({height}) and width ({width}) "
                f"must be divisible by VAE×patchify downsample ({downsample})."
            )
        h_pat = height // downsample
        w_pat = width // downsample
        return (self.latent_channels, h_pat, w_pat)

    def _latent_spec(
        self,
        conditions: Flux2KleinConditions,
        params: Flux2KleinDiffusionParams,
    ) -> SingleStreamLatentSpec:
        if conditions.text is None or conditions.text.embeds is None:
            raise ValueError("Flux2KleinDiffusionStage: conditions.text.embeds is None")
        embeds = conditions.text.embeds
        return SingleStreamLatentSpec(
            device=embeds.device,
            batch_size=int(embeds.shape[0]),
            shape=self._patchified_shape(int(params.height), int(params.width)),
        )

    def _before_diffuse(
        self,
        conditions: Flux2KleinConditions,
        params: Flux2KleinDiffusionParams,
        *,
        schedule: torch.Tensor,
        spec: SingleStreamLatentSpec,
    ) -> None:
        del conditions, params, schedule, spec
        self.model.transformer.eval()

    def _validate_replay_segment(self, segment: LatentSegment) -> None:
        super()._validate_replay_segment(segment)
        if segment.latents.ndim != 5:
            raise ValueError(
                f"Flux2KleinDiffusionStage.replay: expected latents [B, K, C, H_pat, W_pat], "
                f"got {tuple(segment.latents.shape)}"
            )

    def _replay_context(self):
        return temporary_eval(self.model.transformer)


__all__ = [
    "Flux2KleinDiffusionParams",
    "Flux2KleinDiffusionStage",
    "Flux2KleinDiffusionStep",
]
