"""Qwen-Image diffusion: typed params + per-step kernel + rollout-level stage.

Three classes mirror :mod:`unirl.models.sd3.diffusion`:

- :class:`QwenImageDiffusionParams` — typed request-shape knobs (steps /
  guidance / size / seed / sde_indices / eta / init_same_noise /
  samples_per_prompt / noise_group_ids /
  ``distilled_guidance_scale``).
- :class:`QwenImageDiffusionStep` — stateless per-step kernel. Wraps
  :meth:`predict_noise` (which packs latents into the
  ``[B, S, C*4]`` patch layout the Qwen-Image transformer expects,
  builds ``img_shapes``, runs CFG with **norm correction**, then unpacks
  the noise prediction back to ``[B, C, H, W]``) around
  ``StepStrategy.denoise``. The protocol-matching ``forward`` /
  ``step`` / ``step_with_logp`` ride on top.
- :class:`QwenImageDiffusionStage` — implements
  ``DiffusionStage[QwenImageConditions]``. Owns the SDE strategy and
  loop bookkeeping; segment latents stay in spatial ``[B, C, H, W]``
  shape so :class:`QwenImageVAEDecodeStage` can read them directly.

CFG math
--------
The Qwen-Image pipeline does **not** use the standard
``uncond + scale * (cond - uncond)`` form; it applies the combined
prediction, then rescales it to preserve the per-token L2 norm of the
conditional prediction. This is what the legacy
``models/qwen_image.py::forward_denoiser`` (PR #104 lines 506-511)
does, and it ships as the official Qwen-Image inference recipe::

    comb = neg + scale * (cond - neg)
    cond_norm = ||cond||_{dim=-1, keepdim=True}
    comb_norm = ||comb||_{dim=-1, keepdim=True}
    noise_pred = comb * (cond_norm / comb_norm)

The CFG batching is per-branch (two separate transformer forwards),
not the SD3-style ``[uncond, cond]`` chunked forward, because Qwen-VL
prompts have variable-length sequences with attention masks that
don't match between branches.

Latent packing
--------------
The Qwen-Image transformer operates on patchified latents
``[B, (H/2)*(W/2), C*4]`` (2×2 patches in the spatial plane). The
SDE loop, segment storage, and noise generation all use the
**unpacked** ``[B, C, H, W]`` shape; only :meth:`predict_noise`
packs/unpacks at the transformer boundary. This keeps
``LatentSegment.latents`` in the same ``[B, K, C, H, W]`` shape SD3
and Wan use, so :class:`QwenImageVAEDecodeStage` follows the SD3 decode
protocol without per-shape special-casing.

Math mirrors PR #104's ``qwen_image_sampler.py`` / ``forward_denoiser``.
"""

from __future__ import annotations

from typing import Any, ClassVar, Mapping, Optional, Tuple

import torch

from unirl.models.diffusion import DiffusionLatentSpec, DiffusionRunner, DiffusionStep
from unirl.sde.kernels import StepStrategy
from unirl.types.sampling import DiffusionSamplingParams
from unirl.types.segments.latent import LatentSegment

from .bundle import QwenImageBundle
from .conditions import QwenImageConditions

# --------------------------------------------------------------------------
# Pack / unpack helpers — module-level so unit tests can import them
# without constructing the stage.
# --------------------------------------------------------------------------


def _pack_latents(latents: torch.Tensor) -> torch.Tensor:
    """``[B, C, H, W]`` → ``[B, (H/2)*(W/2), C*4]``.

    Reshapes the spatial grid into 2×2 patches and flattens. Mirrors
    ``samplers/fsdp/qwen_image_sampler.py::_pack_latents``.
    """
    if latents.ndim != 4:
        raise ValueError(f"_pack_latents: expected [B, C, H, W], got {tuple(latents.shape)}")
    batch_size, channels, height, width = latents.shape
    if height % 2 != 0 or width % 2 != 0:
        raise ValueError(f"_pack_latents: H ({height}) and W ({width}) must be divisible by 2")
    latents = latents.view(batch_size, channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(batch_size, (height // 2) * (width // 2), channels * 4)


def _unpack_latents(latents: torch.Tensor, *, latent_h: int, latent_w: int) -> torch.Tensor:
    """``[B, S, C*4]`` → ``[B, C, H, W]`` (inverse of :func:`_pack_latents`).

    Requires ``S == (H/2)*(W/2)``; ``H = latent_h``, ``W = latent_w``.
    """
    if latents.ndim != 3:
        raise ValueError(f"_unpack_latents: expected [B, S, C*4], got {tuple(latents.shape)}")
    batch_size, seq, packed_channels = latents.shape
    expected_seq = (latent_h // 2) * (latent_w // 2)
    if seq != expected_seq:
        raise ValueError(
            f"_unpack_latents: seq ({seq}) does not match "
            f"(latent_h/2)*(latent_w/2) = {expected_seq} for "
            f"latent_h={latent_h}, latent_w={latent_w}"
        )
    if packed_channels % 4 != 0:
        raise ValueError(f"_unpack_latents: packed channels ({packed_channels}) must be divisible by 4")
    channels = packed_channels // 4
    latents = latents.view(batch_size, latent_h // 2, latent_w // 2, channels, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    return latents.reshape(batch_size, channels, latent_h, latent_w)


class QwenImageDiffusionStep(DiffusionStep[QwenImageBundle, QwenImageConditions]):
    """Per-step Qwen-Image denoising kernel — stateless.

    Extends the :class:`DiffusionStep` protocol with Qwen-Image-specific
    per-call kwargs (``latent_h`` / ``latent_w`` /
    ``distilled_guidance_scale``) on :meth:`predict_noise`,
    :meth:`step`, and :meth:`step_with_logp`. The protocol surface stays
    structurally compatible because Python protocols are non-strict on
    extra kwargs.
    """

    def predict_noise(
        self,
        model: QwenImageBundle,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        conditions: QwenImageConditions,
        *,
        guidance_scale: float,
        latent_h: int,
        latent_w: int,
        distilled_guidance_scale: Optional[float] = None,
    ) -> torch.Tensor:
        """Run the Qwen-Image transformer with combined-CFG + norm correction.

        Packs ``sample`` ``[B, C, H, W]`` → ``[B, (H/2)*(W/2), C*4]``,
        runs the transformer for the conditional branch (and, when
        ``guidance_scale > 1`` and ``conditions.negative_text`` is set,
        a second forward for the unconditional branch), applies the
        norm-corrected CFG blend, then unpacks the result back to
        ``[B, C, H, W]``.
        """
        if conditions.text is None:
            raise ValueError("QwenImageDiffusionStep.predict_noise: conditions.text is None")
        text = conditions.text
        prompt_embeds = text.embeds
        prompt_embeds_mask = text.attn_mask
        if prompt_embeds is None:
            raise ValueError("QwenImageDiffusionStep.predict_noise: conditions.text.embeds is None")
        if prompt_embeds_mask is None:
            raise ValueError("QwenImageDiffusionStep.predict_noise: conditions.text.attn_mask is None")

        batch_size = sample.shape[0]
        device = sample.device
        dtype = prompt_embeds.dtype
        packed = _pack_latents(sample).to(dtype=dtype)

        # Qwen-Image's transformer takes raw sigma as the timestep
        # input (not sigma * 1000 like SD3).
        if sigma.dim() == 0:
            timestep = sigma.unsqueeze(0).expand(batch_size).to(device, dtype=dtype)
        elif sigma.shape[0] != batch_size:
            timestep = sigma.expand(batch_size).to(device, dtype=dtype)
        else:
            timestep = sigma.to(device, dtype=dtype)

        # The transformer needs the per-sample latent grid shape so it
        # can rebuild positional embeddings; format is
        # ``[[(frames, H/(vae_scale_factor*2), W/(vae_scale_factor*2))]] * B``.
        # Here ``latent_h`` / ``latent_w`` ARE already in the post-VAE
        # spatial grid, so the patchify divisor is just 2.
        img_shapes = [[(1, latent_h // 2, latent_w // 2)]] * batch_size

        # Distilled-guidance scalar — embedded by the transformer when
        # ``guidance_embeds=True`` is set on its config (set by some
        # Qwen-Image variants only). Independent of CFG guidance_scale.
        guidance = None
        if getattr(model.transformer.config, "guidance_embeds", False):
            guidance_value = guidance_scale if distilled_guidance_scale is None else float(distilled_guidance_scale)
            guidance = torch.tensor([guidance_value], device=device, dtype=torch.float32).expand(batch_size)

        # Per-sample true text lengths — the RoPE builder slices its text
        # frequency table by ``max(txt_seq_lens)`` (required positionally by
        # the installed diffusers; passing only the attention mask raises
        # ``max(None)`` TypeError — LIN-382 qwen probe-e). The embeds must be
        # trimmed to this slice's true max first: replay microbatches carry
        # the BATCH-wide pad width (e.g. 18) while their own max true length
        # may be shorter (12) — diffusers applies RoPE over the full tensor
        # width and the freq slice over max(txt_seq_lens), so a width
        # mismatch hard-crashes in apply_rotary_emb_qwen (probe-f).
        true_lens = prompt_embeds_mask.sum(dim=1).to(torch.long)
        max_true = int(true_lens.max().item())
        if prompt_embeds.shape[1] > max_true:
            prompt_embeds = prompt_embeds[:, :max_true]
            prompt_embeds_mask = prompt_embeds_mask[:, :max_true]
        txt_seq_lens = true_lens.tolist()

        noise_pred_packed = model.transformer(
            hidden_states=packed,
            timestep=timestep,
            guidance=guidance,
            encoder_hidden_states_mask=prompt_embeds_mask,
            encoder_hidden_states=prompt_embeds,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            return_dict=False,
        )[0]

        if guidance_scale > 1.0:
            neg = conditions.negative_text
            if neg is not None and neg.embeds is not None:
                negative_prompt_embeds = neg.embeds
                negative_prompt_embeds_mask = neg.attn_mask
                if negative_prompt_embeds_mask is None:
                    raise ValueError("QwenImageDiffusionStep.predict_noise: conditions.negative_text.attn_mask is None")
                neg_true = negative_prompt_embeds_mask.sum(dim=1).to(torch.long)
                neg_max = int(neg_true.max().item())
                if negative_prompt_embeds.shape[1] > neg_max:
                    negative_prompt_embeds = negative_prompt_embeds[:, :neg_max]
                    negative_prompt_embeds_mask = negative_prompt_embeds_mask[:, :neg_max]
                negative_txt_seq_lens = neg_true.tolist()
                negative_noise_pred_packed = model.transformer(
                    hidden_states=packed,
                    timestep=timestep,
                    guidance=guidance,
                    encoder_hidden_states_mask=negative_prompt_embeds_mask,
                    encoder_hidden_states=negative_prompt_embeds,
                    img_shapes=img_shapes,
                    txt_seq_lens=negative_txt_seq_lens,
                    return_dict=False,
                )[0]
                # Combined-CFG with norm correction. Spec: keep the per-token
                # L2 norm of the conditional prediction after CFG blending.
                comb = negative_noise_pred_packed + guidance_scale * (noise_pred_packed - negative_noise_pred_packed)
                cond_norm = torch.norm(noise_pred_packed, dim=-1, keepdim=True)
                comb_norm = torch.norm(comb, dim=-1, keepdim=True)
                noise_pred_packed = comb * (cond_norm / comb_norm)

        return _unpack_latents(noise_pred_packed, latent_h=latent_h, latent_w=latent_w)

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

        Returns ``(prev_sample, log_prob, prev_sample_mean)``. Operates
        on unpacked ``[B, C, H, W]`` tensors (the strategy is shape-
        agnostic).
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
        model: QwenImageBundle,
        conditions: QwenImageConditions,
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
        latent_h: int = 0,
        latent_w: int = 0,
        distilled_guidance_scale: Optional[float] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run model forward + SDE transition. End-to-end one diffusion step."""
        if latent_h <= 0 or latent_w <= 0:
            # Recover from sample shape — diffuse/replay always pass both
            # explicitly, but defaulting here keeps unit tests that hand-
            # roll ``[B, C, H, W]`` simple.
            latent_h = int(sample.shape[-2])
            latent_w = int(sample.shape[-1])
        noise_pred = self.predict_noise(
            model,
            sample,
            sigma,
            conditions,
            guidance_scale=guidance_scale,
            latent_h=latent_h,
            latent_w=latent_w,
            distilled_guidance_scale=distilled_guidance_scale,
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
        model: QwenImageBundle,
        conditions: QwenImageConditions,
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
        latent_h: int = 0,
        latent_w: int = 0,
        distilled_guidance_scale: Optional[float] = None,
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
            latent_h=latent_h,
            latent_w=latent_w,
            distilled_guidance_scale=distilled_guidance_scale,
        )


class QwenImageDiffusionStage(DiffusionRunner[QwenImageBundle, QwenImageConditions]):
    """Qwen-Image rollout-level diffusion stage.

    Owns the SDE ``strategy`` (stateful strategies like ``DPM2Strategy``
    require a stable instance across the loop), the bundle, the kernel,
    and the precision policy. The kernel is stateless and is invoked
    per-step with the strategy + the per-call ``latent_h`` / ``latent_w``
    that pin the packed-latent geometry.

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
    when HF auto-discovery yields nothing — diffusers'
    ``QwenImageTransformer2DModel`` block class is
    ``QwenImageTransformerBlock``.
    """

    _no_split_modules: ClassVar[Tuple[str, ...]] = ("QwenImageTransformerBlock",)

    # Qwen-Image t2i uses an 8× VAE downsample and 16 latent channels
    # in the post-VAE grid. The model bundle's ``transformer.config``
    # carries the authoritative count via ``in_channels // 4`` (the
    # packed-latent format multiplies by 4); we default to that and let
    # callers override via the stage constructor.
    DEFAULT_VAE_SCALE_FACTOR: ClassVar[int] = 8

    def __init__(
        self,
        *,
        model: QwenImageBundle,
        step: QwenImageDiffusionStep,
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
            # Read from the transformer config: in_channels is the
            # packed-input dim (C * 4), so the post-VAE channel count is
            # in_channels // 4. Falls back to 16 if the attr is missing.
            tx_cfg = getattr(model.transformer, "config", None)
            in_channels = getattr(tx_cfg, "in_channels", 64) if tx_cfg is not None else 64
            latent_channels = int(in_channels) // 4
        self.latent_channels = int(latent_channels)

    def _latent_spec(
        self,
        conditions: QwenImageConditions,
        params: DiffusionSamplingParams,
    ) -> DiffusionLatentSpec:
        if conditions.text is None or conditions.text.embeds is None:
            raise ValueError("QwenImageDiffusionStage: conditions.text.embeds is None")
        embeds = conditions.text.embeds
        latent_h = 2 * (int(params.height) // (int(self.vae_scale_factor) * 2))
        latent_w = 2 * (int(params.width) // (int(self.vae_scale_factor) * 2))
        return DiffusionLatentSpec(
            device=embeds.device,
            batch_size=int(embeds.shape[0]),
            shape=(int(self.latent_channels), latent_h, latent_w),
        )

    def _step_kwargs(
        self,
        conditions: QwenImageConditions,
        params: DiffusionSamplingParams,
        *,
        sample: torch.Tensor,
        step_index: int,
        num_steps: int,
        mode: str,
        state: Any,
    ) -> Mapping[str, Any]:
        del conditions, step_index, num_steps, mode, state
        return {
            "latent_h": int(sample.shape[-2]),
            "latent_w": int(sample.shape[-1]),
            "distilled_guidance_scale": params.distilled_guidance_scale,
        }

    def _validate_replay_segment(self, segment: LatentSegment) -> None:
        super()._validate_replay_segment(segment)
        if segment.latents.ndim != 5:
            raise ValueError(
                f"QwenImageDiffusionStage.replay: expected latents [B, K, C, H, W], got {tuple(segment.latents.shape)}"
            )


__all__ = [
    "QwenImageDiffusionStage",
    "QwenImageDiffusionStep",
    "_pack_latents",
    "_unpack_latents",
]
