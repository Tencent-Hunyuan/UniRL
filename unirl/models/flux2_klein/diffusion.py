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

from contextlib import nullcontext
from dataclasses import dataclass
from typing import ClassVar, List, Optional, Set, Tuple

import torch

from unirl.models.types.diffusion import DiffusionStage, DiffusionStep
from unirl.models.types.replay_result import ReplayResult
from unirl.sde.kernels import StepStrategy
from unirl.types.segments.latent import LatentSegment
from unirl.types.trajectory_store import compute_trajectory_positions
from unirl.utils.dtypes import parse_torch_dtype

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


class Flux2KleinDiffusionStep(DiffusionStep[Flux2KleinBundle, Flux2KleinConditions]):
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


class Flux2KleinDiffusionStage(DiffusionStage[Flux2KleinConditions]):
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
        self.model = model
        self.step = step
        self.strategy = strategy
        self.autocast_dtype = parse_torch_dtype(autocast_precision, field_name="autocast_precision")
        self.trajectory_dtype = parse_torch_dtype(trajectory_precision, field_name="trajectory_precision")
        self.logprob_dtype = parse_torch_dtype(logprob_precision, field_name="logprob_precision")
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

    def diffuse(
        self,
        conditions: Flux2KleinConditions,
        *,
        schedule: torch.Tensor,
        params: Flux2KleinDiffusionParams,
        initial_latents: Optional[torch.Tensor] = None,
    ) -> LatentSegment:
        """Run full FLUX.2-klein sampling. Returns a ``LatentSegment``.

        Segment latents stay in patchified spatial form
        ``[B, K, 128, H_pat, W_pat]``. The driver may pre-ship
        ``initial_latents`` (in the same patchified spatial form) via
        ``req.request_conditions['initial_latents']``; when absent we
        sample fresh Gaussian noise.
        """
        from unirl.sde.noise import generate_latents

        if conditions.text is None or conditions.text.embeds is None:
            raise ValueError("Flux2KleinDiffusionStage.diffuse: conditions.text.embeds is None")
        prompt_embeds = conditions.text.embeds
        device = prompt_embeds.device
        batch_size = int(prompt_embeds.shape[0])
        T = int(params.num_inference_steps)
        if int(schedule.shape[0]) != T + 1:
            raise ValueError(f"Flux2KleinDiffusionStage.diffuse: schedule length {schedule.shape[0]} != T+1={T + 1}")
        schedule = schedule.to(device)
        self.strategy.init_schedule(schedule)

        expected_latent_shape = self._patchified_shape(int(params.height), int(params.width))
        if initial_latents is not None:
            if int(initial_latents.shape[0]) != batch_size:
                raise ValueError(
                    f"Flux2KleinDiffusionStage.diffuse: initial_latents.shape[0]="
                    f"{int(initial_latents.shape[0])} != batch_size={batch_size}."
                )
            if tuple(initial_latents.shape[1:]) != expected_latent_shape:
                raise ValueError(
                    f"Flux2KleinDiffusionStage.diffuse: initial_latents.shape[1:]="
                    f"{tuple(initial_latents.shape[1:])} != expected {expected_latent_shape} "
                    f"for height={int(params.height)}, width={int(params.width)}."
                )
            latents = initial_latents.to(device=device, dtype=self.trajectory_dtype)
        else:
            latents = generate_latents(
                batch_size=batch_size,
                latent_shape=expected_latent_shape,
                device=device,
                dtype=self.trajectory_dtype,
                init_same_noise=bool(params.init_same_noise),
                samples_per_prompt=int(params.samples_per_prompt),
                noise_group_ids=params.noise_group_ids,
                base_seed=int(params.seed),
            )

        sde_set: Set[int] = set(int(i) for i in (params.sde_indices or []))
        sde_sorted: List[int] = sorted(sde_set)

        needed: Set[int] = set(compute_trajectory_positions(sde_set, T))
        needed.add(T)

        stored_pairs: List[Tuple[int, torch.Tensor]] = []
        if 0 in needed:
            stored_pairs.append((0, latents.detach().clone()))
        sde_logp_list: List[torch.Tensor] = []

        autocast_ctx = (
            torch.autocast("cuda", self.autocast_dtype)
            if device.type == "cuda" and self.autocast_dtype in (torch.float16, torch.bfloat16)
            else nullcontext()
        )
        sigma_max = schedule[1].float() if int(schedule.shape[0]) > 1 else torch.tensor(0.99)

        # Klein's transformer keeps `.eval()` mode during sampling
        # (matches legacy Flux2Sampler.sample). Caller is responsible
        # for restoring `.train()` after rollout finishes.
        self.model.transformer.eval()

        for i in range(T):
            sigma = schedule[i].to(device)
            sigma_next = schedule[i + 1].to(device)
            step_eta = float(params.eta) if i in sde_set else 0.0

            with torch.no_grad(), autocast_ctx:
                new_latents, log_prob, _ = self.step.step_with_logp(
                    self.model,
                    conditions,
                    strategy=self.strategy,
                    sample=latents,
                    sigma=sigma,
                    sigma_next=sigma_next,
                    guidance_scale=float(params.guidance_scale),
                    eta=step_eta,
                    sigma_max=sigma_max,
                    step_index=i,
                )
            latents = new_latents.to(dtype=self.trajectory_dtype)

            if (i + 1) in needed:
                stored_pairs.append((i + 1, latents.detach().clone()))

            if log_prob is not None:
                sde_logp_list.append(log_prob.to(dtype=self.logprob_dtype))

        positions_collected = [p for p, _ in stored_pairs]
        latents_stacked = torch.stack([t for _, t in stored_pairs], dim=1)  # [B, K, C, H_pat, W_pat]

        sde_logp = torch.stack(sde_logp_list, dim=1) if sde_logp_list else None
        sde_indices_tensor = torch.tensor(sde_sorted, dtype=torch.long, device=device) if sde_sorted else None

        indices_tensor = torch.tensor(positions_collected, dtype=torch.long, device=device)

        return LatentSegment(
            latents=latents_stacked,
            sigmas=schedule,
            indices=indices_tensor,
            sde_logp=sde_logp,
            sde_indices=sde_indices_tensor,
        )

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------

    def replay(
        self,
        conditions: Flux2KleinConditions,
        *,
        segment: LatentSegment,
        params: Flux2KleinDiffusionParams,
        step_indices: Optional[List[int]] = None,
    ) -> ReplayResult:
        """Segment-based log-prob replay over the rollout's SDE transitions.

        Mirrors :class:`SD3DiffusionStage.replay`. Klein-specific
        difference: the transformer is held in ``.eval()`` mode for the
        forward pass (matches legacy
        ``Flux2Sampler.compute_log_prob_for_training`` Klein branch).
        Caller manages the outer ``.train()`` / ``.eval()`` mode and
        grad scope; this method only manages the autocast scope and
        the per-step eval flip.
        """
        if segment.sde_indices is None or segment.latents is None:
            raise ValueError("Flux2KleinDiffusionStage.replay: segment.sde_indices / latents missing")
        if segment.sigmas is None:
            raise ValueError("Flux2KleinDiffusionStage.replay: segment.sigmas missing")
        if segment.latents.ndim != 5:
            raise ValueError(
                f"Flux2KleinDiffusionStage.replay: expected latents [B, K, C, H_pat, W_pat], "
                f"got {tuple(segment.latents.shape)}"
            )

        sde_set = set(int(i) for i in segment.sde_indices.tolist())
        target = (
            [int(i) for i in step_indices]
            if step_indices is not None
            else [int(i) for i in segment.sde_indices.tolist()]
        )
        bad = [i for i in target if i not in sde_set]
        if bad:
            raise ValueError(
                f"Flux2KleinDiffusionStage.replay: step_indices {bad} not in segment.sde_indices={sorted(sde_set)}"
            )

        device = segment.latents.device
        sigmas = segment.sigmas.to(device)
        sigma_max = sigmas[1].float() if int(sigmas.shape[0]) > 1 else torch.tensor(0.99)

        autocast_ctx = (
            torch.autocast("cuda", self.autocast_dtype)
            if device.type == "cuda" and self.autocast_dtype in (torch.float16, torch.bfloat16)
            else nullcontext()
        )

        # Klein replay uses ``.eval()`` to match the legacy sampler.
        prior_training = self.model.transformer.training
        self.model.transformer.eval()
        try:
            log_probs: List[torch.Tensor] = []
            prev_sample_means: List[torch.Tensor] = []
            with autocast_ctx:
                for step_idx in target:
                    sigma = sigmas[step_idx].to(dtype=torch.float32)
                    sigma_next = sigmas[step_idx + 1].to(dtype=torch.float32)
                    sample = segment.latents_at(step_idx)
                    prev_sample = segment.latents_at(step_idx + 1)
                    _, log_prob, prev_mean = self.step.step_with_logp(
                        self.model,
                        conditions,
                        strategy=self.strategy,
                        sample=sample,
                        prev_sample=prev_sample,
                        sigma=sigma,
                        sigma_next=sigma_next,
                        guidance_scale=float(params.guidance_scale),
                        eta=float(params.eta),
                        sigma_max=sigma_max,
                        step_index=step_idx,
                    )
                    if log_prob is None:
                        raise RuntimeError(
                            f"Flux2KleinDiffusionStage.replay: strategy returned None "
                            f"log-prob at step_index={step_idx} (deterministic mode); "
                            f"replay requires a stochastic SDE strategy."
                        )
                    log_probs.append(log_prob)
                    if prev_mean is not None:
                        prev_sample_means.append(prev_mean)
        finally:
            if prior_training:
                self.model.transformer.train()

        log_probs_t = torch.stack(log_probs, dim=1).to(dtype=self.logprob_dtype)
        means_t = torch.stack(prev_sample_means, dim=1).to(dtype=self.trajectory_dtype) if prev_sample_means else None
        return ReplayResult(log_probs=log_probs_t, prev_sample_means=means_t)

    # ------------------------------------------------------------------
    # Single-step noise prediction (forward-process algorithms: DiffusionNFT et al.)
    # ------------------------------------------------------------------

    def predict_noise_at_step(
        self,
        conditions: Flux2KleinConditions,
        *,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        params: Flux2KleinDiffusionParams,
    ) -> torch.Tensor:
        """Single ``(xt, sigma)`` model forward — no scheduler iteration.

        ``sample`` is patchified ``[B, 128, H_pat, W_pat]``. Delegates
        to :meth:`Flux2KleinDiffusionStep.predict_noise`.
        """
        return self.step.predict_noise(
            self.model,
            sample,
            sigma,
            conditions,
            guidance_scale=float(params.guidance_scale),
        )

    # ------------------------------------------------------------------
    # Trainable surface for FSDPPolicy
    # ------------------------------------------------------------------

    def trainable_module(self) -> "torch.nn.Module":
        """Return the module the diffusion forward operates on.

        For FLUX.2-klein, that's the bundle's transformer
        (``Flux2Transformer2DModel``) — the FSDP wrap target. Aux
        modules (VAE, Qwen3 text encoder) are siblings on the bundle,
        never under the transformer.
        """
        return self.model.transformer


__all__ = [
    "Flux2KleinDiffusionParams",
    "Flux2KleinDiffusionStage",
    "Flux2KleinDiffusionStep",
]
