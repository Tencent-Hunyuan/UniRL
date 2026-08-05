"""Flux2KleinPipeline — ``Sample → Sample`` end-to-end for FLUX.2-klein-9B.

Implements the typed four-tier flow::

    Texts ──text_embed──▶ Flux2KleinConditions ──diffuse──▶ LatentSegment
                                                              │
                                                              ▼
                                                         vae_decode
                                                              │
                                                              ▼
                                                            Images

Hydra constructs a pipeline via
``Flux2KleinPipeline.from_config(Flux2KleinPipelineConfig)`` (see
``config.py``); ``from_config`` loads the :class:`Flux2KleinBundle`
then constructs the four stages with the precision policy from the
config.

σ schedule contract
-------------------
The hosting engine (``TrainsideRolloutEngine`` / ``SGLangDiffusionRolloutEngine``
/ ``VLLMOmniRolloutEngine``) pins the σ schedule onto the gen Part's
``DiffusionSamplingParams.sigmas`` BEFORE calling ``generate(sample)``; this
pipeline reads ``params.sigmas`` and uses it
verbatim.

FLUX.2-klein-specific override: μ depends on both ``image_seq_len`` AND
``num_inference_steps`` (the linear-interp
:func:`calculate_dynamic_mu` used by SD3 / Qwen-Image only depends on
``image_seq_len``). :meth:`build_schedule_policy` returns a custom
:class:`Flux2KleinSchedulePolicy` that overrides only
:meth:`FlowMatchSchedulePolicy.compute_mu` (the μ value); the shared
:meth:`FlowMatchSchedulePolicy.compute_sigma` builds the schedule for all
models, Klein included.
"""

from __future__ import annotations

import dataclasses as _dc
from typing import Any, Optional, Tuple

from unirl.models.types.pipeline import Pipeline
from unirl.sde.kernels import DanceSDEStrategy, StepStrategy
from unirl.types.noise_recipe import NoiseRecipe
from unirl.types.primitives import Images, Texts
from unirl.types.sample import Sample

from .bundle import Flux2KleinBundle
from .conditions import Flux2KleinConditions
from .config import Flux2KleinPipelineConfig
from .diffusion import (
    Flux2KleinDiffusionParams,
    Flux2KleinDiffusionStage,
    Flux2KleinDiffusionStep,
)
from .schedule import Flux2KleinSchedulePolicy, build_flux2_klein_schedule_policy
from .text_embed import Flux2KleinTextEmbedStage
from .vae import Flux2KleinVAEDecodeStage, Flux2KleinVAEEncodeStage


class Flux2KleinPipeline(Pipeline):
    """FLUX.2-klein-9B generate pipeline (t2i / image-edit): ``Sample → Sample``.

    Consumes a request ``Sample`` whose frontier Part is a pre-forked diffusion gen
    shell carrying ``DiffusionSamplingParams`` (with ``sigmas`` pinned by the
    hosting engine). Reads the prompt — and, for image-edit, the chained source
    image — via ``sample.conditioning()`` and fills the frontier Part:

    - ``segment: LatentSegment`` (patchified spatial shape
      ``[B, K, 128, H_pat, W_pat]``).
    - ``primitives["image"]: Images`` — the decoded images.

    ``Part.conditions`` carries the encoded conditions for trainer-side replay (the train stack re-types them via ``conditions_cls.from_dict``). User-supplied negatives are
    deferred; the canonical Klein recipe runs at ``guidance_scale=1.0`` with no
    negative branch, so CFG synthesizes an empty negative only when guidance > 1.
    """

    def __init__(
        self,
        *,
        bundle: Flux2KleinBundle,
        text_embed: Optional[Flux2KleinTextEmbedStage] = None,
        diffusion: Optional[Flux2KleinDiffusionStage] = None,
        vae_decode: Optional[Flux2KleinVAEDecodeStage] = None,
        strategy: Optional[StepStrategy] = None,
        shift: float = 1.0,
        autocast_precision: str = "bf16",
        trajectory_precision: str = "fp16",
        logprob_precision: str = "fp32",
        batch_replay_steps: bool = False,
        max_sequence_length: int = 512,
        qwen3_extraction_layers: Tuple[int, ...] = (9, 18, 27),
    ) -> None:
        super().__init__()
        self.bundle = bundle
        self.text_embed = (
            text_embed
            if text_embed is not None
            else Flux2KleinTextEmbedStage(
                bundle,
                max_sequence_length=max_sequence_length,
                extraction_layers=tuple(qwen3_extraction_layers),
            )
        )
        if diffusion is None:
            diffusion = Flux2KleinDiffusionStage(
                model=bundle,
                step=Flux2KleinDiffusionStep(),
                strategy=strategy if strategy is not None else DanceSDEStrategy(),
                autocast_precision=autocast_precision,
                trajectory_precision=trajectory_precision,
                logprob_precision=logprob_precision,
                batch_replay_steps=batch_replay_steps,
            )
        self.diffusion = diffusion
        self.vae_decode = vae_decode if vae_decode is not None else Flux2KleinVAEDecodeStage(bundle)
        self.vae_encode = Flux2KleinVAEEncodeStage(bundle)
        self.shift = shift

    def build_schedule_policy(self):
        """Build the Klein-specific schedule policy.

        FLUX.2-klein-9B was trained with an empirical-μ schedule that
        depends on **both** the packed image_seq_len AND the number of
        inference steps. The standard :class:`FlowMatchSchedulePolicy`
        only encodes the image_seq_len → μ mapping linearly
        (``calculate_dynamic_mu``), so we return a Klein-specific subclass
        that overrides :meth:`compute_mu` with the empirical formula. The
        σ application (base grid + diffusers time-shift) is the shared
        dynamic-shift path. ``time_shift_type`` must match the checkpoint's
        ``scheduler_config.json`` (FLUX.2 uses ``"exponential"``).
        """
        return build_flux2_klein_schedule_policy(self.shift)

    @classmethod
    def latent_shape(cls, *, model_config: Any, sampling_spec: Any) -> tuple:
        """Per-sample patchified latent shape ``(C_pack=128, H_pat, W_pat)``
        for driver-side noise pre-computation.

        FLUX.2-klein-9B: 32-channel post-VAE latents (``AutoencoderKLFlux2``),
        2×2 channel-packed for the transformer input (128 = 32 × 4),
        post-VAE spatial 8× downsample plus the patchify factor of 2.
        ``Flux2KleinDiffusionStage`` operates directly on the patchified
        shape ``[B, 128, H_pix/16, W_pix/16]``; the driver-shipped
        initial-noise tensor must match this geometry.
        """
        height = int(sampling_spec.height)
        width = int(sampling_spec.width)
        downsample = 8 * 2
        if height % downsample != 0 or width % downsample != 0:
            raise ValueError(
                f"Flux2KleinPipeline.latent_shape: height ({height}) and width "
                f"({width}) must be divisible by VAE×patchify downsample "
                f"({downsample})."
            )
        return (128, height // downsample, width // downsample)

    @classmethod
    def from_config(
        cls,
        config: Flux2KleinPipelineConfig,
        *,
        strategy: Optional[StepStrategy] = None,
    ) -> "Flux2KleinPipeline":
        """Build the full pipeline from a config.

        ``strategy`` defaults to :class:`DanceSDEStrategy` — the
        canonical Klein training-script setting
        (``main_flux_bundle/reproduce_scripts/train_grpo_flux2_klein9b_sglang_multinode.sh``
        sets ``SDE_TYPE=dance``). Callers running an alternate SDE
        family pass an explicit strategy.
        """
        bundle = Flux2KleinBundle.from_config(config)
        text_embed = Flux2KleinTextEmbedStage(
            bundle,
            max_sequence_length=config.max_sequence_length,
            extraction_layers=tuple(config.qwen3_extraction_layers),
        )
        step = Flux2KleinDiffusionStep()
        diffusion = Flux2KleinDiffusionStage(
            model=bundle,
            step=step,
            strategy=strategy if strategy is not None else DanceSDEStrategy(),
            autocast_precision=config.autocast_precision,
            trajectory_precision=config.trajectory_precision,
            logprob_precision=config.logprob_precision,
            batch_replay_steps=config.batch_replay_steps,
        )
        vae_decode = Flux2KleinVAEDecodeStage(bundle)
        return cls(
            bundle=bundle,
            text_embed=text_embed,
            diffusion=diffusion,
            vae_decode=vae_decode,
            shift=float(config.shift),
        )

    def build_conditions(
        self,
        texts: Texts,
        *,
        negatives: Optional[Texts] = None,
        guidance_scale: float = 1.0,
    ) -> Flux2KleinConditions:
        """Encode prompts (+ optional CFG negatives) into ``Flux2KleinConditions``.

        CFG empty negative: Klein's canonical training-script setting is
        ``guidance_scale=1.0`` (the script literally hardcodes it; see
        ``main_flux_bundle/reproduce_scripts/train_grpo_flux2_klein9b_sglang_multinode.sh``).
        When CFG is OFF, no negative branch is needed and we leave
        ``negative_text=None`` so the transformer runs only the
        conditional forward. When a downstream user opts in to
        ``guidance_scale > 1`` without supplying ``negative_text``,
        default to an empty string (the Qwen3 chat-template tokenizer
        is robust to ``""``: no chat-template prefix is stripped, so
        the resulting embedding is well-defined).
        """
        if negatives is not None and len(negatives.texts) != len(texts.texts):
            raise ValueError(
                f"Flux2KleinPipeline.build_conditions: negative_text length "
                f"{len(negatives.texts)} != text length {len(texts.texts)}"
            )
        text_cond = self.text_embed.embed(texts)
        if negatives is None and float(guidance_scale) > 1.0:
            negatives = Texts(texts=[""] * len(texts.texts))
        negative_text_cond = self.text_embed.embed(negatives) if negatives is not None else None
        return Flux2KleinConditions(text=text_cond, negative_text=negative_text_cond)

    def generate(self, sample: Sample) -> Sample:
        """Run FLUX.2-klein-9B t2i/edit end-to-end, filling the frontier (pre-forked) gen Part.

        Requires σ to be pinned onto the gen part's ``DiffusionSamplingParams.sigmas``
        by the hosting engine before the call; see the σ ownership note in
        ``unirl.models.types.pipeline``.
        """
        frontier = sample.parts[-1]
        sampling = frontier.sampling_params
        if sampling is None:
            raise TypeError("Flux2KleinPipeline.generate: frontier gen Part must carry DiffusionSamplingParams")
        if getattr(sampling, "sigmas", None) is None:
            raise ValueError(
                "Flux2KleinPipeline.generate: gen part sampling_params.sigmas is None. The hosting "
                "engine must pin σ before invoking pipeline.generate; see the σ ownership note "
                "in unirl.models.types.pipeline."
            )

        conditioning = sample.conditioning()
        texts = conditioning[0] if conditioning else None
        if not isinstance(texts, Texts):
            raise TypeError(
                f"Flux2KleinPipeline.generate: expected a Texts prompt from sample.conditioning()[0], "
                f"got {type(texts).__name__ if texts is not None else 'None'}"
            )
        source_image = next((c for c in conditioning[1:] if isinstance(c, Images)), None)

        allowed = {f.name for f in _dc.fields(Flux2KleinDiffusionParams)}
        params_dict = {k: getattr(sampling, k) for k in allowed if hasattr(sampling, k)}
        params = Flux2KleinDiffusionParams(**params_dict)
        # Pass group IDs to the noise sampler when init_same_noise is enabled.
        if bool(params.init_same_noise) and not params.noise_group_ids:
            params = _dc.replace(params, noise_group_ids=list(frontier.group_ids))

        klein_conds = self.build_conditions(texts, guidance_scale=float(params.guidance_scale))
        if source_image is not None:
            if source_image.pixels is None or int(source_image.pixels.shape[0]) != len(texts.texts):
                raise ValueError(
                    f"Flux2KleinPipeline.generate: image count "
                    f"{None if source_image.pixels is None else int(source_image.pixels.shape[0])} "
                    f"!= text count {len(texts.texts)}"
                )
            image_tokens, image_ids = self.vae_encode.encode(
                source_image,
                height=int(params.height),
                width=int(params.width),
            )
            klein_conds.image_latent = image_tokens
            klein_conds.image_latent_ids = image_ids
        schedule = sampling.sigmas.to(self.bundle.device)

        initial_latents = NoiseRecipe.from_sample(sample).resolve()

        latent_seg = self.diffusion.diffuse(
            klein_conds, schedule=schedule, params=params, initial_latents=initial_latents
        )
        decoded = self.vae_decode.decode(latent_seg)

        filled = frontier.fill(segment=latent_seg, primitives={"image": decoded}, conditions=klein_conds.to_dict())
        return Sample(parts=[*sample.parts[:-1], filled], reward_compute_s=sample.reward_compute_s)


__all__ = ["Flux2KleinPipeline", "Flux2KleinSchedulePolicy", "build_flux2_klein_schedule_policy"]
