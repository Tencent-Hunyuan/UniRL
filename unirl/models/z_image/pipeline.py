"""ZImagePipeline — ``Sample → Sample`` end-to-end for Z-Image.

Implements the four-tier flow::

    Texts ──text_embed──▶ ZImageConditions ──diffuse──▶ LatentSegment
                                                            │
                                                            ▼
                                                        vae_decode
                                                            │
                                                            ▼
                                                          Images

Hydra constructs a pipeline via
``ZImagePipeline.from_config(ZImagePipelineConfig)`` (see ``config.py``);
``from_config`` loads the :class:`ZImageBundle` then constructs the four
stages with the precision policy from the config.

σ schedule contract
-------------------
The hosting engine (``TrainsideRolloutEngine``) pins the σ schedule onto the gen
Part's ``DiffusionSamplingParams.sigmas`` BEFORE calling ``generate(sample)``;
this pipeline reads ``params.sigmas`` and uses it
verbatim. Both Z-Image variants' ``scheduler/scheduler_config.json`` declare
``use_dynamic_shifting: false`` (the diffusers ``ZImagePipeline`` computes a
Flux-style ``mu`` but ``FlowMatchEulerDiscreteScheduler`` discards it on the
static branch), so this pipeline is **static-shift** (unlike Qwen-Image /
Flux.2-Klein, which are dynamic-shift). The shift value differs by variant —
base ``Z-Image`` uses ``6.0``, ``Z-Image-Turbo`` uses ``3.0``;
:meth:`build_schedule_policy` pins that posture via
``FlowMatchSchedulePolicy.static_only(self.shift)``.

Base vs Turbo is purely a config difference (same architecture): base runs
with CFG (``guidance_scale > 0`` + a negative prompt), Turbo runs CFG-free
(``guidance_scale = 0``). The recipe sets the variant-specific values.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

from unirl.models.types.pipeline import Pipeline
from unirl.sde.kernels import FlowSDEStrategy, StepStrategy
from unirl.types.noise_recipe import NoiseRecipe
from unirl.types.primitives import Texts
from unirl.types.sample import Sample
from unirl.types.sampling import DiffusionSamplingParams

from .bundle import ZImageBundle
from .conditions import ZImageConditions
from .config import ZImagePipelineConfig
from .diffusion import ZImageDiffusionStage, ZImageDiffusionStep
from .text_embed import ZImageTextEmbedStage
from .vae import ZImageVAEDecodeStage


class ZImagePipeline(Pipeline):
    """Z-Image generate pipeline: ``Sample → Sample``.

    Consumes a request ``Sample`` whose frontier Part is a pre-forked diffusion gen
    shell carrying ``DiffusionSamplingParams`` (with ``sigmas`` pinned by the
    hosting engine). Reads the prompt via ``sample.conditioning()`` and fills the
    frontier Part:

    - ``segment: LatentSegment`` — the denoising trajectory.
    - ``primitives["image"]: Images`` — the decoded images.

    ``Part.conditions`` carries the encoded conditions for trainer-side replay (the train stack re-types them via ``conditions_cls.from_dict``). User-supplied negatives are
    deferred; CFG uses a synthesized empty negative.
    """

    def __init__(
        self,
        *,
        bundle: ZImageBundle,
        text_embed: Optional[ZImageTextEmbedStage] = None,
        diffusion: Optional[ZImageDiffusionStage] = None,
        vae_decode: Optional[ZImageVAEDecodeStage] = None,
        strategy: Optional[StepStrategy] = None,
        shift: float = 6.0,
        autocast_precision: str = "bf16",
        trajectory_precision: str = "fp16",
        logprob_precision: str = "fp32",
        max_sequence_length: int = 512,
        batch_replay_steps: bool = False,
    ) -> None:
        super().__init__()
        self.bundle = bundle
        self.text_embed = (
            text_embed
            if text_embed is not None
            else ZImageTextEmbedStage(bundle, max_sequence_length=max_sequence_length)
        )
        if diffusion is None:
            diffusion = ZImageDiffusionStage(
                model=bundle,
                step=ZImageDiffusionStep(),
                strategy=strategy if strategy is not None else FlowSDEStrategy(),
                autocast_precision=autocast_precision,
                trajectory_precision=trajectory_precision,
                logprob_precision=logprob_precision,
                batch_replay_steps=batch_replay_steps,
            )
        self.diffusion = diffusion
        self.vae_decode = vae_decode if vae_decode is not None else ZImageVAEDecodeStage(bundle)
        self.shift = shift

    @classmethod
    def latent_shape(cls, *, model_config: Any, sampling_spec: Any) -> tuple:
        """Per-sample latent shape ``(C, H_lat, W_lat)`` for driver-side
        noise pre-computation. Z-Image: 16-channel ``AutoencoderKL``, 8×
        spatial downsample with the patchify-2×2 rounding
        (``latent_h = 2 * (H // 16)``)."""
        height = int(sampling_spec.height)
        width = int(sampling_spec.width)
        vae_scale_factor = 8
        latent_h = 2 * (height // (vae_scale_factor * 2))
        latent_w = 2 * (width // (vae_scale_factor * 2))
        return (16, latent_h, latent_w)

    @classmethod
    def from_config(
        cls,
        config: ZImagePipelineConfig,
        *,
        strategy: Optional[StepStrategy] = None,
    ) -> "ZImagePipeline":
        """Build the full pipeline from a config.

        ``strategy`` is the SDE step strategy. Defaults to
        :class:`FlowSDEStrategy`; callers running GRPO with a different SDE
        family (Dance / CPS / DPM2) pass an explicit strategy.
        """
        bundle = ZImageBundle.from_config(config)
        text_embed = ZImageTextEmbedStage(bundle, max_sequence_length=config.max_sequence_length)
        step = ZImageDiffusionStep()
        diffusion = ZImageDiffusionStage(
            model=bundle,
            step=step,
            strategy=strategy if strategy is not None else FlowSDEStrategy(),
            autocast_precision=config.autocast_precision,
            trajectory_precision=config.trajectory_precision,
            logprob_precision=config.logprob_precision,
            batch_replay_steps=config.batch_replay_steps,
        )
        vae_decode = ZImageVAEDecodeStage(bundle)
        return cls(
            bundle=bundle,
            text_embed=text_embed,
            diffusion=diffusion,
            vae_decode=vae_decode,
            shift=float(config.shift),
        )

    def build_schedule_policy(self):
        """Static-shift FlowMatch σ policy (Z-Image uses no dynamic shift).

        Both Z-Image variants' ``scheduler/scheduler_config.json`` declare
        ``use_dynamic_shifting: false`` (base ``shift: 6.0``, Turbo
        ``shift: 3.0``): the upstream diffusers ``ZImagePipeline`` still
        computes a Flux-style ``mu``, but ``FlowMatchEulerDiscreteScheduler``
        ignores it on the static branch and applies
        ``shift·t / (1 + (shift-1)·t)``. Returning an explicit ``static_only``
        policy built from ``self.shift`` pins that posture regardless of whether
        ``pretrained_path`` is an HF repo id or a local mount (so a checkpoint
        shipping a stray dynamic ``scheduler_config.json`` can't silently flip
        σ and drift the GRPO ratio). Mirrors ``BagelPipeline.build_schedule_policy``.
        """
        from unirl.sde.runtime import FlowMatchSchedulePolicy

        return FlowMatchSchedulePolicy.static_only(float(self.shift))

    def build_conditions(
        self,
        texts: Texts,
        *,
        negatives: Optional[Texts] = None,
        guidance_scale: float = 1.0,
    ) -> ZImageConditions:
        """Encode prompts (+ optional CFG negatives) into ``ZImageConditions``.

        CFG empty negative: Z-Image upstream (diffusers ``ZImagePipeline``
        ``encode_prompt``) defaults to ``""`` (empty string) when CFG is
        enabled and no negative is passed. Z-Image gates CFG on
        ``guidance_scale > 0`` (Turbo runs with 0 → CFG off). The Qwen3
        chat template tokenizes ``""`` cleanly, so no ``" "`` workaround is
        needed (unlike Qwen-Image).
        """
        if negatives is not None and len(negatives.texts) != len(texts.texts):
            raise ValueError(
                f"ZImagePipeline.build_conditions: negative_text length "
                f"{len(negatives.texts)} != text length {len(texts.texts)}"
            )
        text_cond = self.text_embed.embed(texts)
        if negatives is None and float(guidance_scale) > 0.0:
            negatives = Texts(texts=[""] * len(texts.texts))
        negative_text_cond = self.text_embed.embed(negatives) if negatives is not None else None
        return ZImageConditions(text=text_cond, negative_text=negative_text_cond)

    def generate(self, sample: Sample) -> Sample:
        """Run Z-Image t2i end-to-end, filling the frontier (pre-forked) gen Part.

        Requires σ to be pinned onto the gen part's ``DiffusionSamplingParams.sigmas``
        by the hosting engine before the call; see the σ ownership note in
        ``unirl.models.types.pipeline``.
        """
        frontier = sample.parts[-1]
        params = frontier.sampling_params
        if not isinstance(params, DiffusionSamplingParams):
            raise TypeError(
                f"ZImagePipeline.generate: frontier gen Part must carry DiffusionSamplingParams, "
                f"got {type(params).__name__ if params is not None else 'None'}"
            )
        if params.sigmas is None:
            raise ValueError(
                "ZImagePipeline.generate: gen part sampling_params.sigmas is None. The hosting "
                "engine must pin σ before invoking pipeline.generate; see the σ ownership note "
                "in unirl.models.types.pipeline."
            )

        conditioning = sample.conditioning()
        texts = conditioning[0] if conditioning else None
        if not isinstance(texts, Texts):
            raise TypeError(
                f"ZImagePipeline.generate: expected a Texts prompt from sample.conditioning()[0], "
                f"got {type(texts).__name__ if texts is not None else 'None'}"
            )

        if bool(params.init_same_noise) and not params.noise_group_ids:
            params = dataclasses.replace(params, noise_group_ids=list(frontier.group_ids))

        z_conds = self.build_conditions(texts, guidance_scale=float(params.guidance_scale))
        schedule = params.sigmas.to(self.bundle.device)

        initial_latents = NoiseRecipe.from_sample(sample).resolve()

        latent_seg = self.diffusion.diffuse(z_conds, schedule=schedule, params=params, initial_latents=initial_latents)
        images = self.vae_decode.decode(latent_seg)

        filled = frontier.fill(segment=latent_seg, primitives={"image": images}, conditions=z_conds.to_dict())
        return Sample(parts=[*sample.parts[:-1], filled], reward_compute_s=sample.reward_compute_s)


__all__ = ["ZImagePipeline"]
