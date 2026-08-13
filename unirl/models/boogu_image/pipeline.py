"""BooguImagePipeline — ``Sample → Sample`` end-to-end for Boogu-Image."""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

from unirl.models.types.pipeline import Pipeline
from unirl.sde.kernels import FlowSDEStrategy, StepStrategy
from unirl.types.noise_recipe import NoiseRecipe
from unirl.types.primitives import Texts
from unirl.types.sample import Sample
from unirl.types.sampling import DiffusionSamplingParams

from .bundle import BooguImageBundle
from .conditions import BooguImageConditions
from .config import BOOGU_IMAGE_BASE_STATIC_SHIFT, BooguImagePipelineConfig
from .diffusion import BooguImageDiffusionStage, BooguImageDiffusionStep
from .text_embed import BooguImageTextEmbedStage
from .vae import BooguImageVAEDecodeStage


class BooguImagePipeline(Pipeline):
    """Boogu-Image generate pipeline: ``Sample → Sample``."""

    def __init__(
        self,
        *,
        bundle: BooguImageBundle,
        text_embed: Optional[BooguImageTextEmbedStage] = None,
        diffusion: Optional[BooguImageDiffusionStage] = None,
        vae_decode: Optional[BooguImageVAEDecodeStage] = None,
        strategy: Optional[StepStrategy] = None,
        shift: float = BOOGU_IMAGE_BASE_STATIC_SHIFT,
        autocast_precision: str = "bf16",
        trajectory_precision: str = "fp16",
        logprob_precision: str = "fp32",
        max_sequence_length: int = 1280,
    ) -> None:
        super().__init__()
        self.bundle = bundle
        self.text_embed = (
            text_embed
            if text_embed is not None
            else BooguImageTextEmbedStage(bundle, max_sequence_length=max_sequence_length)
        )
        if diffusion is None:
            diffusion = BooguImageDiffusionStage(
                model=bundle,
                step=BooguImageDiffusionStep(),
                strategy=strategy if strategy is not None else FlowSDEStrategy(),
                autocast_precision=autocast_precision,
                trajectory_precision=trajectory_precision,
                logprob_precision=logprob_precision,
            )
        self.diffusion = diffusion
        self.vae_decode = vae_decode if vae_decode is not None else BooguImageVAEDecodeStage(bundle)
        self.shift = shift

    @classmethod
    def latent_shape(cls, *, model_config: Any, sampling_spec: Any) -> tuple:
        """Per-sample latent shape ``(C, H_lat, W_lat)`` for driver-side noise pre-computation."""
        height = int(sampling_spec.height)
        width = int(sampling_spec.width)
        vae_scale_factor = 8
        latent_h = 2 * (height // (vae_scale_factor * 2))
        latent_w = 2 * (width // (vae_scale_factor * 2))
        return (16, latent_h, latent_w)

    @classmethod
    def from_config(
        cls,
        config: BooguImagePipelineConfig,
        *,
        strategy: Optional[StepStrategy] = None,
    ) -> "BooguImagePipeline":
        """Build the full pipeline from a config."""
        bundle = BooguImageBundle.from_config(config)
        text_embed = BooguImageTextEmbedStage(bundle, max_sequence_length=config.max_sequence_length)
        step = BooguImageDiffusionStep()
        diffusion = BooguImageDiffusionStage(
            model=bundle,
            step=step,
            strategy=strategy if strategy is not None else FlowSDEStrategy(),
            autocast_precision=config.autocast_precision,
            trajectory_precision=config.trajectory_precision,
            logprob_precision=config.logprob_precision,
        )
        vae_decode = BooguImageVAEDecodeStage(bundle)
        return cls(
            bundle=bundle,
            text_embed=text_embed,
            diffusion=diffusion,
            vae_decode=vae_decode,
            shift=float(config.shift),
            max_sequence_length=config.max_sequence_length,
        )

    def build_schedule_policy(self):
        """Static-shift FlowMatch σ policy for the released Boogu-Image-0.1."""
        from unirl.sde.runtime import FlowMatchSchedulePolicy

        return FlowMatchSchedulePolicy.static_only(float(self.shift))

    def build_conditions(
        self,
        texts: Texts,
        *,
        negatives: Optional[Texts] = None,
        guidance_scale: float = 1.0,
    ) -> BooguImageConditions:
        """Encode prompts (+ optional CFG negatives) into ``BooguImageConditions``."""
        if negatives is not None and len(negatives.texts) != len(texts.texts):
            raise ValueError(
                f"BooguImagePipeline.build_conditions: negative_text length "
                f"{len(negatives.texts)} != text length {len(texts.texts)}"
            )
        text_cond = self.text_embed.embed(texts)
        if negatives is None and float(guidance_scale) > 1.0:
            negatives = Texts(texts=[""] * len(texts.texts))
        negative_text_cond = self.text_embed.embed(negatives) if negatives is not None else None
        return BooguImageConditions(text=text_cond, negative_text=negative_text_cond)

    def generate(self, sample: Sample) -> Sample:
        """Run Boogu-Image t2i end-to-end, filling the frontier (pre-forked) gen Part."""
        frontier = sample.parts[-1]
        params = frontier.sampling_params
        if not isinstance(params, DiffusionSamplingParams):
            raise TypeError(
                f"BooguImagePipeline.generate: frontier gen Part must carry DiffusionSamplingParams, "
                f"got {type(params).__name__ if params is not None else 'None'}"
            )
        if params.sigmas is None:
            raise ValueError(
                "BooguImagePipeline.generate: gen part sampling_params.sigmas is None. The hosting "
                "engine must pin σ before invoking pipeline.generate; see the σ ownership note "
                "in unirl.models.types.pipeline."
            )

        conditioning = sample.conditioning()
        texts = conditioning[0] if conditioning else None
        if not isinstance(texts, Texts):
            raise TypeError(
                f"BooguImagePipeline.generate: expected a Texts prompt from sample.conditioning()[0], "
                f"got {type(texts).__name__ if texts is not None else 'None'}"
            )

        if bool(params.init_same_noise) and not params.noise_group_ids:
            params = dataclasses.replace(params, noise_group_ids=list(frontier.group_ids))

        conds = self.build_conditions(texts, guidance_scale=float(params.guidance_scale))
        schedule = params.sigmas.to(self.bundle.device)

        initial_latents = NoiseRecipe.from_sample(sample).resolve()

        latent_seg = self.diffusion.diffuse(conds, schedule=schedule, params=params, initial_latents=initial_latents)
        images = self.vae_decode.decode(latent_seg)

        filled = frontier.fill(segment=latent_seg, primitives={"image": images}, conditions=conds.to_dict())
        return Sample(parts=[*sample.parts[:-1], filled], reward_compute_s=sample.reward_compute_s)


__all__ = ["BooguImagePipeline"]
