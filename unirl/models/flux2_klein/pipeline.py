"""Flux2KleinPipeline — ``Sample → Sample`` end-to-end for FLUX.2-klein-9B."""

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
    """FLUX.2-klein-9B generate pipeline (t2i / image-edit): ``Sample → Sample``."""

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
        """Build the Klein-specific schedule policy."""
        return build_flux2_klein_schedule_policy(self.shift)

    @classmethod
    def latent_shape(cls, *, model_config: Any, sampling_spec: Any) -> tuple:
        """Per-sample patchified latent shape ``(C_pack=128, H_pat, W_pat)``"""
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
        """Build the full pipeline from a config."""
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
        """Encode prompts (+ optional CFG negatives) into ``Flux2KleinConditions``."""
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
        """Run FLUX.2-klein-9B t2i/edit end-to-end, filling the frontier (pre-forked) gen Part."""
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
            if len(source_image) != len(texts.texts):
                raise ValueError(
                    f"Flux2KleinPipeline.generate: image count {len(source_image)} != text count {len(texts.texts)}"
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
