"""MiniMax-H3 t2va pipeline -- prompt -> joint video + stereo audio."""

from __future__ import annotations

from typing import Any, Tuple

from unirl.config.require import require
from unirl.sde.kernels import StepStrategy
from unirl.sde.runtime import FlowMatchSchedulePolicy
from unirl.types.noise_recipe import NoiseRecipe
from unirl.types.primitives import Texts
from unirl.types.sample import Sample

from .bundle import MiniMaxH3Bundle
from .conditions import MiniMaxH3Conditions
from .config import MINIMAX_H3_PATCH_SIZE, MiniMaxH3PipelineConfig
from .diffusion import MiniMaxH3DiffusionStage
from .packing import MiniMaxH3Geometry
from .text_embed import MiniMaxH3TextEmbedStage
from .vae import (
    MINIMAX_H3_AUDIO_SAMPLE_RATE,
    MiniMaxH3AudioDecodeStage,
    MiniMaxH3VideoDecodeStage,
)
from .vendor import MINIMAX_H3_AUDIO_CHANNELS, patchify_video_latents


class MiniMaxH3Pipeline:
    """Text -> video+audio via one packed-sequence denoising loop."""

    def __init__(
        self,
        *,
        bundle: MiniMaxH3Bundle,
        text_embed: MiniMaxH3TextEmbedStage,
        diffusion: MiniMaxH3DiffusionStage,
        video_decode: MiniMaxH3VideoDecodeStage,
        audio_decode: MiniMaxH3AudioDecodeStage,
        config: MiniMaxH3PipelineConfig,
    ) -> None:
        self.bundle = bundle
        self.text_embed = text_embed
        self.diffusion = diffusion
        self.video_decode = video_decode
        self.audio_decode = audio_decode
        self.config = config

    @classmethod
    def from_config(cls, config: MiniMaxH3PipelineConfig, strategy: StepStrategy) -> "MiniMaxH3Pipeline":
        return cls.from_bundle(MiniMaxH3Bundle.from_config(config), config=config, strategy=strategy)

    @classmethod
    def from_bundle(
        cls,
        bundle: MiniMaxH3Bundle,
        *,
        config: MiniMaxH3PipelineConfig,
        strategy: StepStrategy,
    ) -> "MiniMaxH3Pipeline":
        return cls(
            bundle=bundle,
            text_embed=MiniMaxH3TextEmbedStage(bundle),
            diffusion=MiniMaxH3DiffusionStage(
                bundle,
                strategy,
                audio_shift=config.audio_shift,
                audio_joint_sde=config.audio_joint_sde,
                autocast_precision=config.autocast_precision,
                trajectory_precision=config.trajectory_precision,
                logprob_precision=config.logprob_precision,
            ),
            video_decode=MiniMaxH3VideoDecodeStage(bundle),
            audio_decode=MiniMaxH3AudioDecodeStage(bundle),
            config=config,
        )

    def build_schedule_policy(self) -> FlowMatchSchedulePolicy:
        """The VIDEO sigma policy the hosting engine pins onto the Part.

        MiniMax-H3's grid is ``linspace(1, 0, N)`` put through
        ``shift*t / (1 + (shift-1)*t)`` -- exactly UniRL's *static* branch, so no
        bespoke policy subclass is needed (contrast LTX-2, which needed one for
        its constant-mu dynamic shift). The audio grid is the same function at
        ``audio_shift`` and is derived inside the diffusion stage rather than
        pinned, so there is only ever one schedule on the wire.
        """
        return FlowMatchSchedulePolicy.static_only(shift=float(self.config.video_shift))

    @classmethod
    def latent_shape(cls, *, model_config: Any, sampling_spec: Any) -> Tuple[int, ...]:
        """Per-sample UNPACKED video latent shape for the driver x_T recipe.

        Noise is authored 5D here and patchified into the transformer's row
        layout in :meth:`generate`, mirroring LTX-2.
        """
        del model_config  # geometry is fully determined by the request
        return MiniMaxH3Geometry.from_params(sampling_spec).latent_shape

    @staticmethod
    def audio_latent_shape(geometry: MiniMaxH3Geometry) -> Tuple[int, ...]:
        """Per-sample audio x_T shape: one row block per stereo channel."""
        return (geometry.num_audio_rows, MINIMAX_H3_AUDIO_CHANNELS)

    def generate(self, sample: Sample) -> Sample:
        gen = sample.parts[-1]
        params = gen.sampling_params
        require(params is not None, "MiniMaxH3Pipeline.generate: generation Part carries no sampling params")
        require(
            params.sigmas is not None,
            "MiniMaxH3Pipeline.generate: params.sigmas is None. The hosting engine pins the schedule onto the "
            "generation Part before generate(); this pipeline does not build one.",
        )

        conditioning = list(sample.conditioning())
        texts = next((c for c in conditioning if isinstance(c, Texts)), None)
        require(texts is not None, "MiniMaxH3Pipeline.generate: no text prompt in the sample conditioning")

        geometry = MiniMaxH3Geometry.from_params(params)
        conditions = MiniMaxH3Conditions(text=self.text_embed.embed(texts))

        # Driver-authoritative x_T. MiniMax-H3 draws VIDEO noise first, then
        # audio, off the one request generator -- the ``salt`` sibling
        # reproduces that split byte-identically, and the ORDER is part of what
        # makes a rollout reproducible.
        recipe = NoiseRecipe.from_sample(sample)
        video_noise = recipe.resolve(device=self.bundle.device, latent_shape=geometry.latent_shape)
        require(
            video_noise is not None,
            "MiniMaxH3Pipeline.generate: no initial latents. The driver x_T recipe (noise_group_ids + "
            "init_noise_latent_shape) is required; DISABLE_DRIVER_XT is not supported here.",
        )
        audio_noise = recipe.resolve(
            device=self.bundle.device, salt="audio", latent_shape=self.audio_latent_shape(geometry)
        )

        initial_latents = patchify_video_latents(video_noise, MINIMAX_H3_PATCH_SIZE)
        initial_audio_latents = audio_noise.reshape(1, geometry.num_audio_rows, -1)

        segment = self.diffusion.generate(
            conditions,
            params=params,
            sigmas=params.sigmas.to(self.bundle.device),
            geometry=geometry,
            initial_latents=initial_latents,
            initial_audio_latents=initial_audio_latents,
            sde_indices=list(params.sde_indices) if params.sde_indices is not None else None,
            denoise_seed_keys=[str(sample_id) for sample_id in sample.sample_ids],
            denoise_base_seed=int(params.seed) if params.seed is not None else 0,
        )

        final_rows = segment.latents_at(int(params.num_inference_steps))
        final_audio_rows = segment.aux_latents_at(int(params.num_inference_steps))
        videos = self.video_decode.decode(final_rows, geometry)
        audios = self.audio_decode.decode(final_audio_rows, geometry)

        return gen.fill(
            segment=segment,
            primitives={"video": videos, "audio": audios},
            primitive_metadata={"audio": {"sample_rate": MINIMAX_H3_AUDIO_SAMPLE_RATE}},
            conditions=conditions.to_dict(),
        )


__all__ = ["MiniMaxH3Pipeline"]
