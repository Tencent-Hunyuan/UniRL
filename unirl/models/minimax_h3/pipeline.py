"""MiniMax-H3 t2va pipeline -- prompt -> joint video + stereo audio."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Tuple

from unirl.config.require import require
from unirl.models.types.pipeline import Pipeline
from unirl.sde.kernels import StepStrategy
from unirl.sde.runtime import FlowMatchSchedulePolicy
from unirl.types.noise_recipe import NoiseRecipe
from unirl.types.primitives import Texts
from unirl.types.sample import Sample

from .bundle import MiniMaxH3Bundle
from .conditions import MiniMaxH3Conditions
from .config import (
    MINIMAX_H3_AUDIO_LATENT_CHANNELS,
    MINIMAX_H3_PATCH_SIZE,
    MiniMaxH3PipelineConfig,
)
from .diffusion import MiniMaxH3DiffusionStage
from .packing import MiniMaxH3Geometry
from .text_embed import MiniMaxH3TextEmbedStage
from .vae import (
    MINIMAX_H3_AUDIO_SAMPLE_RATE,
    MiniMaxH3AudioDecodeStage,
    MiniMaxH3VideoDecodeStage,
)
from .vendor import patchify_video_latents


class MiniMaxH3Pipeline(Pipeline):
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
        super().__init__()
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
                trajectory_precision=config.trajectory_precision,
                logprob_precision=config.logprob_precision,
            ),
            video_decode=MiniMaxH3VideoDecodeStage(bundle),
            audio_decode=MiniMaxH3AudioDecodeStage(bundle),
            config=config,
        )

    def build_schedule_policy(self) -> FlowMatchSchedulePolicy:
        """The VIDEO sigma policy the hosting engine pins onto the Part."""
        return FlowMatchSchedulePolicy.static_only(shift=float(self.config.video_shift))

    @property
    def audio_sampling_rate(self) -> int:
        """Read the generated-audio sampling rate from the loaded VAE config."""
        if getattr(self.bundle, "audio_vae", None) is not None:
            return int(self.bundle.audio_vae.config.sampling_rate)
        return MINIMAX_H3_AUDIO_SAMPLE_RATE

    @classmethod
    def latent_shape(cls, *, model_config: Any, sampling_spec: Any) -> Tuple[int, ...]:
        """Per-sample UNPACKED video latent shape for the driver x_T recipe."""
        del model_config  # geometry is fully determined by the request
        return MiniMaxH3Geometry.from_params(sampling_spec).latent_shape

    @staticmethod
    def audio_latent_shape(geometry: MiniMaxH3Geometry) -> Tuple[int, ...]:
        """Per-sample audio x_T shape ``(rows, latent_channels)``."""
        return (geometry.num_audio_rows, MINIMAX_H3_AUDIO_LATENT_CHANNELS)

    def _append_stage_timing(self, path: str, *, geometry: MiniMaxH3Geometry, **fields: Any) -> None:
        """Append one generate() stage-timing row to a per-rank JSONL sidecar."""
        import torch.distributed as dist

        rank = int(dist.get_rank()) if dist.is_available() and dist.is_initialized() else 0
        row = {
            "rank": rank,
            "wall_utc": time.time(),
            "height": geometry.height,
            "width": geometry.width,
            "num_frames": geometry.num_frames,
            "video_rows": geometry.num_video_rows,
            "audio_rows": geometry.num_audio_rows,
            **fields,
        }
        with open(f"{path}.rank{rank}.jsonl", "a") as handle:
            handle.write(json.dumps(row) + "\n")

    def _stage_clock(self) -> float:
        """Device-synchronized seconds, so a stage timer bills async CUDA work correctly."""
        import torch

        device = torch.device(self.bundle.device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter()

    def generate(self, sample: Sample) -> Sample:
        timing_path = os.environ.get("UNIRL_H3_STAGE_TIMING")
        clock = self._stage_clock if timing_path else time.perf_counter
        t_start = clock()
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
        t_embed = clock()

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

        # `patchify_video_latents` returns 2-D `(batch*rows, C)` -- the reference
        # pipeline is strictly batch-1 so it folds the batch away. The
        # transformer indexes rows on dim 1, so restore the batch axis.
        batch = int(video_noise.shape[0])
        initial_latents = patchify_video_latents(video_noise, MINIMAX_H3_PATCH_SIZE).reshape(
            batch, geometry.num_video_rows, geometry.video_token_dim
        )
        initial_audio_latents = audio_noise.reshape(batch, geometry.num_audio_rows, -1)
        t_noise = clock()

        segment = self.diffusion.generate(
            conditions,
            params=params,
            sigmas=params.sigmas.to(self.bundle.device),
            geometry=geometry,
            initial_latents=initial_latents,
            initial_audio_latents=initial_audio_latents,
            sde_indices=list(params.sde_indices) if params.sde_indices is not None else None,
            # sample_ids live on the PART, not the Sample -- `Sample` has
            # root_group_ids()/split() but no sample_ids of its own.
            denoise_seed_keys=[str(sample_id) for sample_id in gen.sample_ids],
            denoise_base_seed=int(params.seed) if params.seed is not None else 0,
        )

        final_rows = segment.latents_at(int(params.num_inference_steps))
        final_audio_rows = segment.aux_latents_at(int(params.num_inference_steps))
        t_denoise = clock()
        videos = self.video_decode.decode(final_rows, geometry)
        t_video = clock()
        audios = self.audio_decode.decode(final_audio_rows, geometry)
        t_audio = clock()
        if timing_path:
            self._append_stage_timing(
                timing_path,
                geometry=geometry,
                text_tokens=int(conditions.text.embeds.shape[1]),
                num_samples=len(gen.sample_ids),
                embed_s=t_embed - t_start,
                noise_s=t_noise - t_embed,
                denoise_s=t_denoise - t_noise,
                video_decode_s=t_video - t_denoise,
                audio_decode_s=t_audio - t_video,
                total_s=t_audio - t_start,
            )

        filled = gen.fill(
            segment=segment,
            primitives={"video": videos, "audio": audios},
            primitive_metadata={"audio": {"sample_rate": self.audio_sampling_rate}},
            conditions=conditions.to_dict(),
        )
        # `Part.fill` returns a PART; the engine does `chunk.parts[-1]` on what
        # generate() hands back, so the whole Sample has to come back out.
        # Same shape as sd3 / wan21 / ltx2.
        return Sample(parts=[*sample.parts[:-1], filled], reward_compute_s=sample.reward_compute_s)


__all__ = ["MiniMaxH3Pipeline"]
