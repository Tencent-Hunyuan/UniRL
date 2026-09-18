"""WAN22V2VPipeline — ``Sample → Sample`` for WAN 2.2 video-to-video."""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

import torch

from unirl.models.types.pipeline import Pipeline
from unirl.models.wan21.conditions import WAN21Conditions
from unirl.models.wan21.text_embed import WAN21TextEmbedStage
from unirl.models.wan21.vae import WAN21VAEDecodeStage
from unirl.models.wan22.bundle import WAN22Bundle
from unirl.models.wan22.diffusion import WAN22DiffusionStage, WAN22DiffusionStep
from unirl.sde.kernels import DanceSDEStrategy, StepStrategy
from unirl.sde.noise import generate_latents
from unirl.types.noise_recipe import NoiseRecipe
from unirl.types.primitives import Texts, Videos
from unirl.types.sample import Sample
from unirl.types.sampling import DiffusionSamplingParams

from .config import DEFAULT_V2V_STRENGTH
from .video_encode import WAN22VideoLatentEncodeStage


class WAN22V2VPipeline(Pipeline):
    """WAN 2.2 video-to-video generate pipeline."""

    def __init__(
        self,
        *,
        bundle: WAN22Bundle,
        text_embed: Optional[WAN21TextEmbedStage] = None,
        diffusion: Optional[WAN22DiffusionStage] = None,
        vae_decode: Optional[WAN21VAEDecodeStage] = None,
        strategy: Optional[StepStrategy] = None,
        shift: float = 5.0,
        strength: float = DEFAULT_V2V_STRENGTH,
        autocast_precision: str = "bf16",
        trajectory_precision: str = "fp16",
        logprob_precision: str = "fp32",
        max_sequence_length: int = 512,
    ) -> None:
        super().__init__()
        self.bundle = bundle
        self.shift = float(shift)
        self.strength = float(strength)
        self.text_embed = (
            text_embed
            if text_embed is not None
            else WAN21TextEmbedStage(bundle, max_sequence_length=int(max_sequence_length))
        )
        if diffusion is None:
            diffusion = WAN22DiffusionStage(
                model=bundle,
                step=WAN22DiffusionStep(),
                strategy=strategy if strategy is not None else DanceSDEStrategy(),
                autocast_precision=autocast_precision,
                trajectory_precision=trajectory_precision,
                logprob_precision=logprob_precision,
            )
        self.diffusion = diffusion
        self.vae_decode = vae_decode if vae_decode is not None else WAN21VAEDecodeStage(bundle)

    @staticmethod
    def _sde_indices_in_trimmed_frame(sde_indices: Any, *, t_full: int, t_eff: int) -> list:
        """Remap SDE step indices from the full-schedule frame to the trimmed V2V frame."""
        if not sde_indices or int(t_full) <= 0:
            return []
        remapped = {min(int(t_eff) - 1, max(0, round(int(i) * int(t_eff) / int(t_full)))) for i in sde_indices}
        return sorted(remapped)

    @classmethod
    def latent_shape(cls, *, model_config: Any, sampling_spec: Any) -> tuple:
        height = int(sampling_spec.height)
        width = int(sampling_spec.width)
        num_frames = int(sampling_spec.num_frames)
        if (num_frames - 1) % 4 != 0:
            raise ValueError(
                f"WAN VAE temporal_downsample=4 requires (num_frames - 1) % 4 == 0, "
                f"got num_frames={num_frames}; valid choices: 1, 5, 9, 13, 17, 21, ..."
            )
        latent_t = (num_frames - 1) // 4 + 1
        return (16, latent_t, height // 8, width // 8)

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        strategy: Optional[StepStrategy] = None,
    ) -> "WAN22V2VPipeline":
        bundle = WAN22Bundle.from_config(config)
        return cls(
            bundle=bundle,
            strategy=strategy if strategy is not None else DanceSDEStrategy(),
            shift=float(config.shift),
            strength=float(getattr(config, "strength", DEFAULT_V2V_STRENGTH)),
            autocast_precision=config.autocast_precision,
            trajectory_precision=config.trajectory_precision,
            logprob_precision=config.logprob_precision,
            max_sequence_length=int(config.max_sequence_length),
        )

    def build_conditions(
        self,
        texts: Texts,
        *,
        negatives: Optional[Texts] = None,
        guidance_scale: float = 1.0,
    ) -> WAN21Conditions:
        """Build WAN text conditions; source-video encoding stays in generate."""
        if negatives is not None and len(negatives.texts) != len(texts.texts):
            raise ValueError(
                f"WAN22V2VPipeline.build_conditions: negative_text length "
                f"{len(negatives.texts)} != text length {len(texts.texts)}"
            )
        if negatives is None and float(guidance_scale) > 1.0:
            negatives = Texts(texts=[""] * len(texts.texts))
        text_cond = self.text_embed.embed(texts)
        negative_text_cond = self.text_embed.embed(negatives) if negatives is not None else None
        return WAN21Conditions(text=text_cond, negative_text=negative_text_cond)

    def generate(self, sample: Sample) -> Sample:
        """Run V2V over the trimmed denoising tail and fill the frontier Part."""
        frontier = sample.parts[-1]
        params = frontier.sampling_params
        if not isinstance(params, DiffusionSamplingParams):
            raise TypeError(
                "WAN22V2VPipeline.generate: frontier gen Part must carry "
                f"DiffusionSamplingParams, got {type(params).__name__ if params is not None else 'None'}"
            )
        if params.sigmas is None:
            raise ValueError(
                "WAN22V2VPipeline.generate: frontier sampling_params.sigmas is None; "
                "the hosting engine must pin the schedule before pipeline.generate."
            )

        conditioning = sample.conditioning()
        text_inputs = [value for value in conditioning if isinstance(value, Texts)]
        video_inputs = [value for value in conditioning if isinstance(value, Videos)]
        if len(text_inputs) != 1:
            raise TypeError(
                f"WAN22V2VPipeline.generate: expected exactly one Texts conditioning primitive, got {len(text_inputs)}"
            )
        if len(video_inputs) != 1:
            raise TypeError(
                f"WAN22V2VPipeline.generate: expected exactly one Videos conditioning primitive, "
                f"got {len(video_inputs)}"
            )
        texts, videos = text_inputs[0], video_inputs[0]
        if len(videos) != len(texts.texts):
            raise ValueError(f"WAN22V2VPipeline.generate: video count {len(videos)} != text count {len(texts.texts)}")

        primary_g = float(params.guidance_scale)
        low_g = float(params.guidance_scale_2) if params.guidance_scale_2 is not None else primary_g
        wan_conds = self.build_conditions(texts, guidance_scale=max(primary_g, low_g))

        device = self.bundle.device
        video_latent_cond = WAN22VideoLatentEncodeStage(
            self.bundle,
            num_frames=int(params.num_frames),
            height=int(params.height),
            width=int(params.width),
        ).encode(videos)
        video_latents = video_latent_cond.latents.to(device=device, dtype=torch.float32)
        batch_size = int(video_latents.shape[0])

        full_schedule = params.sigmas.to(device)
        t_full = int(full_schedule.shape[0]) - 1
        if t_full < 1:
            raise ValueError(
                f"WAN22V2VPipeline.generate: sigma schedule must contain at least two points, "
                f"got {int(full_schedule.shape[0])}"
            )
        strength = float(params.strength) if params.strength is not None else self.strength
        if not 0.0 < strength <= 1.0:
            raise ValueError(f"WAN22V2VPipeline.generate: strength must be in (0, 1], got {strength}")
        t_eff = max(1, min(t_full, int(round(t_full * strength))))
        t_start = t_full - t_eff
        trimmed_schedule = full_schedule[t_start:].contiguous()
        sigma_start = trimmed_schedule[0].to(torch.float32)

        noise_recipe = NoiseRecipe.from_sample(sample)
        if noise_recipe.initial_latents is not None:
            raise ValueError(
                "WAN22V2VPipeline.generate: source-video conditioning cannot be combined "
                "with initial latents on the generation Part."
            )
        noise = noise_recipe.for_batch(
            batch_size,
            latent_shape=tuple(video_latents.shape[1:]),
        ).resolve(device=device, dtype=torch.float32)
        if noise is None:
            noise_ids = list(frontier.group_ids) if params.init_same_noise else list(frontier.sample_ids)
            noise = generate_latents(
                batch_size=batch_size,
                latent_shape=tuple(video_latents.shape[1:]),
                device=device,
                dtype=torch.float32,
                init_same_noise=bool(params.init_same_noise),
                samples_per_prompt=int(params.samples_per_prompt),
                noise_group_ids=noise_ids,
                base_seed=int(params.seed or 0),
            )
        if tuple(noise.shape) != tuple(video_latents.shape):
            raise ValueError(
                f"WAN22V2VPipeline.generate: noise shape {tuple(noise.shape)} incompatible with "
                f"video latents {tuple(video_latents.shape)}"
            )

        x_start = (1.0 - sigma_start) * video_latents + sigma_start * noise
        sde_indices = self._sde_indices_in_trimmed_frame(
            params.sde_indices,
            t_full=t_full,
            t_eff=t_eff,
        )
        # The returned Part must describe the trajectory that actually ran, not the discarded full schedule.
        v2v_params = dataclasses.replace(
            params,
            num_inference_steps=t_eff,
            sde_indices=sde_indices,
            sigmas=trimmed_schedule,
        )
        latent_seg = self.diffusion.diffuse(
            wan_conds,
            schedule=trimmed_schedule,
            params=v2v_params,
            initial_latents=x_start,
        )
        decoded = self.vae_decode.decode(latent_seg)

        effective_frontier = dataclasses.replace(frontier, sampling_params=v2v_params)
        filled = effective_frontier.fill(
            segment=latent_seg,
            primitives={"video": decoded},
            conditions=wan_conds.to_dict(),
        )
        return sample.replace_frontier(filled)


__all__ = ["WAN22V2VPipeline"]
