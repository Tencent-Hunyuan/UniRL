"""HunyuanVideo15Pipeline — ``Sample → Sample`` end-to-end for HunyuanVideo-1.5."""

from __future__ import annotations

from typing import Any, Optional

from unirl.models.types.pipeline import Pipeline
from unirl.sde.kernels import DanceSDEStrategy, StepStrategy
from unirl.types.noise_recipe import NoiseRecipe
from unirl.types.primitives import Texts
from unirl.types.sample import Sample
from unirl.types.sampling import DiffusionSamplingParams

from .bundle import HunyuanVideo15Bundle
from .conditions import HunyuanVideo15Conditions
from .config import HunyuanVideo15PipelineConfig
from .diffusion import (
    HunyuanVideo15DiffusionStage,
    HunyuanVideo15DiffusionStep,
)
from .text_embed import HunyuanVideo15TextEmbedStage
from .vae import HunyuanVideo15VAEDecodeStage


class HunyuanVideo15Pipeline(Pipeline):
    """HunyuanVideo-1.5 generate pipeline (T2V; I2V deferred): ``Sample → Sample``."""

    def __init__(
        self,
        *,
        bundle: HunyuanVideo15Bundle,
        text_embed: Optional[HunyuanVideo15TextEmbedStage] = None,
        diffusion: Optional[HunyuanVideo15DiffusionStage] = None,
        vae_decode: Optional[HunyuanVideo15VAEDecodeStage] = None,
        strategy: Optional[StepStrategy] = None,
        shift: float = 5.0,
        autocast_precision: str = "bf16",
        trajectory_precision: str = "fp16",
        logprob_precision: str = "fp32",
        mllm_max_length: int = 1000,
        mllm_crop_start: int = 108,
        mllm_skip_layers: int = 2,
        byt5_max_length: int = 256,
        vision_num_semantic_tokens: int = 729,
        vision_states_dim: int = 1152,
        latent_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.bundle = bundle
        self.text_embed = (
            text_embed
            if text_embed is not None
            else HunyuanVideo15TextEmbedStage(
                bundle,
                mllm_max_length=mllm_max_length,
                mllm_crop_start=mllm_crop_start,
                mllm_skip_layers=mllm_skip_layers,
                byt5_max_length=byt5_max_length,
            )
        )
        if diffusion is None:
            diffusion = HunyuanVideo15DiffusionStage(
                model=bundle,
                step=HunyuanVideo15DiffusionStep(),
                strategy=strategy if strategy is not None else DanceSDEStrategy(),
                autocast_precision=autocast_precision,
                trajectory_precision=trajectory_precision,
                logprob_precision=logprob_precision,
                vision_num_semantic_tokens=vision_num_semantic_tokens,
                vision_states_dim=vision_states_dim,
                latent_channels=latent_channels,
            )
        self.diffusion = diffusion
        self.vae_decode = vae_decode if vae_decode is not None else HunyuanVideo15VAEDecodeStage(bundle)
        self.shift = shift

    @classmethod
    def latent_shape(cls, *, model_config: Any, sampling_spec: Any) -> tuple:
        """Per-sample 5D latent shape ``(C, T_lat, H_lat, W_lat)`` for driver-side noise pre-computation."""
        height = int(sampling_spec.height)
        width = int(sampling_spec.width)
        num_frames = int(sampling_spec.num_frames)
        spatial = HunyuanVideo15DiffusionStage.DEFAULT_SPATIAL_DOWNSAMPLE
        temporal = HunyuanVideo15DiffusionStage.DEFAULT_TEMPORAL_DOWNSAMPLE
        config_channels = getattr(model_config, "latent_channels", None)
        channels = (
            int(config_channels)
            if config_channels is not None
            else HunyuanVideo15DiffusionStage.DEFAULT_LATENT_CHANNELS
        )
        latent_t = (num_frames - 1) // temporal + 1
        latent_h = max(1, height // spatial)
        latent_w = max(1, width // spatial)
        return (channels, latent_t, latent_h, latent_w)

    @classmethod
    def from_config(
        cls,
        config: HunyuanVideo15PipelineConfig,
        *,
        strategy: Optional[StepStrategy] = None,
    ) -> "HunyuanVideo15Pipeline":
        """Build the full pipeline from a config."""
        bundle = HunyuanVideo15Bundle.from_config(config)
        text_embed = HunyuanVideo15TextEmbedStage(
            bundle,
            mllm_max_length=config.mllm_max_length,
            mllm_crop_start=config.mllm_crop_start,
            mllm_skip_layers=config.mllm_skip_layers,
            byt5_max_length=config.byt5_max_length,
        )
        step = HunyuanVideo15DiffusionStep()
        diffusion = HunyuanVideo15DiffusionStage(
            model=bundle,
            step=step,
            strategy=strategy if strategy is not None else DanceSDEStrategy(),
            autocast_precision=config.autocast_precision,
            trajectory_precision=config.trajectory_precision,
            logprob_precision=config.logprob_precision,
            vision_num_semantic_tokens=config.vision_num_semantic_tokens,
            vision_states_dim=config.vision_states_dim,
            latent_channels=config.latent_channels,
        )
        vae_decode = HunyuanVideo15VAEDecodeStage(bundle)
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
    ) -> HunyuanVideo15Conditions:
        """Encode prompts (MLLM + Glyph, + optional CFG negatives) into ``HunyuanVideo15Conditions``."""
        if negatives is not None and len(negatives.texts) != len(texts.texts):
            raise ValueError(
                f"HunyuanVideo15Pipeline.build_conditions: negative_text length "
                f"{len(negatives.texts)} != text length {len(texts.texts)}"
            )
        if negatives is None and float(guidance_scale) > 1.0:
            negatives = Texts(texts=[""] * len(texts.texts))

        text_mllm = self.text_embed.embed_mllm(texts)
        text_glyph = self.text_embed.embed_glyph(texts)
        if negatives is not None:
            negative_text_mllm = self.text_embed.embed_mllm(negatives)
            negative_text_glyph = self.text_embed.embed_glyph(negatives)
        else:
            negative_text_mllm = None
            negative_text_glyph = None

        return HunyuanVideo15Conditions(
            text_mllm=text_mllm,
            text_glyph=text_glyph,
            negative_text_mllm=negative_text_mllm,
            negative_text_glyph=negative_text_glyph,
        )

    def generate(self, sample: Sample) -> Sample:
        """Run HunyuanVideo-1.5 T2V end-to-end, filling the frontier (pre-forked) gen Part."""
        frontier = sample.parts[-1]
        params = frontier.sampling_params
        if not isinstance(params, DiffusionSamplingParams):
            raise TypeError(
                f"HunyuanVideo15Pipeline.generate: frontier gen Part must carry DiffusionSamplingParams, "
                f"got {type(params).__name__ if params is not None else 'None'}"
            )
        if params.sigmas is None:
            raise ValueError(
                "HunyuanVideo15Pipeline.generate: gen part sampling_params.sigmas is None. The hosting "
                "engine must pin σ before invoking pipeline.generate; see the σ ownership note "
                "in unirl.models.types.pipeline."
            )

        conditioning = sample.conditioning()
        texts = conditioning[0] if conditioning else None
        if not isinstance(texts, Texts):
            raise TypeError(
                f"HunyuanVideo15Pipeline.generate: expected a Texts prompt from sample.conditioning()[0], "
                f"got {type(texts).__name__ if texts is not None else 'None'}"
            )

        hv_conds = self.build_conditions(texts, guidance_scale=float(params.guidance_scale))
        schedule = params.sigmas.to(self.bundle.device)

        initial_latents = NoiseRecipe.from_sample(sample).resolve()

        latent_seg = self.diffusion.diffuse(hv_conds, schedule=schedule, params=params, initial_latents=initial_latents)
        videos = self.vae_decode.decode(latent_seg)

        filled = frontier.fill(segment=latent_seg, primitives={"video": videos}, conditions=hv_conds.to_dict())
        return Sample(parts=[*sample.parts[:-1], filled], reward_compute_s=sample.reward_compute_s)


__all__ = ["HunyuanVideo15Pipeline"]
