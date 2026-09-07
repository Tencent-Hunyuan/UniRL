"""QwenImageEditPlusPipeline — ``Sample → Sample`` for Edit-Plus."""

from __future__ import annotations

from typing import Any, Optional

from unirl.models.qwen_image.vae import QwenImageVAEDecodeStage
from unirl.models.types.pipeline import Pipeline
from unirl.sde.kernels import FlowSDEStrategy, StepStrategy
from unirl.types.noise_recipe import NoiseRecipe
from unirl.types.primitives import Images, Texts
from unirl.types.sample import Sample
from unirl.types.sampling import DiffusionSamplingParams

from .bundle import QwenImageEditPlusBundle
from .conditions import QwenImageEditPlusConditions
from .config import QwenImageEditPlusPipelineConfig
from .diffusion import (
    QwenImageEditPlusDiffusionStage,
    QwenImageEditPlusDiffusionStep,
)
from .text_embed import QwenImageEditPlusTextEmbedStage
from .vae import QwenImageEditPlusVAEEncodeStage


class QwenImageEditPlusPipeline(Pipeline):
    """Qwen-Image-Edit-Plus generate pipeline."""

    def __init__(
        self,
        *,
        bundle: QwenImageEditPlusBundle,
        text_embed: Optional[QwenImageEditPlusTextEmbedStage] = None,
        diffusion: Optional[QwenImageEditPlusDiffusionStage] = None,
        vae_encode: Optional[QwenImageEditPlusVAEEncodeStage] = None,
        vae_decode: Optional[QwenImageVAEDecodeStage] = None,
        strategy: Optional[StepStrategy] = None,
        shift: float = 3.0,
        autocast_precision: str = "bf16",
        trajectory_precision: str = "fp16",
        logprob_precision: str = "fp32",
        max_sequence_length: int = 512,
        use_condition_image_prompt: bool = True,
        processor_path: Optional[str] = None,
        batch_replay_steps: bool = False,
    ) -> None:
        super().__init__()
        self.bundle = bundle
        self.use_condition_image_prompt = bool(use_condition_image_prompt)
        if text_embed is None and bundle.text_encoder is not None:
            text_embed = QwenImageEditPlusTextEmbedStage(
                bundle,
                max_sequence_length=max_sequence_length,
                processor_path=processor_path,
            )
        self.text_embed = text_embed
        if diffusion is None:
            diffusion = QwenImageEditPlusDiffusionStage(
                model=bundle,
                step=QwenImageEditPlusDiffusionStep(),
                strategy=strategy if strategy is not None else FlowSDEStrategy(),
                autocast_precision=autocast_precision,
                trajectory_precision=trajectory_precision,
                logprob_precision=logprob_precision,
                batch_replay_steps=batch_replay_steps,
            )
        self.diffusion = diffusion
        self.vae_encode = vae_encode if vae_encode is not None else QwenImageEditPlusVAEEncodeStage(bundle)
        self.vae_decode = vae_decode if vae_decode is not None else QwenImageVAEDecodeStage(bundle)
        self.shift = shift

    def build_schedule_policy(self):
        """Build the FlowMatchSchedulePolicy — identical to base Qwen-Image."""
        from unirl.models.qwen_image.config import _qwen_image_dynamic_overrides
        from unirl.sde.runtime import FlowMatchSchedulePolicy

        return FlowMatchSchedulePolicy.from_pretrained(
            getattr(self.bundle, "pretrained_path", None),
            shift=float(self.shift),
            require_dynamic=True,
            dynamic_overrides=_qwen_image_dynamic_overrides(),
        )

    @classmethod
    def latent_shape(cls, *, model_config: Any, sampling_spec: Any) -> tuple:
        """Per-sample latent shape ``(C, H_lat, W_lat)`` for driver-side"""
        height = int(sampling_spec.height)
        width = int(sampling_spec.width)
        vae_scale_factor = 8
        latent_h = 2 * (height // (vae_scale_factor * 2))
        latent_w = 2 * (width // (vae_scale_factor * 2))
        return (16, latent_h, latent_w)

    @classmethod
    def from_config(
        cls,
        config: QwenImageEditPlusPipelineConfig,
        *,
        strategy: Optional[StepStrategy] = None,
    ) -> "QwenImageEditPlusPipeline":
        """Build the full Edit-Plus pipeline from a config."""
        bundle = QwenImageEditPlusBundle.from_config(config)
        text_embed: Optional[QwenImageEditPlusTextEmbedStage] = None
        if bundle.text_encoder is not None:
            text_embed = QwenImageEditPlusTextEmbedStage(
                bundle,
                max_sequence_length=config.max_sequence_length,
                processor_path=config.text_encoder_ckpt_path or config.pretrained_model_ckpt_path,
            )
        step = QwenImageEditPlusDiffusionStep()
        diffusion = QwenImageEditPlusDiffusionStage(
            model=bundle,
            step=step,
            strategy=strategy if strategy is not None else FlowSDEStrategy(),
            autocast_precision=config.autocast_precision,
            trajectory_precision=config.trajectory_precision,
            logprob_precision=config.logprob_precision,
            batch_replay_steps=config.batch_replay_steps,
        )
        vae_encode = QwenImageEditPlusVAEEncodeStage(bundle)
        vae_decode = QwenImageVAEDecodeStage(bundle)
        return cls(
            bundle=bundle,
            text_embed=text_embed,
            diffusion=diffusion,
            vae_encode=vae_encode,
            vae_decode=vae_decode,
            shift=float(config.shift),
            max_sequence_length=config.max_sequence_length,
            use_condition_image_prompt=config.use_condition_image_prompt,
            processor_path=config.text_encoder_ckpt_path or config.pretrained_model_ckpt_path,
        )

    def build_conditions(
        self,
        texts: Texts,
        *,
        negatives: Optional[Texts] = None,
        images: Optional[Images] = None,
        guidance_scale: float = 1.0,
    ) -> QwenImageEditPlusConditions:
        """Build the text side of Edit-Plus conditions."""
        if negatives is not None and len(negatives.texts) != len(texts.texts):
            raise ValueError(
                f"QwenImageEditPlusPipeline.build_conditions: negative_text length "
                f"{len(negatives.texts)} != text length {len(texts.texts)}"
            )
        if self.text_embed is None:
            raise RuntimeError(
                "QwenImageEditPlusPipeline.build_conditions: no text_embed stage "
                "(load_text_encoder=False); trainside conditioning requires load_text_encoder=True."
            )
        if negatives is None and float(guidance_scale) > 1.0:
            negatives = Texts(texts=[" "] * len(texts.texts))
        embed_images = images if self.use_condition_image_prompt else None
        text_cond = self.text_embed.embed(texts, embed_images)
        negative_text_cond = self.text_embed.embed(negatives, embed_images) if negatives is not None else None
        return QwenImageEditPlusConditions(text=text_cond, negative_text=negative_text_cond)

    def generate(self, sample: Sample) -> Sample:
        """Run Edit-Plus text+image → image and fill the diffusion frontier."""
        frontier = sample.parts[-1]
        params = frontier.sampling_params
        if not isinstance(params, DiffusionSamplingParams):
            raise TypeError(
                "QwenImageEditPlusPipeline.generate: frontier gen Part must carry "
                f"DiffusionSamplingParams, got {type(params).__name__ if params is not None else 'None'}"
            )
        if params.sigmas is None:
            raise ValueError(
                "QwenImageEditPlusPipeline.generate: frontier sampling_params.sigmas is None; "
                "the hosting engine must pin the schedule before pipeline.generate."
            )
        if int(params.height) % 16 or int(params.width) % 16:
            raise ValueError(
                f"QwenImageEditPlusPipeline.generate: height ({params.height}) and width "
                f"({params.width}) must be divisible by 16"
            )

        conditioning = sample.conditioning()
        text_inputs = [value for value in conditioning if isinstance(value, Texts)]
        image_inputs = [value for value in conditioning if isinstance(value, Images)]
        if len(text_inputs) != 1:
            raise TypeError(
                "QwenImageEditPlusPipeline.generate: expected exactly one Texts conditioning "
                f"primitive, got {len(text_inputs)}"
            )
        if len(image_inputs) != 1:
            raise TypeError(
                "QwenImageEditPlusPipeline.generate: expected exactly one Images conditioning "
                f"primitive (Edit-Plus is edit-only), got {len(image_inputs)}"
            )
        texts, images = text_inputs[0], image_inputs[0]
        if len(images) != len(texts.texts):
            raise ValueError(
                f"QwenImageEditPlusPipeline.generate: image batch {len(images)} != text batch {len(texts.texts)}"
            )

        edit_conds = self.build_conditions(
            texts,
            images=images,
            guidance_scale=float(params.guidance_scale),
        )
        image_latent = self.vae_encode.encode(images)
        edit_conds.image_latent = image_latent

        schedule = params.sigmas.to(self.bundle.device)
        initial_latents = NoiseRecipe.from_sample(sample).resolve()
        latent_seg = self.diffusion.diffuse(
            edit_conds,
            schedule=schedule,
            params=params,
            initial_latents=initial_latents,
        )
        decoded = self.vae_decode.decode(latent_seg)
        return sample.with_filled_frontier(
            segment=latent_seg,
            primitives={"image": decoded},
            conditions=edit_conds.to_dict(),
        )


__all__ = ["QwenImageEditPlusPipeline"]
