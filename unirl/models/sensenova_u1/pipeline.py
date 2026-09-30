"""Sample-native SenseNova-U1.5 text-to-image pipeline."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Optional

import torch

from unirl.models.types.pipeline import Pipeline
from unirl.sde.kernels import FlowSDEStrategy, StepStrategy
from unirl.sde.runtime import FlowMatchSchedulePolicy
from unirl.types.noise_recipe import NoiseRecipe
from unirl.types.primitives import Texts
from unirl.types.sample import Sample
from unirl.utils.dtypes import parse_torch_dtype

from .bundle import SenseNovaU1Bundle
from .conditions import SenseNovaU1Conditions
from .config import SenseNovaU1PipelineConfig
from .diffusion import SenseNovaU1DiffusionParams, SenseNovaU1DiffusionStage
from .pixels import SenseNovaU1PixelDecodeStage, packed_pixel_shape
from .vendor.neo_unify.utils import SYSTEM_MESSAGE_FOR_GEN

DEFAULT_PIXEL_PATCH_SIZE = 32
IMAGE_START_TOKEN = "<img>"


class SenseNovaU1Pipeline(Pipeline):
    """SenseNova-U1.5 T2I rollout and replay pipeline."""

    def __init__(
        self,
        *,
        bundle: SenseNovaU1Bundle,
        diffusion: Optional[SenseNovaU1DiffusionStage] = None,
        pixel_decode: Optional[SenseNovaU1PixelDecodeStage] = None,
        strategy: Optional[StepStrategy] = None,
        shift: float = 3.0,
        autocast_precision: str = "bf16",
        trajectory_precision: str = "bf16",
        logprob_precision: str = "fp32",
    ) -> None:
        super().__init__()
        self.bundle = bundle
        self.diffusion = (
            diffusion
            if diffusion is not None
            else SenseNovaU1DiffusionStage(
                model=bundle,
                strategy=strategy if strategy is not None else FlowSDEStrategy(),
                autocast_precision=autocast_precision,
                trajectory_precision=trajectory_precision,
                logprob_precision=logprob_precision,
            )
        )
        self.pixel_decode = pixel_decode if pixel_decode is not None else SenseNovaU1PixelDecodeStage(bundle)
        self.shift = float(shift)
        self.autocast_dtype = parse_torch_dtype(autocast_precision, field_name="autocast_precision")

    @classmethod
    def from_config(
        cls,
        config: SenseNovaU1PipelineConfig,
        *,
        strategy: Optional[StepStrategy] = None,
    ) -> "SenseNovaU1Pipeline":
        """Load a bundle and assemble the full trainside pipeline."""
        return cls(
            bundle=SenseNovaU1Bundle.from_config(config),
            strategy=strategy,
            shift=float(config.timestep_shift),
            autocast_precision=config.autocast_precision,
            trajectory_precision=config.trajectory_precision,
            logprob_precision=config.logprob_precision,
        )

    @classmethod
    def latent_shape(cls, *, model_config: Any, sampling_spec: Any) -> tuple:
        """Driver-side NCHW pixel-noise shape, matching upstream seeded RNG."""
        del model_config
        height, width = int(sampling_spec.height), int(sampling_spec.width)
        packed_pixel_shape((height, width), patch_size=DEFAULT_PIXEL_PATCH_SIZE)
        return (3, height, width)

    def build_schedule_policy(self) -> FlowMatchSchedulePolicy:
        """U1.5 uses the standard rational FlowMatch shift."""
        return FlowMatchSchedulePolicy.static_only(self.shift)

    def _autocast_ctx(self):
        device = torch.device(self.bundle.device)
        if device.type == "cuda" and self.autocast_dtype in (torch.float16, torch.bfloat16):
            return torch.autocast("cuda", dtype=self.autocast_dtype)
        if device.type == "cpu" and self.autocast_dtype == torch.bfloat16:
            return torch.autocast("cpu", dtype=torch.bfloat16)
        return nullcontext()

    def _build_prefix(self, prompt: str, *, conditional: bool, image_shape: tuple[int, int]):
        model = self.bundle.model
        if conditional:
            query = model._build_t2i_query(
                prompt,
                system_message=SYSTEM_MESSAGE_FOR_GEN,
                append_text="<think>\n\n</think>\n\n" + IMAGE_START_TOKEN,
            )
        else:
            query = model._build_t2i_query(prompt, append_text=IMAGE_START_TOKEN)

        input_ids, indexes, attention_mask = model._build_t2i_text_inputs(self.bundle.tokenizer, query)
        cache = self.bundle.transformer(
            "prefix",
            input_ids=input_ids,
            indexes=indexes,
            attention_mask=attention_mask,
        )

        height, width = image_shape
        merge = int(1 / float(model.downsample_ratio))
        token_h = height // (int(model.patch_size) * merge)
        token_w = width // (int(model.patch_size) * merge)
        image_indexes = model._build_t2i_image_indexes(
            token_h,
            token_w,
            indexes.shape[1],
            device=input_ids.device,
        )
        return cache, image_indexes

    def build_conditions(
        self,
        texts: Texts,
        *,
        negatives: Optional[Texts] = None,
        guidance_scale: float = 1.0,
        image_shape: tuple[int, int] = (512, 512),
    ) -> SenseNovaU1Conditions:
        """Build frozen conditional/unconditional prefix KV caches."""
        if negatives is not None and len(negatives.texts) != len(texts.texts):
            raise ValueError(
                f"SenseNovaU1Pipeline.build_conditions: negatives={len(negatives.texts)} != prompts={len(texts.texts)}."
            )
        packed_pixel_shape(image_shape, patch_size=self.pixel_decode.pixel_patch_size)

        condition_caches = []
        uncondition_caches = []
        condition_indexes = []
        uncondition_indexes = []
        prefix_cache = {}
        with torch.no_grad(), self._autocast_ctx():
            for index, prompt in enumerate(texts.texts):
                condition_key = ("condition", str(prompt), tuple(image_shape))
                if condition_key not in prefix_cache:
                    prefix_cache[condition_key] = self._build_prefix(
                        str(prompt),
                        conditional=True,
                        image_shape=image_shape,
                    )
                cache, image_indexes = prefix_cache[condition_key]
                condition_caches.append(cache)
                condition_indexes.append(image_indexes)

                if float(guidance_scale) > 1.0:
                    negative = negatives.texts[index] if negatives is not None else ""
                    uncondition_key = ("uncondition", str(negative), tuple(image_shape))
                    if uncondition_key not in prefix_cache:
                        prefix_cache[uncondition_key] = self._build_prefix(
                            str(negative),
                            conditional=False,
                            image_shape=image_shape,
                        )
                    cache, image_indexes = prefix_cache[uncondition_key]
                else:
                    cache, image_indexes = None, None
                uncondition_caches.append(cache)
                uncondition_indexes.append(image_indexes)

        return SenseNovaU1Conditions(
            prompts=[str(text) for text in texts.texts],
            condition_caches=condition_caches,
            uncondition_caches=uncondition_caches,
            condition_image_indexes=condition_indexes,
            uncondition_image_indexes=uncondition_indexes,
            image_shapes=[tuple(image_shape)] * len(texts.texts),
        )

    def generate(self, sample: Sample) -> Sample:
        """Run T2I sampling and fill the pre-forked diffusion frontier."""
        frontier = sample.parts[-1]
        params = frontier.sampling_params
        if not isinstance(params, SenseNovaU1DiffusionParams):
            raise TypeError(
                "SenseNovaU1Pipeline.generate requires SenseNovaU1DiffusionParams, "
                f"got {type(params).__name__ if params is not None else 'None'}."
            )
        if params.sigmas is None:
            raise ValueError(
                "SenseNovaU1Pipeline.generate: sampling sigmas are not pinned; "
                "the hosting engine must apply pipeline.build_schedule_policy()."
            )

        conditioning = sample.conditioning()
        texts = conditioning[0] if conditioning else None
        if not isinstance(texts, Texts):
            raise TypeError(
                "SenseNovaU1Pipeline.generate expected a Texts prompt, "
                f"got {type(texts).__name__ if texts is not None else 'None'}."
            )

        image_shape = (int(params.height), int(params.width))
        conditions = self.build_conditions(
            texts,
            guidance_scale=float(params.guidance_scale),
            image_shape=image_shape,
        )
        initial = NoiseRecipe.from_sample(sample).resolve(
            device=torch.device(self.bundle.device),
            dtype=self.diffusion.trajectory_dtype,
        )
        segment = self.diffusion.diffuse(
            conditions,
            schedule=params.sigmas,
            params=params,
            initial_latents=initial,
        )
        images = self.pixel_decode.decode(segment, image_shape=image_shape)
        filled = frontier.fill(
            segment=segment,
            primitives={"image": images},
            conditions=conditions.to_dict(),
        )
        return sample.replace_frontier(filled)


__all__ = ["SenseNovaU1Pipeline"]
