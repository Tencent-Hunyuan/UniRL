"""QwenImage21Pipeline - ``Sample -> Sample`` text-to-image for Qwen-Image-2.1."""

from __future__ import annotations

from typing import Any, Optional

from unirl.models.qwen_image.conditions import QwenImageConditions
from unirl.models.types.pipeline import Pipeline
from unirl.sde.kernels import FlowSDEStrategy, StepStrategy
from unirl.sde.runtime import FlowMatchSchedulePolicy
from unirl.types.noise_recipe import NoiseRecipe
from unirl.types.primitives import Texts
from unirl.types.sample import Sample
from unirl.types.sampling import DiffusionSamplingParams

from .bundle import QwenImage21Bundle
from .diffusion import LATENT_CHANNELS, VAE_SCALE_FACTOR, QwenImage21DiffusionStage, QwenImage21DiffusionStep
from .text_embed import QwenImage21TextEmbedStage
from .vae import QwenImage21VAEDecodeStage


class QwenImage21Pipeline(Pipeline):
    """Qwen-Image-2.1 text-to-image generate pipeline: ``Sample -> Sample``."""

    def __init__(
        self,
        *,
        bundle: QwenImage21Bundle,
        strategy: Optional[StepStrategy] = None,
        autocast_precision: str = "bf16",
        trajectory_precision: str = "fp16",
        logprob_precision: str = "fp32",
        batch_replay_steps: bool = False,
    ) -> None:
        super().__init__()
        self.bundle = bundle
        self.text_embed = QwenImage21TextEmbedStage(bundle)
        self.diffusion = QwenImage21DiffusionStage(
            model=bundle,
            step=QwenImage21DiffusionStep(),
            strategy=strategy if strategy is not None else FlowSDEStrategy(),
            autocast_precision=autocast_precision,
            trajectory_precision=trajectory_precision,
            logprob_precision=logprob_precision,
            batch_replay_steps=batch_replay_steps,
        )
        self.vae_decode = QwenImage21VAEDecodeStage(bundle)

    def build_schedule_policy(self) -> FlowMatchSchedulePolicy:
        """Dynamic-shift sigma policy with the checkpoint scheduler_config values, on the 16x / patch-1 latent grid."""
        return FlowMatchSchedulePolicy(
            use_dynamic_shifting=True,
            base_shift=0.5,
            max_shift=0.9,
            base_image_seq_len=256,
            max_image_seq_len=8192,
            time_shift_type="exponential",
            shift_terminal=0.02,
            vae_scale_factor=VAE_SCALE_FACTOR,
            patch_size=1,
        )

    @classmethod
    def latent_shape(cls, *, model_config: Any, sampling_spec: Any) -> tuple:
        """Per-sample latent shape ``(64, H/16, W/16)`` for driver-side noise, both sides floored to 32 px."""
        align = VAE_SCALE_FACTOR * 2
        return (LATENT_CHANNELS, 2 * (int(sampling_spec.height) // align), 2 * (int(sampling_spec.width) // align))

    def generate(self, sample: Sample) -> Sample:
        """Run Qwen-Image-2.1 t2i end-to-end, filling the frontier (pre-forked) gen Part."""
        frontier = sample.frontier_gen_part(DiffusionSamplingParams)
        params = frontier.sampling_params
        if params.sigmas is None:
            raise ValueError(
                "QwenImage21Pipeline.generate: gen part sampling_params.sigmas is None. The hosting "
                "engine must pin sigma before invoking pipeline.generate; see unirl.models.types.pipeline."
            )
        align = VAE_SCALE_FACTOR * 2
        if params.height % align or params.width % align:
            raise ValueError(
                f"QwenImage21Pipeline.generate: height x width = {params.height} x {params.width} must be multiples "
                f"of {align}; the sigma policy counts H/16 x W/16 tokens (see README)."
            )
        conditioning = sample.conditioning()
        texts = conditioning[0] if conditioning else None
        if not isinstance(texts, Texts):
            raise TypeError(
                "QwenImage21Pipeline.generate: expected a Texts prompt from sample.conditioning()[0], "
                f"got {type(texts).__name__}"
            )

        conditions = QwenImageConditions(text=self.text_embed.embed(texts))
        segment = self.diffusion.diffuse(
            conditions,
            schedule=params.sigmas.to(self.bundle.device),
            params=params,
            initial_latents=NoiseRecipe.from_sample(sample).resolve(),
        )
        filled = frontier.fill(
            segment=segment, primitives={"image": self.vae_decode.decode(segment)}, conditions=conditions.to_dict()
        )
        return sample.with_parts([*sample.parts[:-1], filled])


__all__ = ["QwenImage21Pipeline"]
