"""HunyuanVideoPipeline -- ``Sample -> Sample`` end-to-end for HunyuanVideo-1.0.

Implements the four-tier flow::

    Texts --text_embed (llama + clip)--> HunyuanVideoConditions
        --diffuse--> LatentSegment (6D video) --vae_decode--> Videos

Hydra constructs a pipeline via
``HunyuanVideoPipeline.from_config(HunyuanVideoPipelineConfig)``;
``from_config`` loads the :class:`HunyuanVideoBundle` then constructs
the stages with the precision policy from the config.

sigma schedule contract
-----------------------
The hosting engine (``TrainsideRolloutEngine`` / ``SGLangDiffusionRolloutEngine``
/ ``VLLMOmniRolloutEngine``) pins the σ schedule onto the gen Part's
``DiffusionSamplingParams.sigmas`` BEFORE calling ``generate(sample)``; this
pipeline reads ``params.sigmas`` and uses it
verbatim. HunyuanVideo-1.0 uses **static** flow-match shift (default
5.0); the engine builds
:meth:`FlowMatchSchedulePolicy.from_pretrained(path, shift=pipeline.shift)`
and the checkpoint's ``scheduler_config.json`` carries
``use_dynamic_shifting=False``, so the policy stays on the static branch.

No negative prompts (guidance embedding)
-----------------------------------------
HunyuanVideo-1.0 uses guidance embedding (``guidance_embeds=True``) instead
of classifier-free guidance. The guidance scale is passed as a tensor to
the transformer -- there is NO negative branch encoding. This simplifies
the pipeline significantly vs HV15.
"""

from __future__ import annotations

from typing import Any, Optional

from unirl.models.types.pipeline import Pipeline
from unirl.sde.kernels import DanceSDEStrategy, StepStrategy
from unirl.types.noise_recipe import NoiseRecipe
from unirl.types.primitives import Texts
from unirl.types.sample import Sample
from unirl.types.sampling import DiffusionSamplingParams

from .bundle import HunyuanVideoBundle
from .conditions import HunyuanVideoConditions
from .config import HunyuanVideoPipelineConfig
from .diffusion import (
    HunyuanVideoDiffusionStage,
    HunyuanVideoDiffusionStep,
)
from .text_embed import HunyuanVideoTextEmbedStage
from .vae import HunyuanVideoVAEDecodeStage


class HunyuanVideoPipeline(Pipeline):
    """HunyuanVideo-1.0 generate pipeline (T2V): ``Sample → Sample``.

    Consumes a request ``Sample`` whose frontier Part is a pre-forked diffusion gen
    shell carrying ``DiffusionSamplingParams`` (with ``sigmas`` pinned by the
    hosting engine). Reads the prompt via ``sample.conditioning()`` and fills the
    frontier Part:

    - ``segment: LatentSegment`` (6D video) — the denoising trajectory.
    - ``primitives["video"]: Videos`` — the decoded videos.

    ``Part.conditions`` carries the encoded conditions for trainer-side replay (the train stack re-types them via ``conditions_cls.from_dict``). No CFG negative branch (T2V).
    """

    def __init__(
        self,
        *,
        bundle: HunyuanVideoBundle,
        text_embed: Optional[HunyuanVideoTextEmbedStage] = None,
        diffusion: Optional[HunyuanVideoDiffusionStage] = None,
        vae_decode: Optional[HunyuanVideoVAEDecodeStage] = None,
        strategy: Optional[StepStrategy] = None,
        shift: float = 5.0,
        autocast_precision: str = "bf16",
        trajectory_precision: str = "fp16",
        logprob_precision: str = "fp32",
        llama_max_length: int = 256,
        crop_start: int = 95,
        clip_max_length: int = 77,
        hidden_state_skip_layer: int = 2,
        latent_channels: Optional[int] = None,
    ) -> None:
        self.bundle = bundle
        self.text_embed = (
            text_embed
            if text_embed is not None
            else HunyuanVideoTextEmbedStage(
                bundle,
                llama_max_length=llama_max_length,
                clip_max_length=clip_max_length,
                crop_start=crop_start,
                hidden_state_skip_layer=hidden_state_skip_layer,
            )
        )
        if diffusion is None:
            diffusion = HunyuanVideoDiffusionStage(
                model=bundle,
                step=HunyuanVideoDiffusionStep(),
                strategy=strategy if strategy is not None else DanceSDEStrategy(),
                autocast_precision=autocast_precision,
                trajectory_precision=trajectory_precision,
                logprob_precision=logprob_precision,
                latent_channels=latent_channels,
            )
        self.diffusion = diffusion
        self.vae_decode = vae_decode if vae_decode is not None else HunyuanVideoVAEDecodeStage(bundle)
        self.shift = shift

    @classmethod
    def latent_shape(cls, *, model_config: Any, sampling_spec: Any) -> tuple:
        """Per-sample 5D latent shape ``(C, T_lat, H_lat, W_lat)`` for
        driver-side noise pre-computation. Mirrors
        :meth:`HunyuanVideoDiffusionStage._latent_shape`.

        Channel count is read from ``model_config.latent_channels``
        first; when the YAML leaves it ``None`` we fall back to
        :attr:`HunyuanVideoDiffusionStage.DEFAULT_LATENT_CHANNELS`
        (16, matching HunyuanVideo-1.0).
        """
        height = int(sampling_spec.height)
        width = int(sampling_spec.width)
        num_frames = int(sampling_spec.num_frames)
        spatial = HunyuanVideoDiffusionStage.DEFAULT_SPATIAL_DOWNSAMPLE
        temporal = HunyuanVideoDiffusionStage.DEFAULT_TEMPORAL_DOWNSAMPLE
        config_channels = getattr(model_config, "latent_channels", None)
        channels = (
            int(config_channels) if config_channels is not None else HunyuanVideoDiffusionStage.DEFAULT_LATENT_CHANNELS
        )
        latent_t = (num_frames - 1) // temporal + 1
        latent_h = max(1, height // spatial)
        latent_w = max(1, width // spatial)
        return (channels, latent_t, latent_h, latent_w)

    @classmethod
    def from_config(
        cls,
        config: HunyuanVideoPipelineConfig,
        *,
        strategy: Optional[StepStrategy] = None,
    ) -> "HunyuanVideoPipeline":
        """Build the full pipeline from a config.

        ``strategy`` is the SDE step strategy. Defaults to
        :class:`DanceSDEStrategy`; callers running GRPO with a different
        SDE family (Flow / CPS / DPM2) pass an explicit strategy built
        from ``cfg.sampling.sde_strategy``.
        """
        bundle = HunyuanVideoBundle.from_config(config)
        text_embed = HunyuanVideoTextEmbedStage(
            bundle,
            llama_max_length=config.llama_max_length,
            clip_max_length=config.clip_max_length,
            crop_start=config.crop_start,
            hidden_state_skip_layer=config.hidden_state_skip_layer,
        )
        step = HunyuanVideoDiffusionStep()
        diffusion = HunyuanVideoDiffusionStage(
            model=bundle,
            step=step,
            strategy=strategy if strategy is not None else DanceSDEStrategy(),
            autocast_precision=config.autocast_precision,
            trajectory_precision=config.trajectory_precision,
            logprob_precision=config.logprob_precision,
            latent_channels=config.latent_channels,
        )
        vae_decode = HunyuanVideoVAEDecodeStage(bundle)
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
    ) -> HunyuanVideoConditions:
        """Encode prompts (LLaMA + CLIP) into ``HunyuanVideoConditions``.

        HunyuanVideo-1.0 uses guidance embedding (``guidance_embeds=True``)
        instead of classifier-free guidance — there is NO negative branch,
        so ``negatives`` / ``guidance_scale`` are accepted for signature
        parity with the other families but unused here.
        """
        text_llama = self.text_embed.embed_llama(texts)
        pooled_clip = self.text_embed.embed_clip(texts)
        return HunyuanVideoConditions(
            text_llama=text_llama,
            pooled_clip=pooled_clip,
        )

    def generate(self, sample: Sample) -> Sample:
        """Run HunyuanVideo-1.0 T2V end-to-end, filling the frontier (pre-forked) gen Part.

        Requires σ to be pinned onto the gen part's ``DiffusionSamplingParams.sigmas``
        by the hosting engine before the call; see the σ ownership note in
        ``unirl.models.types.pipeline``.
        """
        frontier = sample.parts[-1]
        params = frontier.sampling_params
        if not isinstance(params, DiffusionSamplingParams):
            raise TypeError(
                f"HunyuanVideoPipeline.generate: frontier gen Part must carry DiffusionSamplingParams, "
                f"got {type(params).__name__ if params is not None else 'None'}"
            )
        if params.sigmas is None:
            raise ValueError(
                "HunyuanVideoPipeline.generate: gen part sampling_params.sigmas is None. The hosting "
                "engine must pin σ before invoking pipeline.generate; see the σ ownership note "
                "in unirl.models.types.pipeline."
            )

        conditioning = sample.conditioning()
        texts = conditioning[0] if conditioning else None
        if not isinstance(texts, Texts):
            raise TypeError(
                f"HunyuanVideoPipeline.generate: expected a Texts prompt from sample.conditioning()[0], "
                f"got {type(texts).__name__ if texts is not None else 'None'}"
            )

        hv_conds = self.build_conditions(texts)
        schedule = params.sigmas.to(self.bundle.device)

        initial_latents = NoiseRecipe.from_sample(sample).resolve()

        latent_seg = self.diffusion.diffuse(hv_conds, schedule=schedule, params=params, initial_latents=initial_latents)
        videos = self.vae_decode.decode(latent_seg)

        filled = frontier.fill(segment=latent_seg, primitives={"video": videos}, conditions=hv_conds.to_dict())
        return Sample(parts=[*sample.parts[:-1], filled], reward_compute_s=sample.reward_compute_s)


__all__ = ["HunyuanVideoPipeline"]
