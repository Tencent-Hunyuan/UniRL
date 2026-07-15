"""HunyuanVideo15Pipeline — ``Sample → Sample`` end-to-end for HunyuanVideo-1.5.

Implements the four-tier flow::

    Texts ──text_embed (mllm + glyph, ×2 for CFG)──▶ HunyuanVideo15Conditions
        ──diffuse──▶ LatentSegment (6D video) ──vae_decode──▶ Videos

Hydra constructs a pipeline via
``HunyuanVideo15Pipeline.from_config(HunyuanVideo15PipelineConfig)``;
``from_config`` loads the :class:`HunyuanVideo15Bundle` then constructs
the stages with the precision policy and the vision-placeholder shape
constants from the config.

σ schedule contract
-------------------
The hosting engine (``TrainsideRolloutEngine`` / ``SGLangDiffusionRolloutEngine``
/ ``VLLMOmniRolloutEngine``) pins the σ schedule onto the gen Part's
``DiffusionSamplingParams.sigmas`` BEFORE calling ``generate(sample)``; this
pipeline reads ``params.sigmas`` and uses it
verbatim. HunyuanVideo-1.5 uses **static** flow-match shift (default
5.0); the engine builds
:meth:`FlowMatchSchedulePolicy.from_pretrained(path, shift=pipeline.shift)`
and the checkpoint's ``scheduler_config.json`` carries
``use_dynamic_shifting=False`` (unlike Qwen-Image / SD3.5), so the
policy stays on the static branch.

Negative prompts (CFG-on contract)
----------------------------------
HunyuanVideo-1.5's CFG is part of its inference contract — the upstream
pipeline ALWAYS encodes a negative branch (defaulting to empty strings
when not provided). This pipeline preserves this behavior: user-supplied
negatives are deferred, and when ``guidance_scale > 1.0`` we synthesize
``Texts(texts=[""] * batch_size)`` so the diffusion stage always has both
``negative_text_mllm`` and ``negative_text_glyph`` populated.
"""

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
    """HunyuanVideo-1.5 generate pipeline (T2V; I2V deferred): ``Sample → Sample``.

    Consumes a request ``Sample`` whose frontier Part is a pre-forked diffusion gen
    shell carrying ``DiffusionSamplingParams`` (with ``sigmas`` pinned by the
    hosting engine). Reads the prompt via ``sample.conditioning()`` and fills the
    frontier Part:

    - ``segment: LatentSegment`` — the denoising trajectory.
    - ``primitives["video"]: Videos`` — the decoded videos.

    ``Part.conditions`` carries the encoded conditions for trainer-side replay (the train stack re-types them via ``conditions_cls.from_dict``). User-supplied negatives are
    deferred; CFG uses a synthesized empty negative across the MLLM + Glyph encoders.
    """

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
        # ``shift`` is retained as an attribute so the hosting engine can
        # build the FlowMatchSchedulePolicy at startup. Static shift only
        # (HunyuanVideo-1.5 doesn't use dynamic mu).
        self.shift = shift

    @classmethod
    def latent_shape(cls, *, model_config: Any, sampling_spec: Any) -> tuple:
        """Per-sample 5D latent shape ``(C, T_lat, H_lat, W_lat)`` for
        driver-side noise pre-computation. Mirrors
        :meth:`HunyuanVideo15DiffusionStage._latent_shape`.

        Channel count is read from ``model_config.latent_channels``
        first; when the YAML leaves it ``None`` we fall back to
        :attr:`HunyuanVideo15DiffusionStage.DEFAULT_LATENT_CHANNELS`
        (32, matching ``hunyuanvideo-community/HunyuanVideo-1.5-Diffusers``).
        The stage init then receives the same config value and cross-
        checks against ``vae.config.latent_channels``; if the driver
        allocates ``C_d`` channels but the stage resolves to ``C_s != C_d``,
        ``diffuse(initial_latents=...)`` fails the shape check with a
        clear error — there is no silent drift.

        Spatial / temporal downsample factors are reused from the stage
        class constants (16× spatial, 4× temporal on the canonical VAE).
        ``T_lat = (num_frames - 1) // 4 + 1``.
        """
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
        """Build the full pipeline from a config.

        ``strategy`` is the SDE step strategy. Defaults to
        :class:`DanceSDEStrategy` (legacy HunyuanVideo-1.5 default); callers running
        GRPO with a different SDE family (Flow / CPS / DPM2) pass an
        explicit strategy built from ``cfg.sampling.sde_strategy``.
        """
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
            # Pass through the config-side override so the stage uses the
            # same channel count the driver assumed in ``latent_shape``.
            # When ``None``, the stage's existing VAE/transformer
            # inference takes over.
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
        """Encode prompts (MLLM + Glyph, + optional CFG negatives) into ``HunyuanVideo15Conditions``.

        CFG empty negative: when CFG is on (``guidance_scale > 1``) and
        caller didn't supply a negative, default to ``[""] * B``. When
        CFG is off, leave ``negatives=None`` so the negative branch is
        skipped entirely — saves two text-encoder forwards (MLLM +
        Glyph) per request. ``HunyuanVideo15DiffusionStep.predict_noise``
        (diffusion.py:188) already gates the CFG branch on
        ``guidance_scale > 1 and negative_text_mllm is not None``, so
        passing None for both negative_text_* is the canonical CFG-off
        signal. Either-both-or-both-None: the diffusion step (line 191-202)
        raises if only one of mllm/glyph is set.
        """
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
        """Run HunyuanVideo-1.5 T2V end-to-end, filling the frontier (pre-forked) gen Part.

        Requires σ to be pinned onto the gen part's ``DiffusionSamplingParams.sigmas``
        by the hosting engine before the call; see the σ ownership note in
        ``unirl.models.types.pipeline``.
        """
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

        # Driver-authoritative x_T via the model-aware recipe (NoiseRecipe); a
        # pre-shipped initial_latents tensor (on the gen part's segment) still wins.
        initial_latents = NoiseRecipe.from_sample(sample).resolve()

        latent_seg = self.diffusion.diffuse(hv_conds, schedule=schedule, params=params, initial_latents=initial_latents)
        videos = self.vae_decode.decode(latent_seg)

        # Fill the frontier shell, carrying the encoded conditions for trainer-side
        # replay (FlowGRPO re-types Part.conditions via conditions_cls.from_dict).
        filled = frontier.fill(segment=latent_seg, primitives={"video": videos}, conditions=hv_conds.to_dict())
        return Sample(parts=[*sample.parts[:-1], filled], reward_compute_s=sample.reward_compute_s)


__all__ = ["HunyuanVideo15Pipeline"]
