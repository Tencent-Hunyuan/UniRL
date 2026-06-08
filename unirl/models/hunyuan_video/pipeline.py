"""HunyuanVideoPipeline -- RolloutReq -> RolloutResp end-to-end for HunyuanVideo-1.0.

Implements the four-tier flow::

    Texts --text_embed (llama + clip)--> HunyuanVideoConditions
        --diffuse--> LatentSegment (6D video) --vae_decode--> Videos

Hydra constructs a pipeline via
``HunyuanVideoPipeline.from_config(HunyuanVideoPipelineConfig)``;
``from_config`` loads the :class:`HunyuanVideoBundle` then constructs
the stages with the precision policy from the config.

sigma schedule contract
-----------------------
The hosting engine (``TrainsideRolloutEngine`` / ``SGLangRolloutEngine``
/ ``VLLMOmniRolloutEngine``) pins ``req.sigmas`` via
:func:`unirl.sde.runtime.ensure_req_sigmas` BEFORE calling
``generate(req)``; this pipeline reads ``req.sigmas`` and uses it
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
from unirl.types.primitives import Texts
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp, RolloutTrack
from unirl.types.sampling import DiffusionSamplingParams, get_diffusion_params

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
    """HunyuanVideo-1.0 generate pipeline (T2V).

    Reads from ``RolloutReq``:

    - ``primitives["text"]: Texts`` -- required prompts.
    - ``sampling_params: DiffusionSamplingParams`` -- per-rollout sampling
      knobs (steps / guidance / size / num_frames / eta / sde_indices /
      ...). Read via :func:`get_diffusion_params`.
    - ``sigmas: Tensor[T+1]`` -- pinned by the engine adapter (required).

    Writes to ``RolloutResp``:

    - ``conditions["text_llama" | "pooled_clip"]: TextEmbedCondition``.
    - ``rollout_traces["video"]: LatentSegment``.
    - ``decoded["video"]: Videos``.
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
        # ``shift`` is retained as an attribute so the hosting engine can
        # build the FlowMatchSchedulePolicy at startup. Static shift only
        # (HunyuanVideo-1.0 doesn't use dynamic mu).
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
        )
        step = HunyuanVideoDiffusionStep()
        diffusion = HunyuanVideoDiffusionStage(
            model=bundle,
            step=step,
            strategy=strategy if strategy is not None else DanceSDEStrategy(),
            autocast_precision=config.autocast_precision,
            trajectory_precision=config.trajectory_precision,
            logprob_precision=config.logprob_precision,
            # Pass through the config-side override so the stage uses the
            # same channel count the driver assumed in ``latent_shape``.
            # When ``None``, the stage's existing VAE/transformer
            # inference takes over.
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

    def generate(self, req: RolloutReq) -> RolloutResp:
        """Run HunyuanVideo-1.0 T2V end-to-end. Requires ``req.sigmas`` to
        be pinned by the hosting engine adapter."""
        if req.sigmas is None:
            raise ValueError(
                "HunyuanVideoPipeline.generate: req.sigmas is None. The hosting "
                "engine (Trainside / SGLang / VLLMOmni) must call "
                "unirl.sde.runtime.ensure_req_sigmas(req, policy) before "
                "invoking pipeline.generate."
            )
        texts = req.primitives.get("text")
        if not isinstance(texts, Texts):
            raise TypeError(
                f"HunyuanVideoPipeline.generate: req.primitives['text'] must be "
                f"Texts, got {type(texts).__name__ if texts is not None else 'None'}"
            )

        params: DiffusionSamplingParams = get_diffusion_params(req.sampling_params)

        # Encode texts via LLaMA + CLIP (no negative branch needed).
        text_llama = self.text_embed.embed_llama(texts)
        pooled_clip = self.text_embed.embed_clip(texts)

        hv_conds = HunyuanVideoConditions(
            text_llama=text_llama,
            pooled_clip=pooled_clip,
        )

        schedule = req.sigmas.to(self.bundle.device)

        initial_cond = (req.request_conditions or {}).get("initial_latents")
        initial_latents = getattr(initial_cond, "latents", None) if initial_cond is not None else None

        latent_seg = self.diffusion.diffuse(hv_conds, schedule=schedule, params=params, initial_latents=initial_latents)
        videos = self.vae_decode.decode(latent_seg)

        return RolloutResp(
            tracks={
                "video": RolloutTrack(
                    sample_ids=list(req.sample_ids),
                    parent_ids=list(req.group_ids),
                    conditions=hv_conds.to_dict(),
                    segment=latent_seg,
                    decoded=videos,
                ),
            }
        )


__all__ = ["HunyuanVideoPipeline"]
