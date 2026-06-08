"""SD3Pipeline — RolloutReq → RolloutResp end-to-end for SD3.

Implements the new four-tier flow::

    Texts ──text_embed──▶ SD3Conditions ──diffuse──▶ LatentSegment ──vae_decode──▶ Images

Hydra constructs a pipeline via ``SD3Pipeline.from_config(SD3PipelineConfig)``
(see ``config.py``); ``from_config`` loads the ``SD3Bundle`` then constructs
the four stages with the precision policy from the config.

σ schedule contract
-------------------
The hosting engine (``TrainsideRolloutEngine`` / ``SGLangRolloutEngine`` /
``VLLMOmniRolloutEngine``) pins ``req.sigmas`` via
:func:`unirl.sde.runtime.ensure_req_sigmas` BEFORE calling
``generate(req)``; this pipeline reads ``req.sigmas`` and uses it
verbatim. The pipeline neither owns a σ builder nor reads model-
specific scheduler config — both responsibilities live in
:class:`unirl.sde.runtime.FlowMatchSchedulePolicy` which the
engine loads once at startup.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

from unirl.models.types.pipeline import Pipeline
from unirl.sde.kernels import CPSSDEStrategy, StepStrategy
from unirl.types.noise_recipe import NoiseRecipe
from unirl.types.primitives import Texts
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp, RolloutTrack
from unirl.types.sampling import DiffusionSamplingParams, get_diffusion_params

from .bundle import SD3Bundle
from .conditions import SD3Conditions
from .config import SD3PipelineConfig
from .diffusion import SD3DiffusionStage, SD3DiffusionStep
from .text_embed import SD3TextEmbedStage
from .vae import SD3VAEDecodeStage


class SD3Pipeline(Pipeline):
    """SD3 generate pipeline.

    Reads from ``RolloutReq``:

    - ``primitives["text"]: Texts`` — required prompts.
    - ``primitives["negative_text"]: Texts`` — optional CFG negatives.
    - ``stage_params["diffusion"]: dict`` — kwargs for
      :class:`SD3DiffusionParams`.
    - ``sigmas: Tensor[T+1]`` — pinned by the engine adapter (required).

    Writes to ``RolloutResp`` (single ``"image"`` track):

    - ``conditions["text"]: TextEmbedCondition``; plus
      ``conditions["negative_text"]: TextEmbedCondition`` when negative prompts
      were supplied.
    - ``segment: LatentSegment``.
    - ``decoded: Images``.
    """

    def __init__(
        self,
        *,
        bundle: SD3Bundle,
        text_embed: Optional[SD3TextEmbedStage] = None,
        diffusion: Optional[SD3DiffusionStage] = None,
        vae_decode: Optional[SD3VAEDecodeStage] = None,
        strategy: Optional[StepStrategy] = None,
        shift: float = 3.0,
        autocast_precision: str = "bf16",
        trajectory_precision: str = "fp16",
        logprob_precision: str = "fp32",
    ) -> None:
        super().__init__()
        self.bundle = bundle
        self.text_embed = text_embed if text_embed is not None else SD3TextEmbedStage(bundle)
        if diffusion is None:
            diffusion = SD3DiffusionStage(
                model=bundle,
                step=SD3DiffusionStep(),
                strategy=strategy if strategy is not None else CPSSDEStrategy(),
                autocast_precision=autocast_precision,
                trajectory_precision=trajectory_precision,
                logprob_precision=logprob_precision,
            )
        self.diffusion = diffusion
        self.vae_decode = vae_decode if vae_decode is not None else SD3VAEDecodeStage(bundle)
        # ``shift`` is retained as an attribute so the hosting engine
        # (TrainsideRolloutEngine) can read it when constructing the
        # FlowMatchSchedulePolicy at startup. It is NOT used by
        # ``generate`` itself.
        self.shift = shift

    @classmethod
    def latent_shape(cls, *, model_config: Any, sampling_spec: Any) -> tuple:
        """Per-sample latent shape ``(C, H_lat, W_lat)`` for driver-side
        noise pre-computation. SD3 / SD3.5: 16-channel z, /8 spatial."""
        height = int(sampling_spec.height)
        width = int(sampling_spec.width)
        return (16, height // 8, width // 8)

    @classmethod
    def from_config(
        cls,
        config: SD3PipelineConfig,
        *,
        strategy: Optional[StepStrategy] = None,
    ) -> "SD3Pipeline":
        """Build the full pipeline from a config.

        ``strategy`` is the SDE step strategy. Defaults to
        :class:`CPSSDEStrategy` (legacy SD3 default in
        ``samplers/fsdp/sd3_sampler.py:139``); callers running GRPO with
        Flow / Dance / DPM2 should pass an explicit strategy built from
        ``cfg.sampling.sde_strategy``.
        """
        bundle = SD3Bundle.from_config(config)
        return cls(
            bundle=bundle,
            strategy=strategy,
            shift=float(config.shift),
            autocast_precision=config.autocast_precision,
            trajectory_precision=config.trajectory_precision,
            logprob_precision=config.logprob_precision,
        )

    def generate(self, req: RolloutReq) -> RolloutResp:
        """Run SD3 t2i end-to-end. Requires ``req.sigmas`` to be pinned by
        the hosting engine adapter."""
        if req.sigmas is None:
            raise ValueError(
                "SD3Pipeline.generate: req.sigmas is None. The hosting engine "
                "(Trainside / SGLang / VLLMOmni) must call "
                "unirl.sde.runtime.ensure_req_sigmas(req, policy) before "
                "invoking pipeline.generate; see the σ ownership note in "
                "unirl.models.types.pipeline."
            )
        texts = req.primitives.get("text")
        if not isinstance(texts, Texts):
            raise TypeError(
                f"SD3Pipeline.generate: req.primitives['text'] must be Texts, "
                f"got {type(texts).__name__ if texts is not None else 'None'}"
            )
        negatives_raw = req.primitives.get("negative_text")
        negatives = negatives_raw if isinstance(negatives_raw, Texts) else None
        if negatives is not None and len(negatives.texts) != len(texts.texts):
            raise ValueError(
                f"SD3Pipeline.generate: negative_text length {len(negatives.texts)} != text length {len(texts.texts)}"
            )

        params: DiffusionSamplingParams = get_diffusion_params(req.sampling_params)
        # init_same_noise shares the initial latent within each prompt group. The
        # group key is the per-sample group id, which rides on the (already-sliced)
        # req — surface it to the noise sampler when the driver didn't pre-ship
        # noise_group_ids on sampling_params (a shared_field that isn't batch-sliced).
        if bool(params.init_same_noise) and not params.noise_group_ids:
            params = dataclasses.replace(params, noise_group_ids=list(req.group_ids))

        text_cond = self.text_embed.embed(texts)
        # CFG empty negative: SD3 upstream (diffusers v0.37.1
        # ``pipeline_stable_diffusion_3.py:466-467``) auto-defaults to
        # ``""`` (empty string) when CFG is enabled and no negative is
        # passed. Without this default the SD3 diffusion step would fall
        # back to a zero-init negative-condition path that doesn't match
        # what the model was trained against; the rollout/replay log-prob
        # ratio drifts away from 1.0 in GRPO.
        #
        # SD3's three text encoders (CLIP + CLIP + T5) tokenize ``""``
        # cleanly — unlike Qwen-Image, there's no chat-template +
        # prefix-strip that would degenerate the embedding. Hence the
        # value ``""`` here vs Qwen's ``" "``; both are the model's
        # canonical empty-negative per its upstream pipeline.
        if negatives is None and float(params.guidance_scale) > 1.0:
            negatives = Texts(texts=[""] * len(texts.texts))
        negative_text_cond = self.text_embed.embed(negatives) if negatives is not None else None
        sd3_conds = SD3Conditions(text=text_cond, negative_text=negative_text_cond)

        schedule = req.sigmas.to(self.bundle.device)

        # Driver-authoritative x_T via the model-aware recipe (NoiseRecipe); a
        # pre-shipped initial_latents tensor (img2img / i2v first-frame) still wins.
        initial_latents = NoiseRecipe.from_rollout_req(req).resolve()

        latent_seg = self.diffusion.diffuse(
            sd3_conds, schedule=schedule, params=params, initial_latents=initial_latents
        )
        images = self.vae_decode.decode(latent_seg)

        return RolloutResp(
            tracks={
                "image": RolloutTrack(
                    sample_ids=list(req.sample_ids),
                    parent_ids=list(req.group_ids),
                    conditions=sd3_conds.to_dict(),
                    segment=latent_seg,
                    decoded=images,
                ),
            }
        )


__all__ = ["SD3Pipeline"]
