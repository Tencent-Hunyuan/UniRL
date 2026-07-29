"""BooguImagePipeline — ``Sample → Sample`` end-to-end for Boogu-Image.

Implements the four-tier flow::

    Texts ──text_embed──▶ BooguImageConditions ──diffuse──▶ LatentSegment
                                                                │
                                                                ▼
                                                            vae_decode
                                                                │
                                                                ▼
                                                              Images

Hydra constructs a pipeline via
``BooguImagePipeline.from_config(BooguImagePipelineConfig)`` (see
``config.py``); ``from_config`` loads the :class:`BooguImageBundle` then
constructs the stages with the precision policy from the config.

σ schedule contract
-------------------
The hosting engine (``TrainsideRolloutEngine``) pins the σ schedule onto the
gen Part's ``DiffusionSamplingParams.sigmas`` BEFORE calling
``generate(sample)``; this pipeline reads ``params.sigmas`` verbatim. The released
``Boogu-Image-0.1-Base`` scheduler config is a **static v1 time shift**
(``do_shift: true, dynamic_time_shift: false, time_shift_version: "v1",
seq_len: 4096``) whose sigma-space form is exactly the standard static shift
``σ' = s·σ / (1 + (s−1)·σ)`` with ``s = e^{lin(seq_len)} = e^{1.15}``
(verified to 1 fp32 ulp against the reference scheduler at N ∈ {12, 50}).
:meth:`build_schedule_policy` pins that static posture from ``self.shift``
— Boogu's ``scheduler_config.json`` uses custom field names that the base
``FlowMatchSchedulePolicy.from_pretrained`` would silently ignore, so the
config-pinned ``static_only`` posture (z_image / bagel precedent) is the
safe wiring.

CFG convention
--------------
Boogu's guidance-off value is ``guidance_scale = 1.0`` (the combine is
``pred + (g−1)·(pred − pred_neg)``), so :meth:`build_conditions` gates the
auto-``""`` negative on ``guidance_scale > 1.0`` — a deliberate divergence
from z_image's ``> 0.0`` gate. The empty negative ``""`` is re-encoded
through the chat template, where it routes to the DROP system prompt (see
``text_embed.py``).
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

from unirl.models.types.pipeline import Pipeline
from unirl.sde.kernels import FlowSDEStrategy, StepStrategy
from unirl.types.noise_recipe import NoiseRecipe
from unirl.types.primitives import Texts
from unirl.types.sample import Sample
from unirl.types.sampling import DiffusionSamplingParams

from .bundle import BooguImageBundle
from .conditions import BooguImageConditions
from .config import BOOGU_IMAGE_BASE_STATIC_SHIFT, BooguImagePipelineConfig
from .diffusion import BooguImageDiffusionStage, BooguImageDiffusionStep
from .text_embed import BooguImageTextEmbedStage
from .vae import BooguImageVAEDecodeStage


class BooguImagePipeline(Pipeline):
    """Boogu-Image generate pipeline: ``Sample → Sample``.

    Consumes a request ``Sample`` whose frontier Part is a pre-forked diffusion
    gen shell carrying ``DiffusionSamplingParams`` (with ``sigmas`` pinned by
    the hosting engine). Reads the prompt via ``sample.conditioning()`` and
    fills the frontier Part:

    - ``segment: LatentSegment`` — the denoising trajectory.
    - ``primitives["image"]: Images`` — the decoded images.

    ``Part.conditions`` carries the encoded conditions for trainer-side replay
    (the train stack re-types them via ``conditions_cls.from_dict``).
    User-supplied negatives are deferred; CFG (``guidance_scale > 1.0``) uses a
    synthesized empty negative routed to the DROP system prompt.
    """

    def __init__(
        self,
        *,
        bundle: BooguImageBundle,
        text_embed: Optional[BooguImageTextEmbedStage] = None,
        diffusion: Optional[BooguImageDiffusionStage] = None,
        vae_decode: Optional[BooguImageVAEDecodeStage] = None,
        strategy: Optional[StepStrategy] = None,
        shift: float = BOOGU_IMAGE_BASE_STATIC_SHIFT,
        autocast_precision: str = "bf16",
        trajectory_precision: str = "fp16",
        logprob_precision: str = "fp32",
        max_sequence_length: int = 1280,
    ) -> None:
        super().__init__()
        self.bundle = bundle
        self.text_embed = (
            text_embed
            if text_embed is not None
            else BooguImageTextEmbedStage(bundle, max_sequence_length=max_sequence_length)
        )
        if diffusion is None:
            diffusion = BooguImageDiffusionStage(
                model=bundle,
                step=BooguImageDiffusionStep(),
                strategy=strategy if strategy is not None else FlowSDEStrategy(),
                autocast_precision=autocast_precision,
                trajectory_precision=trajectory_precision,
                logprob_precision=logprob_precision,
            )
        self.diffusion = diffusion
        self.vae_decode = vae_decode if vae_decode is not None else BooguImageVAEDecodeStage(bundle)
        # Retained so the hosting engine can read it when constructing the
        # FlowMatchSchedulePolicy at startup (static shift, e^{1.15}).
        self.shift = shift

    @classmethod
    def latent_shape(cls, *, model_config: Any, sampling_spec: Any) -> tuple:
        """Per-sample latent shape ``(C, H_lat, W_lat)`` for driver-side
        noise pre-computation. Boogu-Image: 16-channel FLUX ``AutoencoderKL``,
        8× spatial downsample with the patchify-2×2 rounding
        (``latent_h = 2 * (H // 16)``)."""
        height = int(sampling_spec.height)
        width = int(sampling_spec.width)
        vae_scale_factor = 8  # FLUX.1 AutoencoderKL with 4 block_out_channels
        latent_h = 2 * (height // (vae_scale_factor * 2))
        latent_w = 2 * (width // (vae_scale_factor * 2))
        return (16, latent_h, latent_w)

    @classmethod
    def from_config(
        cls,
        config: BooguImagePipelineConfig,
        *,
        strategy: Optional[StepStrategy] = None,
    ) -> "BooguImagePipeline":
        """Build the full pipeline from a config.

        ``strategy`` is the SDE step strategy. Defaults to
        :class:`FlowSDEStrategy`; callers running GRPO with a different SDE
        family (Dance / CPS / DPM2) pass an explicit strategy.
        """
        bundle = BooguImageBundle.from_config(config)
        text_embed = BooguImageTextEmbedStage(bundle, max_sequence_length=config.max_sequence_length)
        step = BooguImageDiffusionStep()
        diffusion = BooguImageDiffusionStage(
            model=bundle,
            step=step,
            strategy=strategy if strategy is not None else FlowSDEStrategy(),
            autocast_precision=config.autocast_precision,
            trajectory_precision=config.trajectory_precision,
            logprob_precision=config.logprob_precision,
        )
        vae_decode = BooguImageVAEDecodeStage(bundle)
        return cls(
            bundle=bundle,
            text_embed=text_embed,
            diffusion=diffusion,
            vae_decode=vae_decode,
            shift=float(config.shift),
            max_sequence_length=config.max_sequence_length,
        )

    def build_schedule_policy(self):
        """Static-shift FlowMatch σ policy for the released Boogu-Image-0.1.

        Boogu's released Base scheduler is static v1 with ``seq_len: 4096``:
        μ = lin(4096) = 1.15 (the reference hardcodes the 256→0.5 / 4096→1.15
        linear map), and v1's logistic t-shift equals the standard static
        sigma shift with ``s = e^μ = 3.158192909689768``. Returning an
        explicit ``static_only`` policy built from ``self.shift`` pins that
        posture regardless of whether ``pretrained_path`` is an HF repo id or
        a local mount — Boogu's ``scheduler_config.json`` uses custom field
        names (``do_shift`` / ``dynamic_time_shift`` / ``time_shift_version``
        / ``seq_len``) that the base policy's JSON loader would silently
        ignore, which would otherwise produce a wrong-shift schedule without
        an error. Mirrors ``ZImagePipeline.build_schedule_policy``.
        """
        from unirl.sde.runtime import FlowMatchSchedulePolicy

        return FlowMatchSchedulePolicy.static_only(float(self.shift))

    def build_conditions(
        self,
        texts: Texts,
        *,
        negatives: Optional[Texts] = None,
        guidance_scale: float = 1.0,
    ) -> BooguImageConditions:
        """Encode prompts (+ optional CFG negatives) into ``BooguImageConditions``.

        CFG empty negative: the reference ``encode_instruction`` defaults the
        negative instruction to ``""`` when CFG is active
        (pipeline_boogu.py:2491-2494); inside the embed stage the empty
        string routes to the DROP system prompt (dataset logic). Boogu gates
        CFG on ``guidance_scale > 1.0`` — 1.0 is guidance-off (unlike
        z_image's ``> 0.0`` gate).
        """
        if negatives is not None and len(negatives.texts) != len(texts.texts):
            raise ValueError(
                f"BooguImagePipeline.build_conditions: negative_text length "
                f"{len(negatives.texts)} != text length {len(texts.texts)}"
            )
        text_cond = self.text_embed.embed(texts)
        if negatives is None and float(guidance_scale) > 1.0:
            negatives = Texts(texts=[""] * len(texts.texts))
        negative_text_cond = self.text_embed.embed(negatives) if negatives is not None else None
        return BooguImageConditions(text=text_cond, negative_text=negative_text_cond)

    def generate(self, sample: Sample) -> Sample:
        """Run Boogu-Image t2i end-to-end, filling the frontier (pre-forked) gen Part.

        Requires σ to be pinned onto the gen part's ``DiffusionSamplingParams.sigmas``
        by the hosting engine before the call; see the σ ownership note in
        ``unirl.models.types.pipeline``.
        """
        frontier = sample.parts[-1]
        params = frontier.sampling_params
        if not isinstance(params, DiffusionSamplingParams):
            raise TypeError(
                f"BooguImagePipeline.generate: frontier gen Part must carry DiffusionSamplingParams, "
                f"got {type(params).__name__ if params is not None else 'None'}"
            )
        if params.sigmas is None:
            raise ValueError(
                "BooguImagePipeline.generate: gen part sampling_params.sigmas is None. The hosting "
                "engine must pin σ before invoking pipeline.generate; see the σ ownership note "
                "in unirl.models.types.pipeline."
            )

        conditioning = sample.conditioning()
        texts = conditioning[0] if conditioning else None
        if not isinstance(texts, Texts):
            raise TypeError(
                f"BooguImagePipeline.generate: expected a Texts prompt from sample.conditioning()[0], "
                f"got {type(texts).__name__ if texts is not None else 'None'}"
            )

        if bool(params.init_same_noise) and not params.noise_group_ids:
            params = dataclasses.replace(params, noise_group_ids=list(frontier.group_ids))

        conds = self.build_conditions(texts, guidance_scale=float(params.guidance_scale))
        schedule = params.sigmas.to(self.bundle.device)

        # Driver-authoritative x_T via the model-aware recipe (NoiseRecipe); a
        # pre-shipped initial_latents tensor still wins.
        initial_latents = NoiseRecipe.from_sample(sample).resolve()

        latent_seg = self.diffusion.diffuse(conds, schedule=schedule, params=params, initial_latents=initial_latents)
        images = self.vae_decode.decode(latent_seg)

        # Fill the frontier shell, carrying the encoded conditions for trainer-side
        # replay (FlowGRPO re-types Part.conditions via conditions_cls.from_dict).
        filled = frontier.fill(segment=latent_seg, primitives={"image": images}, conditions=conds.to_dict())
        return Sample(parts=[*sample.parts[:-1], filled], reward_compute_s=sample.reward_compute_s)


__all__ = ["BooguImagePipeline"]
