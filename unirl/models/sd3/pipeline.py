"""SD3Pipeline — ``Sample → Sample`` end-to-end for SD3.

Implements the new four-tier flow::

    Texts ──text_embed──▶ SD3Conditions ──diffuse──▶ LatentSegment ──vae_decode──▶ Images

Hydra constructs a pipeline via ``SD3Pipeline.from_config(SD3PipelineConfig)``
(see ``config.py``); ``from_config`` loads the ``SD3Bundle`` then constructs
the four stages with the precision policy from the config.

σ schedule contract
-------------------
The hosting engine (``TrainsideRolloutEngine`` / ``SGLangDiffusionRolloutEngine`` /
``VLLMOmniRolloutEngine``) pins the σ schedule onto the gen part's
``DiffusionSamplingParams.sigmas`` BEFORE calling ``generate(sample)``; this
pipeline reads ``params.sigmas`` and uses it verbatim. The pipeline neither owns
a σ builder nor reads model-specific scheduler config — both responsibilities
live in :class:`unirl.sde.runtime.FlowMatchSchedulePolicy` which the engine
loads once at startup.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

from unirl.models.types.pipeline import Pipeline
from unirl.sde.kernels import CPSSDEStrategy, StepStrategy
from unirl.types.noise_recipe import NoiseRecipe
from unirl.types.primitives import Texts
from unirl.types.sample import Sample
from unirl.types.sampling import DiffusionSamplingParams

from .bundle import SD3Bundle
from .conditions import SD3Conditions
from .config import SD3PipelineConfig
from .diffusion import SD3DiffusionStage, SD3DiffusionStep
from .text_embed import SD3TextEmbedStage
from .vae import SD3VAEDecodeStage, SD3VAEEncodeStage


class SD3Pipeline(Pipeline):
    """SD3 generate pipeline: ``Sample → Sample``.

    Consumes a request ``Sample`` whose frontier (last) Part is a pre-forked
    diffusion gen shell carrying ``DiffusionSamplingParams`` (with ``sigmas``
    pinned by the hosting engine). Reads the prompt via ``sample.conditioning()``
    and fills the frontier Part:

    - ``segment: LatentSegment`` — the denoising trajectory.
    - ``primitives["image"]: Images`` — the decoded images.

    ``Part.conditions`` carries the encoded conditions for trainer-side replay.
    Sample generation uses the model's default CFG negative; direct callers can
    pass explicit negatives through :meth:`build_conditions`.
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
        batch_replay_steps: bool = False,
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
                batch_replay_steps=batch_replay_steps,
            )
        self.diffusion = diffusion
        self.vae_decode = vae_decode if vae_decode is not None else SD3VAEDecodeStage(bundle)
        # Encode side of the codec (target images → clean latents). Rollout
        # never calls it; diffusion SFT / future img2img conditioning do.
        self.vae_encode = SD3VAEEncodeStage(bundle)
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
            batch_replay_steps=config.batch_replay_steps,
        )

    def build_conditions(
        self,
        texts: Texts,
        *,
        negatives: Optional[Texts] = None,
        guidance_scale: float = 1.0,
    ) -> SD3Conditions:
        """Encode prompts (+ optional CFG negatives) into ``SD3Conditions``.

        Shared by :meth:`generate` and grad-sampling adaptations (e.g. the
        experimental ReFL pipeline).
        Applies SD3's empty-negative default (diffusers parity) when CFG is on and
        no negative was supplied — see the rationale quoted in :meth:`generate`.
        """
        if negatives is not None and len(negatives.texts) != len(texts.texts):
            raise ValueError(
                f"SD3Pipeline.build_conditions: negative_text length "
                f"{len(negatives.texts)} != text length {len(texts.texts)}"
            )
        text_cond = self.text_embed.embed(texts)
        if negatives is None and float(guidance_scale) > 1.0:
            negatives = Texts(texts=[""] * len(texts.texts))
        negative_text_cond = self.text_embed.embed(negatives) if negatives is not None else None
        return SD3Conditions(text=text_cond, negative_text=negative_text_cond)

    def generate(self, sample: Sample) -> Sample:
        """Run SD3 t2i end-to-end, filling the frontier (pre-forked) gen Part.

        Requires σ to be pinned onto the gen part's ``DiffusionSamplingParams.sigmas``
        by the hosting engine (e.g. ``TrainsideRolloutEngine._ensure_sample_sigmas``)
        before the call; see the σ ownership note in ``unirl.models.types.pipeline``.
        """
        frontier = sample.parts[-1]
        params = frontier.sampling_params
        if not isinstance(params, DiffusionSamplingParams):
            raise TypeError(
                f"SD3Pipeline.generate: frontier gen Part must carry DiffusionSamplingParams, "
                f"got {type(params).__name__ if params is not None else 'None'}"
            )
        if params.sigmas is None:
            raise ValueError(
                "SD3Pipeline.generate: gen part sampling_params.sigmas is None. The hosting "
                "engine must pin σ before invoking pipeline.generate; see the σ ownership note "
                "in unirl.models.types.pipeline."
            )

        conditioning = sample.conditioning()
        texts = conditioning[0] if conditioning else None
        if not isinstance(texts, Texts):
            raise TypeError(
                f"SD3Pipeline.generate: expected a Texts prompt from sample.conditioning()[0], "
                f"got {type(texts).__name__ if texts is not None else 'None'}"
            )

        # init_same_noise shares the initial latent within each prompt group; surface
        # the gen part's group ids to the noise sampler when the driver didn't pre-ship
        # noise_group_ids on sampling_params (a shared_field that isn't batch-sliced).
        if bool(params.init_same_noise) and not params.noise_group_ids:
            params = dataclasses.replace(params, noise_group_ids=list(frontier.group_ids))

        sd3_conds = self.build_conditions(texts, guidance_scale=float(params.guidance_scale))
        schedule = params.sigmas.to(self.bundle.device)

        # Driver-authoritative x_T via the model-aware recipe (NoiseRecipe); a
        # pre-shipped initial_latents tensor (img2img / i2v first-frame) still wins.
        initial_latents = NoiseRecipe.from_sample(sample).resolve()

        latent_seg = self.diffusion.diffuse(
            sd3_conds, schedule=schedule, params=params, initial_latents=initial_latents
        )
        images = self.vae_decode.decode(latent_seg)

        # Fill the frontier shell, carrying the encoded conditions for trainer-side
        # replay: Part.conditions is the train stack's source in prepare_segment
        # (matches every sibling pipeline + the SD3 sglang_diffusion adapter).
        filled = frontier.fill(segment=latent_seg, primitives={"image": images}, conditions=sd3_conds.to_dict())
        return Sample(parts=[*sample.parts[:-1], filled], reward_compute_s=sample.reward_compute_s)


__all__ = ["SD3Pipeline"]
