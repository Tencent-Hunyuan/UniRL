"""WAN21Pipeline — ``Sample → Sample`` end-to-end for WAN 2.1 T2V/I2V.

Implements the new four-tier flow::

    Texts ──text_embed──▶ WAN21Conditions ──diffuse──▶ LatentSegment ──vae_decode──▶ Videos

Hydra constructs a pipeline via
``WAN21Pipeline.from_config(WAN21PipelineConfig)`` (see ``config.py``);
``from_config`` loads the ``WAN21Bundle`` then constructs the four
stages with the precision policy from the config.

Default SDE strategy is :class:`DanceSDEStrategy` (legacy WAN default in
``samplers/fsdp/wan_sampler.py::FSDPWanSampler.__init__``). Callers
running other strategies (Flow / CPS / DPM2) should pass an explicit
``strategy=`` built from ``cfg.sampling.sde_strategy``.

Schedule policy: WAN does NOT have a diffusers-side scheduler that
ships with the checkpoint (the bundle may set ``scheduler=None``); the
pipeline always uses :func:`unirl.sde.runtime.get_sigma_schedule`
with the configured ``shift``. This mirrors legacy
``samplers/fsdp/wan_sampler.py::sample()``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

from unirl.models.types.pipeline import Pipeline
from unirl.sde.kernels import DanceSDEStrategy, StepStrategy
from unirl.types.noise_recipe import NoiseRecipe
from unirl.types.primitives import Images, Texts
from unirl.types.sample import Sample
from unirl.types.sampling import DiffusionSamplingParams

from .bundle import WAN21Bundle
from .clip_vision_encode import WAN21CLIPVisionEncodeStage
from .conditions import WAN21Conditions
from .config import WAN21PipelineConfig
from .diffusion import WAN21DiffusionStage, WAN21DiffusionStep
from .image_encode import WAN21ImageLatentEncodeStage
from .text_embed import WAN21TextEmbedStage
from .vae import WAN21VAEDecodeStage


class WAN21Pipeline(Pipeline):
    """WAN 2.1 T2V/I2V generate pipeline: ``Sample → Sample``.

    Consumes a request ``Sample`` whose frontier Part is a pre-forked diffusion gen
    shell carrying ``DiffusionSamplingParams`` (with ``sigmas`` pinned by the
    hosting engine). Reads the prompt — and, for I2V, the chained first-frame image
    — via ``sample.conditioning()`` and fills the frontier Part:

    - ``segment: LatentSegment`` — the denoising trajectory.
    - ``primitives["video"]: Videos`` — the decoded videos.

    ``Part.conditions`` carries the encoded conditions for trainer-side replay (the train stack re-types them via ``conditions_cls.from_dict``). User-supplied text negatives are
    deferred; CFG uses a synthesized empty negative.
    """

    def __init__(
        self,
        *,
        bundle: WAN21Bundle,
        text_embed: Optional[WAN21TextEmbedStage] = None,
        diffusion: Optional[WAN21DiffusionStage] = None,
        vae_decode: Optional[WAN21VAEDecodeStage] = None,
        strategy: Optional[StepStrategy] = None,
        shift: float = 5.0,
        autocast_precision: str = "bf16",
        trajectory_precision: str = "fp16",
        logprob_precision: str = "fp32",
        max_sequence_length: int = 512,
    ) -> None:
        super().__init__()
        self.bundle = bundle
        self.text_embed = (
            text_embed
            if text_embed is not None
            else WAN21TextEmbedStage(bundle, max_sequence_length=int(max_sequence_length))
        )
        if diffusion is None:
            diffusion = WAN21DiffusionStage(
                model=bundle,
                step=WAN21DiffusionStep(),
                strategy=strategy if strategy is not None else DanceSDEStrategy(),
                autocast_precision=autocast_precision,
                trajectory_precision=trajectory_precision,
                logprob_precision=logprob_precision,
            )
        self.diffusion = diffusion
        self.vae_decode = vae_decode if vae_decode is not None else WAN21VAEDecodeStage(bundle)
        self.shift = shift

    @classmethod
    def latent_shape(cls, *, model_config: Any, sampling_spec: Any) -> tuple:
        """Per-sample 5D latent shape ``(C, T_lat, H_lat, W_lat)`` for
        driver-side noise pre-computation. Matches
        ``WAN21DiffusionStage._latent_shape``.

        WAN 2.1: ``AutoencoderKLWan`` is 16-channel, /8 spatial, /4
        temporal. ``T_lat = (num_frames - 1) // 4 + 1``.
        """
        height = int(sampling_spec.height)
        width = int(sampling_spec.width)
        num_frames = int(sampling_spec.num_frames)
        if (num_frames - 1) % 4 != 0:
            raise ValueError(
                f"WAN VAE temporal_downsample=4 requires "
                f"(num_frames - 1) % 4 == 0, got num_frames={num_frames}; "
                f"valid choices: 1, 5, 9, 13, 17, 21, ..."
            )
        latent_t = (num_frames - 1) // 4 + 1
        return (16, latent_t, height // 8, width // 8)

    @classmethod
    def from_config(
        cls,
        config: WAN21PipelineConfig,
        *,
        strategy: Optional[StepStrategy] = None,
    ) -> "WAN21Pipeline":
        """Build the full pipeline from a config.

        ``strategy`` is the SDE step strategy. Defaults to
        :class:`DanceSDEStrategy` (legacy WAN default in
        ``samplers/fsdp/wan_sampler.py``); callers running GRPO with
        Flow / CPS / DPM2 should pass an explicit strategy built from
        ``cfg.sampling.sde_strategy``.
        """
        bundle = WAN21Bundle.from_config(config)
        text_embed = WAN21TextEmbedStage(bundle, max_sequence_length=int(config.max_sequence_length))
        step = WAN21DiffusionStep()
        diffusion = WAN21DiffusionStage(
            model=bundle,
            step=step,
            strategy=strategy if strategy is not None else DanceSDEStrategy(),
            autocast_precision=config.autocast_precision,
            trajectory_precision=config.trajectory_precision,
            logprob_precision=config.logprob_precision,
        )
        vae_decode = WAN21VAEDecodeStage(bundle)
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
    ) -> WAN21Conditions:
        """Encode prompts (+ optional CFG negatives) into ``WAN21Conditions``.

        Builds only the text-conditioning slots (``text`` / ``negative_text``);
        the optional I2V ``image_latent`` / ``image_embed`` slots are left
        ``None`` and attached by :meth:`generate` when an input image is
        supplied (their encode path needs ``req`` / ``params``, outside this
        text-only contract).

        CFG negative encoding: WAN's training-time convention encodes
        an empty-string negative when none is provided (legacy
        ``models/wan21.py::encode_inputs`` does ``[""] * len(prompts)``
        — and so does diffusers' upstream WAN pipeline). Without this,
        falling back to ``torch.zeros_like(prompt_embeds)`` in
        ``WAN21DiffusionStep.predict_noise`` would silently use a
        different unconditional embedding than what the model was
        trained against, shifting the distribution and making the
        rollout / replay log-prob ratio drift away from 1.0 in GRPO.
        Encoding ``[""] * B`` explicitly here keeps both sides aligned.
        """
        if negatives is not None and len(negatives.texts) != len(texts.texts):
            raise ValueError(
                f"WAN21Pipeline.build_conditions: negative_text length "
                f"{len(negatives.texts)} != text length {len(texts.texts)}"
            )
        text_cond = self.text_embed.embed(texts)
        if negatives is None and float(guidance_scale) > 1.0:
            negatives = Texts(texts=[""] * len(texts.texts))
        negative_text_cond = self.text_embed.embed(negatives) if negatives is not None else None
        return WAN21Conditions(text=text_cond, negative_text=negative_text_cond)

    def generate(self, sample: Sample) -> Sample:
        """Run WAN 2.1 T2V (or I2V) end-to-end, filling the frontier (pre-forked) gen Part.

        Requires σ to be pinned onto the gen part's ``DiffusionSamplingParams.sigmas``
        by the hosting engine before the call; see the σ ownership note in
        ``unirl.models.types.pipeline``.
        """
        frontier = sample.parts[-1]
        params = frontier.sampling_params
        if not isinstance(params, DiffusionSamplingParams):
            raise TypeError(
                f"WAN21Pipeline.generate: frontier gen Part must carry DiffusionSamplingParams, "
                f"got {type(params).__name__ if params is not None else 'None'}"
            )
        if params.sigmas is None:
            raise ValueError(
                "WAN21Pipeline.generate: gen part sampling_params.sigmas is None. The hosting "
                "engine must pin σ before invoking pipeline.generate; see the σ ownership note "
                "in unirl.models.types.pipeline."
            )

        conditioning = sample.conditioning()
        texts = conditioning[0] if conditioning else None
        if not isinstance(texts, Texts):
            raise TypeError(
                f"WAN21Pipeline.generate: expected a Texts prompt from sample.conditioning()[0], "
                f"got {type(texts).__name__ if texts is not None else 'None'}"
            )
        images_prim = next((c for c in conditioning[1:] if isinstance(c, Images)), None)

        wan_conds = self.build_conditions(texts, guidance_scale=float(params.guidance_scale))

        if images_prim is not None:
            if len(images_prim) != len(texts.texts):
                raise ValueError(
                    f"WAN21Pipeline.generate: image count {len(images_prim)} != text count {len(texts.texts)}"
                )
            image_latent = WAN21ImageLatentEncodeStage(
                self.bundle,
                num_frames=int(params.num_frames),
                height=int(params.height),
                width=int(params.width),
            ).encode(images_prim)
            image_embed = (
                WAN21CLIPVisionEncodeStage(self.bundle).encode(images_prim) if self.bundle.uses_clip_vision else None
            )
            wan_conds = dataclasses.replace(
                wan_conds,
                image_latent=image_latent,
                image_embed=image_embed,
            )

        schedule = params.sigmas.to(self.bundle.device)

        initial_latents = NoiseRecipe.from_sample(sample).resolve()

        latent_seg = self.diffusion.diffuse(
            wan_conds, schedule=schedule, params=params, initial_latents=initial_latents
        )
        videos = self.vae_decode.decode(latent_seg)

        filled = frontier.fill(segment=latent_seg, primitives={"video": videos}, conditions=wan_conds.to_dict())
        return Sample(parts=[*sample.parts[:-1], filled], reward_compute_s=sample.reward_compute_s)


__all__ = ["WAN21Pipeline"]
