"""WAN21Pipeline — RolloutReq → RolloutResp end-to-end for WAN 2.1 T2V.

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

from typing import Any, Optional

from unirl.models.types.pipeline import Pipeline
from unirl.sde.kernels import DanceSDEStrategy, StepStrategy
from unirl.types.conditions import ImageEmbedCondition, ImageLatentCondition
from unirl.types.noise_recipe import NoiseRecipe
from unirl.types.primitives import Images, Texts
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp, RolloutTrack
from unirl.types.sampling import DiffusionSamplingParams, get_diffusion_params

from .bundle import WAN21Bundle
from .clip_vision_encode import WAN21CLIPVisionEncodeStage
from .conditions import WAN21Conditions
from .config import WAN21PipelineConfig
from .diffusion import WAN21DiffusionStage, WAN21DiffusionStep
from .image_encode import WAN21ImageLatentEncodeStage
from .text_embed import WAN21TextEmbedStage
from .vae import WAN21VAEDecodeStage


class WAN21Pipeline(Pipeline):
    """WAN 2.1 T2V generate pipeline.

    Reads from ``RolloutReq``:

    - ``primitives["text"]: Texts`` — required prompts.
    - ``primitives["negative_text"]: Texts`` — optional CFG negatives.
    - ``stage_params["diffusion"]: dict`` — kwargs for
      :class:`WAN21DiffusionParams`.

    Writes to ``RolloutResp``:

    - ``conditions["text"]: TextEmbedCondition``; plus
      ``conditions["negative_text"]: TextEmbedCondition`` when negative
      prompts were supplied.
    - ``tracks["video"].segment: LatentSegment``.
    - ``tracks["video"].decoded: Videos``.
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
        # Stages default to None and are built from the (trainer-injected)
        # bundle — mirrors SD3Pipeline so the v2 trainer can construct the
        # pipeline via ``remote_hydra(pipeline_cfg, bundle=self.bundle)`` without
        # reloading weights. ``from_config`` still passes pre-built stages.
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

    def generate(self, req: RolloutReq) -> RolloutResp:
        """Run WAN 2.1 T2V end-to-end."""
        texts = req.primitives.get("text")
        if not isinstance(texts, Texts):
            raise TypeError(
                f"WAN21Pipeline.generate: req.primitives['text'] must be Texts, "
                f"got {type(texts).__name__ if texts is not None else 'None'}"
            )
        negatives_raw = req.primitives.get("negative_text")
        negatives = negatives_raw if isinstance(negatives_raw, Texts) else None
        if negatives is not None and len(negatives.texts) != len(texts.texts):
            raise ValueError(
                f"WAN21Pipeline.generate: negative_text length {len(negatives.texts)} != text length {len(texts.texts)}"
            )

        params: DiffusionSamplingParams = get_diffusion_params(req.sampling_params)

        text_cond = self.text_embed.embed(texts)
        # CFG negative encoding: WAN's training-time convention encodes
        # an empty-string negative when none is provided (legacy
        # ``models/wan21.py::encode_inputs`` does ``[""] * len(prompts)``
        # — and so does diffusers' upstream WAN pipeline). Without this,
        # falling back to ``torch.zeros_like(prompt_embeds)`` in
        # ``WAN21DiffusionStep.predict_noise`` would silently use a
        # different unconditional embedding than what the model was
        # trained against, shifting the distribution and making the
        # rollout / replay log-prob ratio drift away from 1.0 in GRPO.
        # Encoding ``[""] * B`` explicitly here keeps both sides aligned.
        if negatives is None and float(params.guidance_scale) > 1.0:
            negatives = Texts(texts=[""] * len(texts.texts))
        negative_text_cond = self.text_embed.embed(negatives) if negatives is not None else None

        image_latent_cond: Optional[ImageLatentCondition] = None
        image_embed_cond: Optional[ImageEmbedCondition] = None
        images_prim = req.primitives.get("image")
        if images_prim is not None:
            if not isinstance(images_prim, Images):
                raise TypeError(
                    f"WAN21Pipeline.generate: req.primitives['image'] must be Images, got {type(images_prim).__name__}"
                )
            if int(images_prim.pixels.shape[0]) != len(texts.texts):
                raise ValueError(
                    f"WAN21Pipeline.generate: image count {images_prim.pixels.shape[0]} "
                    f"!= text count {len(texts.texts)}"
                )
            image_latent_cond = WAN21ImageLatentEncodeStage(
                self.bundle,
                num_frames=int(params.num_frames),
                height=int(params.height),
                width=int(params.width),
            ).encode(images_prim)
            # CLIP-vision branch fires only when the bundle loaded a
            # vision tower (transformer.config.image_dim > 0). T2V
            # bundles skip this entirely; WAN 2.2 mainstream checkpoints
            # also skip it (image_dim=0).
            if self.bundle.uses_clip_vision:
                image_embed_cond = WAN21CLIPVisionEncodeStage(self.bundle).encode(images_prim)

        wan_conds = WAN21Conditions(
            text=text_cond,
            negative_text=negative_text_cond,
            image_latent=image_latent_cond,
            image_embed=image_embed_cond,
        )

        if req.sigmas is None:
            raise ValueError(
                "WAN21Pipeline.generate: req.sigmas is None. Engine adapter "
                "must call unirl.sde.runtime.ensure_req_sigmas before "
                "pipeline.generate."
            )
        schedule = req.sigmas.to(self.bundle.device)

        # Driver-authoritative x_T via the model-aware recipe (NoiseRecipe); a
        # pre-shipped initial_latents tensor (img2img / i2v first-frame) still wins.
        initial_latents = NoiseRecipe.from_rollout_req(req).resolve()

        latent_seg = self.diffusion.diffuse(
            wan_conds, schedule=schedule, params=params, initial_latents=initial_latents
        )
        videos = self.vae_decode.decode(latent_seg)

        return RolloutResp(
            tracks={
                "video": RolloutTrack(
                    sample_ids=list(req.sample_ids),
                    parent_ids=list(req.group_ids),
                    conditions=wan_conds.to_dict(),
                    segment=latent_seg,
                    decoded=videos,
                ),
            }
        )


__all__ = ["WAN21Pipeline"]
