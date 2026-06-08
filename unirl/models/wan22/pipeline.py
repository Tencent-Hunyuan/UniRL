"""WAN22Pipeline — RolloutReq → RolloutResp end-to-end for WAN 2.2 T2V.

Implements the new four-tier flow::

    Texts ──text_embed (wan21)──▶ WAN21Conditions ──diffuse (wan22)──▶ LatentSegment ──vae_decode (wan21)──▶ Videos

Hydra constructs a pipeline via
``WAN22Pipeline.from_config(WAN22PipelineConfig)`` (see ``config.py``);
``from_config`` loads the :class:`WAN22Bundle` (dual transformer + WAN
2.1 VAE/text encoder) then constructs the four stages with the
precision policy from the config.

WAN 2.2 reuses WAN 2.1's text embedding and VAE stages verbatim (same
UMT5 with zero-padding, same 3D VAE with per-channel norm) — only the
diffusion stage swaps in for dual-transformer routing. We do **not**
inherit ``WAN21Pipeline``: the reuse is by composition (import the
sibling stages), matching the SD3 convention of one-package-per-model.
"""

from __future__ import annotations

from typing import Any, Optional

from unirl.models.types.pipeline import Pipeline
from unirl.models.wan21.clip_vision_encode import WAN21CLIPVisionEncodeStage
from unirl.models.wan21.conditions import WAN21Conditions
from unirl.models.wan21.image_encode import WAN21ImageLatentEncodeStage
from unirl.models.wan21.text_embed import WAN21TextEmbedStage
from unirl.models.wan21.vae import WAN21VAEDecodeStage
from unirl.sde.kernels import DanceSDEStrategy, StepStrategy
from unirl.types.conditions import ImageEmbedCondition, ImageLatentCondition
from unirl.types.noise_recipe import NoiseRecipe
from unirl.types.primitives import Images, Texts
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp, RolloutTrack
from unirl.types.sampling import DiffusionSamplingParams, get_diffusion_params

from .bundle import WAN22Bundle
from .config import WAN22PipelineConfig
from .diffusion import WAN22DiffusionStage, WAN22DiffusionStep


class WAN22Pipeline(Pipeline):
    """WAN 2.2 T2V generate pipeline.

    Reads from ``RolloutReq``:

    - ``primitives["text"]: Texts`` — required prompts.
    - ``primitives["negative_text"]: Texts`` — optional CFG negatives.
    - ``stage_params["diffusion"]: dict`` — kwargs for
      :class:`WAN22DiffusionParams` (extends WAN21 params with optional
      ``guidance_scale_2``).

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
        bundle: WAN22Bundle,
        text_embed: Optional[WAN21TextEmbedStage] = None,
        diffusion: Optional[WAN22DiffusionStage] = None,
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
        # reloading the dual transformer. ``from_config`` still passes pre-built stages.
        super().__init__()
        self.bundle = bundle
        self.text_embed = (
            text_embed
            if text_embed is not None
            else WAN21TextEmbedStage(bundle, max_sequence_length=int(max_sequence_length))
        )
        if diffusion is None:
            diffusion = WAN22DiffusionStage(
                model=bundle,
                step=WAN22DiffusionStep(),
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
        driver-side noise pre-computation. Same VAE family as WAN 2.1
        (``AutoencoderKLWan``: 16-channel, /8 spatial, /4 temporal); the
        dual-transformer routing in WAN 2.2 does not change latent
        geometry."""
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
        config: WAN22PipelineConfig,
        *,
        strategy: Optional[StepStrategy] = None,
    ) -> "WAN22Pipeline":
        """Build the full pipeline from a config.

        ``strategy`` is the SDE step strategy. Defaults to
        :class:`DanceSDEStrategy` (legacy WAN family default). Callers
        running other strategies (Flow / CPS / DPM2) should pass an
        explicit strategy built from ``cfg.sampling.sde_strategy``.
        """
        bundle = WAN22Bundle.from_config(config)

        # WAN 2.1's text embed stage expects a ``WAN21Bundle``-compatible
        # object — we satisfy that contract with ``WAN22Bundle`` (it
        # exposes the same ``text_encoder`` / ``tokenizer`` /
        # ``max_sequence_length`` / ``device`` fields). The stage uses
        # duck-typing so no isinstance check fires.
        text_embed = WAN21TextEmbedStage(bundle, max_sequence_length=int(config.max_sequence_length))
        step = WAN22DiffusionStep()
        diffusion = WAN22DiffusionStage(
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
        """Run WAN 2.2 T2V end-to-end."""
        texts = req.primitives.get("text")
        if not isinstance(texts, Texts):
            raise TypeError(
                f"WAN22Pipeline.generate: req.primitives['text'] must be Texts, "
                f"got {type(texts).__name__ if texts is not None else 'None'}"
            )
        negatives_raw = req.primitives.get("negative_text")
        negatives = negatives_raw if isinstance(negatives_raw, Texts) else None
        if negatives is not None and len(negatives.texts) != len(texts.texts):
            raise ValueError(
                f"WAN22Pipeline.generate: negative_text length {len(negatives.texts)} != text length {len(texts.texts)}"
            )

        params: DiffusionSamplingParams = get_diffusion_params(req.sampling_params)

        text_cond = self.text_embed.embed(texts)
        # CFG empty negative: same rationale as WAN21Pipeline (see that
        # method's comment) — WAN training encodes an empty-string
        # negative when none is supplied. WAN22 routes CFG by sigma /
        # ``guidance_scale_2`` so we trigger the empty-negative encoding
        # whenever either branch's effective guidance is > 1.
        primary_g = float(params.guidance_scale)
        low_g = float(params.guidance_scale_2) if params.guidance_scale_2 is not None else primary_g
        cfg_active = max(primary_g, low_g) > 1.0
        if negatives is None and cfg_active:
            negatives = Texts(texts=[""] * len(texts.texts))
        negative_text_cond = self.text_embed.embed(negatives) if negatives is not None else None

        image_latent_cond: Optional[ImageLatentCondition] = None
        image_embed_cond: Optional[ImageEmbedCondition] = None
        images_prim = req.primitives.get("image")
        if images_prim is not None:
            if not isinstance(images_prim, Images):
                raise TypeError(
                    f"WAN22Pipeline.generate: req.primitives['image'] must be Images, got {type(images_prim).__name__}"
                )
            if int(images_prim.pixels.shape[0]) != len(texts.texts):
                raise ValueError(
                    f"WAN22Pipeline.generate: image count {images_prim.pixels.shape[0]} "
                    f"!= text count {len(texts.texts)}"
                )
            image_latent_cond = WAN21ImageLatentEncodeStage(
                self.bundle,
                num_frames=int(params.num_frames),
                height=int(params.height),
                width=int(params.width),
            ).encode(images_prim)
            # CLIP-vision branch fires only on bundles that actually
            # loaded a vision tower. WAN 2.2's mainstream checkpoints
            # set ``image_dim == 0`` and skip this; left in place so a
            # future 2.2 variant with ``image_dim > 0`` (if it ever
            # ships) wires up automatically without a pipeline change.
            if getattr(self.bundle, "uses_clip_vision", False):
                image_embed_cond = WAN21CLIPVisionEncodeStage(self.bundle).encode(images_prim)

        wan_conds = WAN21Conditions(
            text=text_cond,
            negative_text=negative_text_cond,
            image_latent=image_latent_cond,
            image_embed=image_embed_cond,
        )

        if req.sigmas is None:
            raise ValueError(
                "WAN22Pipeline.generate: req.sigmas is None. Engine adapter "
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


__all__ = ["WAN22Pipeline"]
