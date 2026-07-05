"""QwenImageEditPlusPipeline — RolloutReq → RolloutResp for Edit-Plus.

Text+image → image editing flow::

    Texts ──text_embed──▶ ┐
                         ├─▶ QwenImageEditPlusConditions ──diffuse──▶ LatentSegment
    Images ──vae_encode──▶ ┘                                              │
                                                                         ▼
                                                                     vae_decode
                                                                         │
                                                                         ▼
                                                                       Images

The text-embed and VAE-decode stages are reused from
:mod:`unirl.models.qwen_image` (V1 does standard text encoding — the
low-res 384² condition-image path into the Qwen2.5-VL text encoder is
deferred to V2). The VAE-encode stage and the diffusion step/stage are
Edit-Plus-specific.

σ schedule contract: identical to :class:`QwenImagePipeline` — the hosting
engine pins ``req.sigmas`` before calling ``generate(req)``. The schedule's
``image_seq_len`` is derived from the **noise** latent shape only
(boundary condition #3): the source-image concat happens inside
``predict_noise`` after the schedule is fixed.
"""

from __future__ import annotations

from typing import Any, Optional

from unirl.models.qwen_image.text_embed import QwenImageTextEmbedStage
from unirl.models.qwen_image.vae import QwenImageVAEDecodeStage
from unirl.models.types.pipeline import Pipeline
from unirl.sde.kernels import FlowSDEStrategy, StepStrategy
from unirl.types.noise_recipe import NoiseRecipe
from unirl.types.primitives import Images, Texts
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp, RolloutTrack
from unirl.types.sampling import DiffusionSamplingParams

from .bundle import QwenImageEditPlusBundle
from .conditions import QwenImageEditPlusConditions
from .config import QwenImageEditPlusPipelineConfig
from .diffusion import (
    QwenImageEditPlusDiffusionStage,
    QwenImageEditPlusDiffusionStep,
)
from .vae import QwenImageEditPlusVAEEncodeStage


class QwenImageEditPlusPipeline(Pipeline):
    """Qwen-Image-Edit-Plus generate pipeline.

    Reads from ``RolloutReq``:

    - ``primitives["text"]: Texts`` — required edit instructions.
    - ``primitives["image"]: Images`` — **required** source images (Edit-Plus
      is edit-only; raises ``TypeError`` if absent — fail-fast, constraint #27).
    - ``primitives["negative_text"]: Texts`` — optional CFG negatives.
    - ``sigmas: Tensor[T+1]`` — pinned by the engine adapter (required).

    Writes to ``RolloutResp``:

    - ``conditions["text"]``; plus ``conditions["negative_text"]`` when
      negatives supplied and ``conditions["image_latent"]`` always.
    - ``tracks["image"].segment: LatentSegment``.
    - ``tracks["image"].decoded: Images``.
    """

    def __init__(
        self,
        *,
        bundle: QwenImageEditPlusBundle,
        text_embed: Optional[QwenImageTextEmbedStage] = None,
        diffusion: Optional[QwenImageEditPlusDiffusionStage] = None,
        vae_encode: Optional[QwenImageEditPlusVAEEncodeStage] = None,
        vae_decode: Optional[QwenImageVAEDecodeStage] = None,
        strategy: Optional[StepStrategy] = None,
        shift: float = 3.0,
        autocast_precision: str = "bf16",
        trajectory_precision: str = "fp16",
        logprob_precision: str = "fp32",
        max_sequence_length: int = 512,
    ) -> None:
        super().__init__()
        self.bundle = bundle
        if text_embed is None and bundle.text_encoder is not None:
            text_embed = QwenImageTextEmbedStage(bundle, max_sequence_length=max_sequence_length)
        self.text_embed = text_embed
        if diffusion is None:
            diffusion = QwenImageEditPlusDiffusionStage(
                model=bundle,
                step=QwenImageEditPlusDiffusionStep(),
                strategy=strategy if strategy is not None else FlowSDEStrategy(),
                autocast_precision=autocast_precision,
                trajectory_precision=trajectory_precision,
                logprob_precision=logprob_precision,
            )
        self.diffusion = diffusion
        self.vae_encode = vae_encode if vae_encode is not None else QwenImageEditPlusVAEEncodeStage(bundle)
        self.vae_decode = vae_decode if vae_decode is not None else QwenImageVAEDecodeStage(bundle)
        self.shift = shift

    def build_schedule_policy(self):
        """Build the FlowMatchSchedulePolicy — identical to base Qwen-Image.

        The shift is derived from the noise latent's ``image_seq_len``
        (boundary condition #3); the source-image concat happens inside
        ``predict_noise`` and never enters the schedule. Reuses the base
        pipeline's canonical dynamic overrides.
        """
        from unirl.models.qwen_image.config import _qwen_image_dynamic_overrides
        from unirl.sde.runtime import FlowMatchSchedulePolicy

        return FlowMatchSchedulePolicy.from_pretrained(
            getattr(self.bundle, "pretrained_path", None),
            shift=float(self.shift),
            require_dynamic=True,
            dynamic_overrides=_qwen_image_dynamic_overrides(),
        )

    @classmethod
    def latent_shape(cls, *, model_config: Any, sampling_spec: Any) -> tuple:
        """Per-sample latent shape ``(C, H_lat, W_lat)`` for driver-side
        noise pre-computation. Identical to base Qwen-Image: the noise
        latent is 16-channel, 8× VAE downsample + 2× patchify rounding.
        """
        height = int(sampling_spec.height)
        width = int(sampling_spec.width)
        vae_scale_factor = 8
        latent_h = 2 * (height // (vae_scale_factor * 2))
        latent_w = 2 * (width // (vae_scale_factor * 2))
        return (16, latent_h, latent_w)

    @classmethod
    def from_config(
        cls,
        config: QwenImageEditPlusPipelineConfig,
        *,
        strategy: Optional[StepStrategy] = None,
    ) -> "QwenImageEditPlusPipeline":
        """Build the full Edit-Plus pipeline from a config."""
        bundle = QwenImageEditPlusBundle.from_config(config)
        text_embed = (
            QwenImageTextEmbedStage(bundle, max_sequence_length=config.max_sequence_length)
            if bundle.text_encoder is not None
            else None
        )
        step = QwenImageEditPlusDiffusionStep()
        diffusion = QwenImageEditPlusDiffusionStage(
            model=bundle,
            step=step,
            strategy=strategy if strategy is not None else FlowSDEStrategy(),
            autocast_precision=config.autocast_precision,
            trajectory_precision=config.trajectory_precision,
            logprob_precision=config.logprob_precision,
        )
        vae_encode = QwenImageEditPlusVAEEncodeStage(bundle)
        vae_decode = QwenImageVAEDecodeStage(bundle)
        return cls(
            bundle=bundle,
            text_embed=text_embed,
            diffusion=diffusion,
            vae_encode=vae_encode,
            vae_decode=vae_decode,
            shift=float(config.shift),
        )

    def generate(self, req: RolloutReq) -> RolloutResp:
        """Run Edit-Plus text+image→image end-to-end. Requires ``req.sigmas``
        to be pinned by the hosting engine adapter."""
        if req.sigmas is None:
            raise ValueError(
                "QwenImageEditPlusPipeline.generate: req.sigmas is None. The hosting "
                "engine (Trainside / SGLang / VLLMOmni) must call "
                "unirl.sde.runtime.ensure_req_sigmas(req, policy) before "
                "invoking pipeline.generate; see the σ ownership note in "
                "unirl.models.types.pipeline."
            )
        texts = req.primitives.get("text")
        if not isinstance(texts, Texts):
            raise TypeError(
                f"QwenImageEditPlusPipeline.generate: req.primitives['text'] must be Texts, "
                f"got {type(texts).__name__ if texts is not None else 'None'}"
            )
        images = req.primitives.get("image")
        if not isinstance(images, Images):
            raise TypeError(
                f"QwenImageEditPlusPipeline.generate: req.primitives['image'] must be Images "
                f"(Edit-Plus is edit-only), got "
                f"{type(images).__name__ if images is not None else 'None'}"
            )
        if images.pixels is None or int(images.pixels.shape[0]) != len(texts.texts):
            raise ValueError(
                f"QwenImageEditPlusPipeline.generate: req.primitives['image'] batch "
                f"{None if images.pixels is None else int(images.pixels.shape[0])} != "
                f"text batch {len(texts.texts)}"
            )
        negatives_raw = req.primitives.get("negative_text")
        negatives = negatives_raw if isinstance(negatives_raw, Texts) else None
        if negatives is not None and len(negatives.texts) != len(texts.texts):
            raise ValueError(
                f"QwenImageEditPlusPipeline.generate: negative_text length "
                f"{len(negatives.texts)} != text length {len(texts.texts)}"
            )

        params: DiffusionSamplingParams = req.sampling_params.get("diffusion")

        if self.text_embed is None:
            raise RuntimeError(
                "QwenImageEditPlusPipeline.generate: no text_embed stage "
                "(load_text_encoder=False). The trainer-side pipeline cannot "
                "encode prompts in this configuration — separate-engine "
                "recipes encode in the rollout engine; trainside rollout "
                "requires load_text_encoder=True."
            )
        text_cond = self.text_embed.embed(texts)
        # CFG empty negative: single-space " " (mirrors base Qwen-Image —
        # the 34-token chat-template prefix strip makes "" unsafe).
        if negatives is None and float(params.guidance_scale) > 1.0:
            negatives = Texts(texts=[" "] * len(texts.texts))
        negative_text_cond = self.text_embed.embed(negatives) if negatives is not None else None

        image_latent_cond = self.vae_encode.encode(images, height=int(params.height), width=int(params.width))

        edit_conds = QwenImageEditPlusConditions(
            text=text_cond,
            negative_text=negative_text_cond,
            image_latent=image_latent_cond,
        )

        schedule = req.sigmas.to(self.bundle.device)
        initial_latents = NoiseRecipe.from_rollout_req(req).resolve()

        latent_seg = self.diffusion.diffuse(
            edit_conds, schedule=schedule, params=params, initial_latents=initial_latents
        )
        decoded = self.vae_decode.decode(latent_seg)

        return RolloutResp(
            tracks={
                "image": RolloutTrack(
                    sample_ids=list(req.sample_ids),
                    parent_ids=list(req.group_ids),
                    conditions=edit_conds.to_dict(),
                    segment=latent_seg,
                    decoded=decoded,
                ),
            }
        )


__all__ = ["QwenImageEditPlusPipeline"]
