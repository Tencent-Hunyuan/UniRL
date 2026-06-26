"""LTX2 pipeline — T2V / I2V / T2AV dispatch.

Composes text embedding, diffusion, and VAE decode stages into a complete
rollout pipeline. Mode is determined by the request primitives:
- T2V: text only → video generation
- I2V: text + image → image-conditioned video generation
- T2AV: text → video + audio joint generation (LTX-2.3)
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import torch

from unirl.models.types.pipeline import Pipeline
from unirl.sde.kernels import StepStrategy
from unirl.sde.runtime import get_sigma_schedule
from unirl.types.noise_recipe import NoiseRecipe
from unirl.types.primitives import Images, Texts
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp, RolloutTrack

from .bundle import LTX2Bundle
from .conditions import LTX2Conditions
from .config import (
    LTX2_LATENT_CHANNELS,
    LTX2_SPATIAL_COMPRESSION,
    LTX2_TEMPORAL_COMPRESSION,
    LTX2PipelineConfig,
)
from .diffusion import LTX2DiffusionStage
from .schedule import build_ltx2_schedule_policy
from .text_embed import LTX2TextEmbedStage
from .vae import LTX2VAEDecodeStage, LTX2VAEEncodeStage

logger = logging.getLogger(__name__)

# LTX-2 3D-VAE geometry constants now live in config.py (shared with the
# diffusion stage). Local aliases kept for readability in this module.
_LTX2_SPATIAL_COMPRESSION = LTX2_SPATIAL_COMPRESSION
_LTX2_TEMPORAL_COMPRESSION = LTX2_TEMPORAL_COMPRESSION
_LTX2_LATENT_CHANNELS = LTX2_LATENT_CHANNELS


class LTX2Pipeline(Pipeline):
    """LTX-2/2.3 T2V / I2V / T2AV pipeline."""

    def __init__(
        self,
        *,
        bundle: LTX2Bundle,
        text_embed: LTX2TextEmbedStage,
        diffusion: LTX2DiffusionStage,
        vae_decode: LTX2VAEDecodeStage,
        vae_encode: Optional[LTX2VAEEncodeStage],
        config: LTX2PipelineConfig,
    ) -> None:
        self.bundle = bundle
        self.text_embed = text_embed
        self.diffusion = diffusion
        self.vae_decode = vae_decode
        self.vae_encode = vae_encode
        self.config = config
        # Exposed for the hosting engine (TrainsideRolloutEngine reads
        # ``pipeline.shift`` to build a FlowMatchSchedulePolicy at startup) —
        # same convention as SD3/flux2klein. ``generate`` itself reads
        # ``self.config.shift`` directly.
        self.shift = config.shift

    @classmethod
    def from_config(
        cls,
        config: LTX2PipelineConfig,
        *,
        strategy: Optional[StepStrategy] = None,
    ) -> "LTX2Pipeline":
        """Build pipeline from config + bundle."""
        bundle = LTX2Bundle.from_config(config)
        return cls.from_bundle(config=config, bundle=bundle, strategy=strategy)

    @classmethod
    def from_bundle(
        cls,
        *,
        config: LTX2PipelineConfig,
        bundle: LTX2Bundle,
        strategy: Optional[StepStrategy] = None,
    ) -> "LTX2Pipeline":
        """Build pipeline stages from an existing bundle."""
        from unirl.sde.kernels import FlowSDEStrategy

        if strategy is None:
            strategy = FlowSDEStrategy()

        text_embed = LTX2TextEmbedStage(bundle)
        diffusion = LTX2DiffusionStage(
            bundle,
            strategy=strategy,
            autocast_precision=config.autocast_precision,
            trajectory_precision=config.trajectory_precision,
            logprob_precision=config.logprob_precision,
            audio_joint_sde=config.audio_joint_sde,
        )
        vae_decode = LTX2VAEDecodeStage(bundle)
        vae_encode = LTX2VAEEncodeStage(bundle)

        return cls(
            bundle=bundle,
            text_embed=text_embed,
            diffusion=diffusion,
            vae_decode=vae_decode,
            vae_encode=vae_encode,
            config=config,
        )

    def build_schedule_policy(self):
        """Build the LTX-2 schedule policy (constant-μ exponential shift).

        The hosting engine (``TrainsideRolloutEngine``) calls this at startup
        to pin ``req.sigmas`` before ``generate``. LTX-2 uses dynamic-shifting
        with μ ≡ ``max_shift`` (2.05) — NOT the static ``shift=1.0`` the engine
        would otherwise fall back to (which under-resolves the trajectory and
        yields blurry frames). See ``schedule.py`` for the diffusers alignment.
        """
        return build_ltx2_schedule_policy(self.shift)

    @classmethod
    def latent_shape(cls, *, model_config: Any, sampling_spec: Any) -> Tuple[int, ...]:
        """Per-sample 5D latent shape ``(C, T_lat, H_lat, W_lat)`` for the
        driver-side x_T recipe (``LatentShapeProvider`` contract).

        LTX-2 3D-VAE: 32x spatial, 8x temporal, 128 latent channels. The
        temporal axis is causal, so ``T_lat = (num_frames - 1) // 8 + 1``.
        Noise is generated UNPACKED (5D) here; the pipeline packs it into the
        transformer's ``(B, seq, C)`` token layout in ``generate``. Mirrors
        diffusers' ``LTX2Pipeline.prepare_latents``.
        """
        height = int(sampling_spec.height)
        width = int(sampling_spec.width)
        num_frames = int(sampling_spec.num_frames)
        if (num_frames - 1) % _LTX2_TEMPORAL_COMPRESSION != 0:
            raise ValueError(
                f"LTX2 VAE temporal_compression={_LTX2_TEMPORAL_COMPRESSION} requires "
                f"(num_frames - 1) % {_LTX2_TEMPORAL_COMPRESSION} == 0, got num_frames={num_frames}; "
                f"valid choices: 1, 9, 17, 25, 33, ..."
            )
        latent_t = (num_frames - 1) // _LTX2_TEMPORAL_COMPRESSION + 1
        latent_h = height // _LTX2_SPATIAL_COMPRESSION
        latent_w = width // _LTX2_SPATIAL_COMPRESSION
        return (_LTX2_LATENT_CHANNELS, latent_t, latent_h, latent_w)

    def _patch_sizes(self) -> Tuple[int, int]:
        """``(patch_size, patch_size_t)`` read off the transformer config
        (defaults 1/1 — LTX-2 patchifies in the proj_in linear, not by reshape).
        """
        cfg = self.bundle.transformer.config
        return int(getattr(cfg, "patch_size", 1)), int(getattr(cfg, "patch_size_t", 1))

    @staticmethod
    def _pack_latents(latents: torch.Tensor, patch_size: int, patch_size_t: int) -> torch.Tensor:
        """5D ``(B, C, F, H, W)`` → packed ``(B, seq, C·p_t·p·p)``.

        Verbatim from diffusers ``LTX2Pipeline._pack_latents``.
        """
        batch_size, num_channels, num_frames, height, width = latents.shape
        post_f = num_frames // patch_size_t
        post_h = height // patch_size
        post_w = width // patch_size
        latents = latents.reshape(batch_size, -1, post_f, patch_size_t, post_h, patch_size, post_w, patch_size)
        latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
        return latents

    @staticmethod
    def _unpack_latents(
        latents: torch.Tensor, num_frames: int, height: int, width: int, patch_size: int, patch_size_t: int
    ) -> torch.Tensor:
        """Packed ``(B, seq, D)`` → 5D ``(B, C, F, H, W)`` — inverse of pack.

        Verbatim from diffusers ``LTX2Pipeline._unpack_latents``.
        """
        batch_size = latents.size(0)
        latents = latents.reshape(batch_size, num_frames, height, width, -1, patch_size_t, patch_size, patch_size)
        latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
        return latents

    def _denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Channel-wise denormalize a 5D latent by the VAE's ``latents_mean/std``
        and ``scaling_factor`` before decode (diffusers ``_denormalize_latents``).

        Inverse of the VAE's normalization. (Pure x_T noise is fed RAW to the
        transformer — see ``generate`` — so there is no forward ``_normalize``
        on the rollout path; only the produced latents are denormalized here.)
        """
        vae = self.bundle.vae
        mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        std = vae.latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
        return latents * std / float(vae.config.scaling_factor) + mean

    def generate(self, req: RolloutReq) -> RolloutResp:
        """Run T2V / I2V / T2AV based on request primitives."""
        texts = req.primitives.get("text")
        if not isinstance(texts, Texts):
            raise TypeError(
                f"LTX2Pipeline.generate: req.primitives['text'] must be Texts, "
                f"got {type(texts).__name__ if texts is not None else 'None'}"
            )

        images = req.primitives.get("image")
        # ``req.sampling_params`` is the per-stage dict keyed by stage name —
        # same accessor every sibling pipeline uses (sd3/wan21/…). (The earlier
        # ``get_diffusion_params`` helper only exists on the in-flight
        # ComposedSamplingParams branch, not on main, so it broke import here.)
        params = req.sampling_params.get("diffusion")
        if params is None:
            raise ValueError("LTX2Pipeline.generate: DiffusionSamplingParams required.")

        # Determine mode
        has_image = isinstance(images, Images)
        if has_image:
            # I2V is NOT wired end-to-end yet: the encode step below sets
            # conditions.image_latent, but LTX2DiffusionStep.predict_noise never
            # consumes it, so the image condition would be silently dropped
            # (I2V degrades to T2V with no error). Fail loudly until the
            # transformer image-conditioning path is implemented.
            raise NotImplementedError(
                "LTX2Pipeline: I2V (req.primitives['image']) is not supported yet — "
                "the diffusion stage does not consume conditions.image_latent. "
                "Only T2V is wired. Drop the image primitive, or implement the "
                "image-conditioning path in LTX2DiffusionStep.predict_noise."
            )

        # 1. Text embedding. CFG empty-negative: LTX-2's diffusers pipeline
        # defaults negative_prompt to "" when guidance is on, so the model sees
        # its trained unconditional embedding. Without this, predict_noise would
        # skip the CFG branch entirely (negative_text is None) and guidance_scale
        # would be a silent no-op.
        negative_texts = req.primitives.get("negative_text")
        neg = negative_texts if isinstance(negative_texts, Texts) else None
        if neg is None and float(params.guidance_scale) > 1.0:
            neg = Texts(texts=[""] * len(texts.texts))
        embed_result = self.text_embed.encode(texts, negative_texts=neg)

        # 2. Build conditions
        conditions = LTX2Conditions.from_dict(embed_result)

        # I2V: encode condition image
        if has_image and self.vae_encode is not None:
            image_latents = self.vae_encode.encode(images.pixels)
            conditions.image_latent = image_latents

        # 3. Sigma schedule
        num_steps = int(params.num_inference_steps)
        sigmas = get_sigma_schedule(
            num_steps=num_steps,
            shift=self.config.shift,
            device=self.bundle.device,
        )
        if req.sigmas is not None:
            sigmas = req.sigmas.to(self.bundle.device)

        # 4. Initial latents — driver-authoritative x_T via the model-aware
        # recipe (NoiseRecipe). The driver ships only a lightweight recipe
        # (init_noise_group_ids + init_noise_latent_shape, the 5D shape from
        # this pipeline's ``latent_shape``); we regenerate the byte-identical
        # UNPACKED 5D noise here, then pack into the transformer's
        # ``(B, seq, C)`` token layout. Pure x_T noise is NOT normalized —
        # diffusers ``prepare_latents`` only normalizes PROVIDED img2img latents;
        # the randn path packs raw N(0,1) noise (flow-matching x_T). Only the
        # FINAL latents are denormalized before VAE decode (step 6).
        # ``resolve()`` returns None only under DISABLE_DRIVER_XT — then the
        # recipe shape is None too and we cannot draw video noise without a
        # shape, so that path is unsupported here.
        latents_5d = NoiseRecipe.from_rollout_req(req).resolve(device=self.bundle.device)
        if latents_5d is None:
            raise ValueError(
                "LTX2Pipeline.generate: no initial latents. The driver x_T recipe "
                "(init_noise_group_ids + init_noise_latent_shape) is required; "
                "DISABLE_DRIVER_XT is not supported for LTX2 (video noise needs a "
                "driver-resolved 5D latent shape)."
            )
        patch_size, patch_size_t = self._patch_sizes()
        initial_latents = self._pack_latents(latents_5d.to(self.bundle.device), patch_size, patch_size_t)

        # 5. Diffusion loop
        sde_indices = list(params.sde_indices) if params.sde_indices is not None else None
        segment = self.diffusion.generate(
            conditions,
            params=params,
            sigmas=sigmas,
            initial_latents=initial_latents,
            sde_indices=sde_indices,
        )

        # 6. Unpack + denormalize → 5D latents → VAE decode → video frames.
        # The clean final latent is the last trajectory step (step T); the
        # segment stores it sparsely, retrieved via ``latents_at``.
        _, latent_t, latent_h, latent_w = self.latent_shape(model_config=self.config, sampling_spec=params)
        final_latents = segment.latents_at(int(params.num_inference_steps))
        unpacked = self._unpack_latents(final_latents, latent_t, latent_h, latent_w, patch_size, patch_size_t)
        unpacked = self._denormalize_latents(unpacked)
        decoded = self.vae_decode.decode(unpacked)  # → varlen-packed Videos

        # 7. Build response. ``parent_ids=req.group_ids`` makes sibling samples
        # of one prompt a GRPO group (RolloutTrack.group_ids is a derived
        # read-only property — NOT a constructor arg). ``decoded`` is the single
        # Videos primitive for this track (the reward service reads it directly),
        # not a modality-keyed dict. Track key ``"video"`` matches the WAN21
        # video convention.
        track = RolloutTrack(
            sample_ids=list(req.sample_ids),
            parent_ids=list(req.group_ids),
            conditions=conditions.to_dict(),
            segment=segment,
            decoded=decoded,
        )

        return RolloutResp(tracks={"video": track})


__all__ = ["LTX2Pipeline"]
