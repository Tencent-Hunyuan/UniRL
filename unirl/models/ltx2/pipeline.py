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
from unirl.types.primitives import Audio, Audios, Images, Texts
from unirl.types.sample import Sample

from .bundle import LTX2Bundle
from .conditions import LTX2Conditions
from .config import (
    LTX2_LATENT_CHANNELS,
    LTX2_SPATIAL_COMPRESSION,
    LTX2_TEMPORAL_COMPRESSION,
    LTX2PipelineConfig,
)
from .diffusion import LTX2DiffusionStage, audio_latent_shape
from .schedule import build_ltx2_schedule_policy
from .text_embed import LTX2TextEmbedStage
from .vae import LTX2AudioDecodeStage, LTX2VAEDecodeStage, LTX2VAEEncodeStage

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
        audio_decode: Optional[LTX2AudioDecodeStage] = None,
    ) -> None:
        self.bundle = bundle
        self.text_embed = text_embed
        self.diffusion = diffusion
        self.vae_decode = vae_decode
        self.vae_encode = vae_encode
        self.audio_decode = audio_decode
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
        # Audio decode is only meaningful for LTX-2.3 T2AV (bundle has audio_vae
        # + vocoder). For T2V it stays None and the audio path is never taken.
        audio_decode = LTX2AudioDecodeStage(bundle) if bundle.has_audio else None

        return cls(
            bundle=bundle,
            text_embed=text_embed,
            diffusion=diffusion,
            vae_decode=vae_decode,
            vae_encode=vae_encode,
            config=config,
            audio_decode=audio_decode,
        )

    def build_schedule_policy(self):
        """Build the LTX-2 schedule policy (constant-μ exponential shift).

        The hosting engine (``TrainsideRolloutEngine``) calls this at startup
        to pin the gen Part's ``DiffusionSamplingParams.sigmas`` before ``generate``.
        LTX-2 uses dynamic-shifting
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

    def build_conditions(
        self,
        texts: Texts,
        *,
        negatives: Optional[Texts] = None,
        guidance_scale: float = 1.0,
    ) -> LTX2Conditions:
        """Encode prompts and optional CFG negatives into ``LTX2Conditions``.

        CFG empty-negative: LTX-2's diffusers pipeline defaults negative_prompt to
        ``""`` when guidance is on, so the model sees its trained unconditional
        embedding. Without this, predict_noise would skip the CFG branch entirely
        (negative_text is None) and guidance_scale would be a silent no-op.
        The Sample path intentionally carries no negative prompt; direct callers
        such as ReFL can supply one through this public builder.
        """
        if negatives is not None and len(negatives.texts) != len(texts.texts):
            raise ValueError(
                f"LTX2Pipeline.build_conditions: negative_text length "
                f"{len(negatives.texts)} != text length {len(texts.texts)}"
            )
        if negatives is None and float(guidance_scale) > 1.0:
            negatives = Texts(texts=[""] * len(texts.texts))
        embed_result = self.text_embed.encode(texts, negative_texts=negatives)
        return LTX2Conditions.from_dict(embed_result)

    def generate(self, sample: Sample) -> Sample:
        """Run LTX-2 T2V end-to-end, filling the frontier (pre-forked) gen Part.

        Requires σ to be pinned onto the gen part's ``DiffusionSamplingParams.sigmas``
        by the hosting engine (a fallback ``get_sigma_schedule`` is computed if not,
        for parity with the prior req path); see the σ ownership note in
        ``unirl.models.types.pipeline``.
        """
        frontier = sample.parts[-1]
        params = frontier.sampling_params
        if params is None:
            raise ValueError("LTX2Pipeline.generate: DiffusionSamplingParams required.")

        conditioning = sample.conditioning()
        texts = conditioning[0] if conditioning else None
        if not isinstance(texts, Texts):
            raise TypeError(
                f"LTX2Pipeline.generate: expected a Texts prompt from sample.conditioning()[0], "
                f"got {type(texts).__name__ if texts is not None else 'None'}"
            )

        # I2V is NOT wired end-to-end yet: the encode step sets
        # conditions.image_latent, but LTX2DiffusionStep.predict_noise never
        # consumes it, so the image condition would be silently dropped
        # (I2V degrades to T2V with no error). Fail loudly until the
        # transformer image-conditioning path is implemented.
        if any(isinstance(c, Images) for c in conditioning[1:]):
            raise NotImplementedError(
                "LTX2Pipeline: I2V (a chained image input) is not supported yet — "
                "the diffusion stage does not consume conditions.image_latent. "
                "Only T2V is wired. Drop the image input, or implement the "
                "image-conditioning path in LTX2DiffusionStep.predict_noise."
            )

        # 1-2. Text embedding → conditions (shared re-encode path).
        conditions = self.build_conditions(texts, guidance_scale=float(params.guidance_scale))

        # 3. Sigma schedule: engine-pinned σ on the gen part wins; else fall back to
        # the configured schedule (parity with the prior pinned-or-compute path).
        sigmas = get_sigma_schedule(
            num_steps=int(params.num_inference_steps),
            shift=self.config.shift,
            device=self.bundle.device,
        )
        if params.sigmas is not None:
            sigmas = params.sigmas.to(self.bundle.device)

        # 4. Initial latents — driver-authoritative x_T via the model-aware
        # recipe (NoiseRecipe). The driver ships only a lightweight recipe
        # (noise_group_ids + init_noise_latent_shape, the 5D shape from this
        # pipeline's ``latent_shape``); we regenerate the byte-identical UNPACKED
        # 5D noise here, then pack into the transformer's ``(B, seq, C)`` token
        # layout. Pure x_T noise is NOT normalized — diffusers ``prepare_latents``
        # only normalizes PROVIDED img2img latents; the randn path packs raw
        # N(0,1) noise (flow-matching x_T). Only the FINAL latents are
        # denormalized before VAE decode (step 6). ``resolve()`` returns None only
        # under DISABLE_DRIVER_XT — then the recipe shape is None too and we cannot
        # draw video noise without a shape, so that path is unsupported here.
        video_recipe = NoiseRecipe.from_sample(sample)
        latents_5d = video_recipe.resolve(device=self.bundle.device)
        if latents_5d is None:
            raise ValueError(
                "LTX2Pipeline.generate: no initial latents. The driver x_T recipe "
                "(noise_group_ids + init_noise_latent_shape) is required; "
                "DISABLE_DRIVER_XT is not supported for LTX2 (video noise needs a "
                "driver-resolved 5D latent shape)."
            )
        patch_size, patch_size_t = self._patch_sizes()
        initial_latents = self._pack_latents(latents_5d.to(self.bundle.device), patch_size, patch_size_t)

        # Audio x_T: an ``::audio``-salted sibling of the SAME video recipe
        # (driver-authoritative, independent but reproducible) instead of a bare
        # randn inside the stage.
        initial_audio_latents = video_recipe.resolve(
            device=self.bundle.device, salt="audio", latent_shape=audio_latent_shape(params)
        )

        # 5. Diffusion loop
        sde_indices = list(params.sde_indices) if params.sde_indices is not None else None
        segment = self.diffusion.generate(
            conditions,
            params=params,
            sigmas=sigmas,
            initial_latents=initial_latents,
            initial_audio_latents=initial_audio_latents,
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

        # 6b. LTX-2.3 T2AV: decode the jointly-generated audio. The audio latent
        # trajectory rides on ``segment.aux_latents`` (same sparse indices as
        # the video latents), so the clean final-step audio is at step T. Decode
        # it to a waveform and carry it as a parallel ``Audios`` on the Part so
        # the reward service can feed audio scorers (CLAP / ImageBind) alongside
        # the video. T2V (no audio_decode) skips this entirely.
        decoded_audio = None
        audio_sample_rate = None
        if self.audio_decode is not None and segment.aux_latents is not None:
            from .diffusion import _LTX2_FRAME_RATE, _audio_num_frames

            audio_t = _audio_num_frames(int(params.num_frames), _LTX2_FRAME_RATE)
            final_audio = segment.aux_latents_at(int(params.num_inference_steps))
            waveforms = self.audio_decode.decode(final_audio, audio_latent_length=audio_t)
            # vocoder output: (B, C, L) or (B, L); package one Audio per sample.
            wf = waveforms.detach().float().cpu()
            # Store one mono ``[L]`` waveform per sample so ``Audios`` packs
            # cleanly along its varlen L axis and ``to_list`` recovers ``[L]``.
            audio_list = []
            for i in range(int(wf.shape[0])):
                w = wf[i]
                if w.ndim == 2:
                    w = w.mean(dim=0) if w.shape[0] <= w.shape[1] else w.mean(dim=1)
                audio_list.append(Audio(waveform=w.reshape(-1)))
            decoded_audio = Audios.from_list(audio_list)
            audio_sample_rate = int(self.bundle.vocoder.config.output_sampling_rate)

        # 7. Video and audio are one joint generation and therefore occupy one
        # frontier Part. Keep their row alignment in the modality-keyed map and
        # store the audio rate as shared modality metadata.
        primitives = {"video": decoded}
        primitive_metadata = {}
        if decoded_audio is not None:
            primitives["audio"] = decoded_audio
            primitive_metadata["audio"] = {"sample_rate": audio_sample_rate}
        filled = frontier.fill(
            segment=segment,
            primitives=primitives,
            primitive_metadata=primitive_metadata,
            conditions=conditions.to_dict(),
        )
        return sample.replace_frontier(filled)


__all__ = ["LTX2Pipeline"]
