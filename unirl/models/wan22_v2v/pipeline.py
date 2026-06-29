"""WAN22V2VPipeline — RolloutReq → RolloutResp for WAN 2.2 video-to-video."""

from __future__ import annotations

import dataclasses
from typing import Any, Optional

import torch

from unirl.models.types.pipeline import Pipeline
from unirl.models.wan21.conditions import WAN21Conditions
from unirl.models.wan21.text_embed import WAN21TextEmbedStage
from unirl.models.wan21.vae import WAN21VAEDecodeStage
from unirl.models.wan22.bundle import WAN22Bundle
from unirl.models.wan22.diffusion import WAN22DiffusionStage, WAN22DiffusionStep
from unirl.sde.kernels import DanceSDEStrategy, StepStrategy
from unirl.sde.noise import generate_latents
from unirl.types.noise_recipe import NoiseRecipe
from unirl.types.primitives import Texts, Videos
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp, RolloutTrack
from unirl.types.sampling import DiffusionSamplingParams

from .config import DEFAULT_V2V_STRENGTH
from .video_encode import WAN22VideoLatentEncodeStage


class WAN22V2VPipeline(Pipeline):
    """WAN 2.2 video-to-video generate pipeline."""

    def __init__(
        self,
        *,
        bundle: WAN22Bundle,
        text_embed: Optional[WAN21TextEmbedStage] = None,
        diffusion: Optional[WAN22DiffusionStage] = None,
        vae_decode: Optional[WAN21VAEDecodeStage] = None,
        strategy: Optional[StepStrategy] = None,
        shift: float = 5.0,
        strength: float = DEFAULT_V2V_STRENGTH,
        autocast_precision: str = "bf16",
        trajectory_precision: str = "fp16",
        logprob_precision: str = "fp32",
        max_sequence_length: int = 512,
    ) -> None:
        super().__init__()
        self.bundle = bundle
        self.shift = float(shift)
        self.strength = float(strength)
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

    @staticmethod
    def _sde_indices_in_trimmed_frame(sde_indices: Any, *, t_full: int, t_eff: int) -> list:
        """Remap SDE step indices from the full-schedule frame to the trimmed V2V frame.

        ``params.sde_indices`` is resolved by the trainer over the *full*
        ``num_inference_steps`` (``AllSDEScheduler(num_timesteps=num_inference_steps)``),
        but V2V only denoises the trimmed tail of ``t_eff`` steps. So a
        ``timestep_fraction`` window must be re-expressed in the trimmed frame,
        otherwise it lands on the wrong steps (the old code reinterpreted
        full-frame indices as trimmed-frame ones, which only happened to work when
        the fraction started at 0). Map each index by its fractional position
        ``round(i * t_eff / t_full)`` so "first 20% of denoising" stays "first 20%
        of the *denoised* tail" for any fraction. Empty in -> empty out (the
        deterministic forward-process path).
        """
        if not sde_indices or int(t_full) <= 0:
            return []
        remapped = {min(int(t_eff) - 1, max(0, round(int(i) * int(t_eff) / int(t_full)))) for i in sde_indices}
        return sorted(remapped)

    @classmethod
    def latent_shape(cls, *, model_config: Any, sampling_spec: Any) -> tuple:
        height = int(sampling_spec.height)
        width = int(sampling_spec.width)
        num_frames = int(sampling_spec.num_frames)
        if (num_frames - 1) % 4 != 0:
            raise ValueError(
                f"WAN VAE temporal_downsample=4 requires (num_frames - 1) % 4 == 0, "
                f"got num_frames={num_frames}; valid choices: 1, 5, 9, 13, 17, 21, ..."
            )
        latent_t = (num_frames - 1) // 4 + 1
        return (16, latent_t, height // 8, width // 8)

    @classmethod
    def from_config(
        cls,
        config: Any,
        *,
        strategy: Optional[StepStrategy] = None,
    ) -> "WAN22V2VPipeline":
        bundle = WAN22Bundle.from_config(config)
        return cls(
            bundle=bundle,
            strategy=strategy if strategy is not None else DanceSDEStrategy(),
            shift=float(config.shift),
            strength=float(getattr(config, "strength", DEFAULT_V2V_STRENGTH)),
            autocast_precision=config.autocast_precision,
            trajectory_precision=config.trajectory_precision,
            logprob_precision=config.logprob_precision,
            max_sequence_length=int(config.max_sequence_length),
        )

    def generate(self, req: RolloutReq) -> RolloutResp:
        texts = req.primitives.get("text")
        if not isinstance(texts, Texts):
            raise TypeError(
                f"WAN22V2VPipeline.generate: req.primitives['text'] must be Texts, "
                f"got {type(texts).__name__ if texts is not None else 'None'}"
            )
        videos = req.primitives.get("video")
        if not isinstance(videos, Videos):
            raise TypeError(
                f"WAN22V2VPipeline.generate: req.primitives['video'] must be Videos, "
                f"got {type(videos).__name__ if videos is not None else 'None'}."
            )
        if len(videos) != len(texts.texts):
            raise ValueError(f"WAN22V2VPipeline.generate: video count {len(videos)} != text count {len(texts.texts)}")

        negatives_raw = req.primitives.get("negative_text")
        negatives = negatives_raw if isinstance(negatives_raw, Texts) else None
        if negatives is not None and len(negatives.texts) != len(texts.texts):
            raise ValueError(
                f"WAN22V2VPipeline.generate: negative_text length {len(negatives.texts)} "
                f"!= text length {len(texts.texts)}"
            )

        params: DiffusionSamplingParams = req.sampling_params.get("diffusion")
        if params is None:
            raise ValueError("WAN22V2VPipeline.generate: req.sampling_params must contain 'diffusion'.")
        device = self.bundle.device

        text_cond = self.text_embed.embed(texts)
        primary_g = float(params.guidance_scale)
        low_g = float(params.guidance_scale_2) if params.guidance_scale_2 is not None else primary_g
        if negatives is None and max(primary_g, low_g) > 1.0:
            negatives = Texts(texts=[""] * len(texts.texts))
        negative_text_cond = self.text_embed.embed(negatives) if negatives is not None else None
        wan_conds = WAN21Conditions(text=text_cond, negative_text=negative_text_cond)

        video_latent_cond = WAN22VideoLatentEncodeStage(
            self.bundle,
            num_frames=int(params.num_frames),
            height=int(params.height),
            width=int(params.width),
        ).encode(videos)
        video_latents = video_latent_cond.latents.to(device=device, dtype=torch.float32)
        batch_size = int(video_latents.shape[0])

        if req.sigmas is None:
            raise ValueError(
                "WAN22V2VPipeline.generate: req.sigmas is None. Engine adapter must call "
                "unirl.sde.runtime.ensure_req_sigmas before pipeline.generate."
            )
        full_schedule = req.sigmas.to(device)
        t_full = int(full_schedule.shape[0]) - 1
        strength = float(params.strength) if params.strength is not None else self.strength
        if not 0.0 < strength <= 1.0:
            raise ValueError(f"WAN22V2VPipeline.generate: strength must be in (0, 1], got {strength}")
        t_eff = max(1, min(t_full, int(round(t_full * strength))))
        t_start = t_full - t_eff
        trimmed_schedule = full_schedule[t_start:].contiguous()
        sigma_start = trimmed_schedule[0].to(torch.float32)

        noise_recipe = NoiseRecipe.from_rollout_req(req)
        if noise_recipe.initial_latents is not None:
            raise ValueError(
                "WAN22V2VPipeline.generate: V2V video primitive cannot be combined with "
                "request_conditions['initial_latents']."
            )
        noise = noise_recipe.for_batch(batch_size, latent_shape=tuple(video_latents.shape[1:])).resolve(
            device=device,
            dtype=torch.float32,
        )
        if noise is None:
            noise = generate_latents(
                batch_size=batch_size,
                latent_shape=tuple(video_latents.shape[1:]),
                device=device,
                dtype=torch.float32,
                init_same_noise=bool(params.init_same_noise),
                samples_per_prompt=int(params.samples_per_prompt),
                noise_group_ids=params.noise_group_ids,
                base_seed=int(params.seed),
            )
        if int(noise.shape[0]) != batch_size or tuple(noise.shape[1:]) != tuple(video_latents.shape[1:]):
            raise ValueError(
                f"WAN22V2VPipeline.generate: noise shape {tuple(noise.shape)} incompatible with "
                f"video latents {tuple(video_latents.shape)}."
            )

        x_start = (1.0 - sigma_start) * video_latents + sigma_start * noise
        sde_indices = self._sde_indices_in_trimmed_frame(params.sde_indices, t_full=t_full, t_eff=t_eff)
        v2v_params = dataclasses.replace(params, num_inference_steps=t_eff, sde_indices=sde_indices)

        latent_seg = self.diffusion.diffuse(
            wan_conds,
            schedule=trimmed_schedule,
            params=v2v_params,
            initial_latents=x_start,
        )
        decoded = self.vae_decode.decode(latent_seg)

        return RolloutResp(
            tracks={
                "video": RolloutTrack(
                    sample_ids=list(req.sample_ids),
                    parent_ids=list(req.group_ids),
                    conditions=wan_conds.to_dict(),
                    segment=latent_seg,
                    decoded=decoded,
                ),
            }
        )


__all__ = ["WAN22V2VPipeline"]
