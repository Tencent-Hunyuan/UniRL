"""Cosmos3 packed omnimodal stage and the SFT / text-to-image pipeline."""

from __future__ import annotations

import dataclasses
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch

from unirl.models.types.pipeline import Pipeline
from unirl.sde.kernels import FlowSDEStrategy, StepStrategy
from unirl.sde.runtime import FlowMatchSchedulePolicy
from unirl.types.conditions import TextTokenCondition
from unirl.types.noise_recipe import NoiseRecipe
from unirl.types.primitives import Images, Texts
from unirl.types.sample import Sample
from unirl.types.sampling import DiffusionSamplingParams
from unirl.types.segments.latent import LatentSegment

from .bundle import Cosmos3Bundle, _import_diffusers_classes
from .conditions import Cosmos3Conditions
from .diffusion import Cosmos3DiffusionStage, Cosmos3DiffusionStep
from .packing import (
    noise_action_latents,
    noise_vision_latents,
    pack_joint_sequence,
    pad_action_chunk,
    resolution_tier,
)


@dataclass
class Cosmos3JointPrediction:
    """One packed forward aligned to its vision/action velocity targets."""

    vision_pred: torch.Tensor
    vision_target: torch.Tensor
    action_pred: Optional[torch.Tensor]
    action_target: Optional[torch.Tensor]
    sigma: torch.Tensor


class Cosmos3JointStage:
    """Model boundary for one Cosmos3 text+vision(+action) packed forward."""

    def __init__(self, bundle: Cosmos3Bundle) -> None:
        self.bundle = bundle
        self.config = bundle.config
        *_, Cosmos3OmniPipeline = _import_diffusers_classes()
        self.pipe = Cosmos3OmniPipeline(
            transformer=bundle.transformer,
            text_tokenizer=bundle.text_tokenizer,
            vae=bundle.vae,
            scheduler=bundle.scheduler,
            sound_tokenizer=None,
            safety_checker=None,
            enable_safety_checker=False,
        )
        self._token_cache: OrderedDict[tuple[Any, ...], list[int]] = OrderedDict()
        self._token_cache_max = 256

    @property
    def device(self) -> torch.device:
        return self.bundle.device

    @torch.no_grad()
    def encode_video(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode uint8 ``[T,3,H,W]`` frames to fp32 ``[1,C,T',H',W']``."""
        pixels = frames.to(device=self.device, dtype=torch.float32).div(127.5).sub(1.0)
        pixels = pixels.permute(1, 0, 2, 3).unsqueeze(0)
        return self.pipe._encode_video(pixels)

    def tokenize_prompt(
        self,
        prompt: str,
        *,
        num_frames: int,
        height: int,
        width: int,
        fps: float,
        action_mode: Optional[str],
    ) -> list[int]:
        key = (prompt, num_frames, height, width, round(float(fps), 3), action_mode)
        ids = self._token_cache.get(key)
        if ids is None:
            ids, _ = self.pipe.tokenize_prompt(
                prompt,
                None,
                num_frames=num_frames,
                height=height,
                width=width,
                fps=fps,
                use_system_prompt=self.config.use_system_prompt,
                add_resolution_template=self.config.add_resolution_template,
                add_duration_template=self.config.add_duration_template,
                action_mode=action_mode,
                action_view_point=self.config.action_view_point if action_mode else None,
            )
            if len(self._token_cache) >= self._token_cache_max:
                self._token_cache.popitem(last=False)
            self._token_cache[key] = ids
        else:
            self._token_cache.move_to_end(key)
        return ids

    def flow_shift(self, height: int, width: int) -> float:
        """Resolve an explicit override, then the upstream resolution tier."""
        cfg = self.config
        if cfg.flow_shift is None and cfg.flow_shift_by_resolution:
            shift = cfg.flow_shift_by_resolution.get(resolution_tier(height, width))
            if shift is not None:
                return float(shift)
        return self.bundle.flow_shift

    def predict_velocity(
        self,
        *,
        input_ids: Sequence[int],
        x0: torch.Tensor,
        fps: float,
        sigma: torch.Tensor,
        actions: Optional[torch.Tensor],
        generator: Optional[torch.Generator],
    ) -> Cosmos3JointPrediction:
        """Noise one sample at ``sigma`` (fp32 scalar in (0,1)), run one packed forward, align targets."""
        cfg = self.config
        if x0.ndim != 4:
            raise ValueError(f"Cosmos3JointStage: x0 must be [C,T,H,W], got {tuple(x0.shape)}")
        x0 = x0.unsqueeze(0).float()
        condition_frames = [0] if (cfg.condition_on_first_frame and x0.shape[2] > 1) else []
        sigma = sigma.to(device=x0.device, dtype=torch.float32).reshape(())
        x_t, vision_target = noise_vision_latents(x0, sigma, condition_frames, generator)

        action_kwargs: dict[str, Any] = {}
        action_target: Optional[torch.Tensor] = None
        if actions is not None:
            from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import _EMBODIMENT_TO_DOMAIN_ID

            if cfg.action_domain_name not in _EMBODIMENT_TO_DOMAIN_ID:
                raise ValueError(f"Unknown Cosmos3 action_domain_name={cfg.action_domain_name!r}.")
            x0_action = pad_action_chunk(actions.float(), self.bundle.transformer.config.action_dim)
            x_t_action, action_target = noise_action_latents(
                x0_action,
                sigma,
                cfg.raw_action_dim,
                generator,
            )
            action_kwargs = {
                "action_tokens": x_t_action,
                "action_condition_frame_indexes": (),
                "action_domain_id": torch.tensor(
                    [_EMBODIMENT_TO_DOMAIN_ID[cfg.action_domain_name]],
                    dtype=torch.long,
                    device=x0.device,
                ),
                "action_fps": fps,
            }

        kwargs, meta = pack_joint_sequence(
            self.pipe,
            input_ids=input_ids,
            vision_tokens=x_t,
            condition_frame_indexes=condition_frames,
            vision_fps=fps,
            device=x0.device,
            compute_dtype=self.bundle.dtype,
            **action_kwargs,
        )
        # Continuous fp32 sigma*T like official Cosmos training; README: timestep conditioning.
        timestep = float(sigma.item()) * self.pipe.scheduler.config.num_train_timesteps
        kwargs["vision_timesteps"] = torch.full(
            (meta["num_noisy_vision_tokens"],),
            timestep,
            dtype=torch.float32,
            device=x0.device,
        )
        if actions is not None:
            kwargs["action_timesteps"] = torch.full(
                (meta["num_noisy_action_tokens"],),
                timestep,
                dtype=torch.float32,
                device=x0.device,
            )

        if meta["num_noisy_vision_tokens"] <= 0:
            raise RuntimeError(
                "Cosmos3JointStage: no noisy vision tokens (latent T="
                f"{tuple(x_t.shape)} cond_frames={condition_frames}). "
                "condition_on_first_frame with a single latent frame leaves an empty GEMM."
            )
        # diffusers 0.39 returns a bare (vision, sound, action) tuple, >=0.40 an output object (README # Gotchas).
        output = self.bundle.transformer(**kwargs)
        if isinstance(output, tuple):
            preds_vision, _preds_sound, preds_action = output
        else:
            preds_vision, preds_action = output.sample, output.action
        if not preds_vision:
            raise RuntimeError("Cosmos3JointStage: transformer returned no vision prediction.")
        vision_pred = preds_vision[0]
        if vision_pred.ndim == 4:
            vision_pred = vision_pred.unsqueeze(0)
        noisy_frames = meta["vision_noisy_frames"]
        vision_pred = vision_pred[:, :, noisy_frames]
        vision_target = vision_target[:, :, noisy_frames]
        if vision_pred.shape != vision_target.shape:
            raise RuntimeError(
                "Cosmos3JointStage: vision prediction/target shape mismatch: "
                f"{tuple(vision_pred.shape)} != {tuple(vision_target.shape)}."
            )

        action_pred: Optional[torch.Tensor] = None
        if actions is not None:
            if not preds_action:
                raise RuntimeError("Cosmos3JointStage: transformer returned no action prediction.")
            action_pred = preds_action[0]
            if action_pred.ndim == 3:
                action_pred = action_pred.squeeze(0)
            raw_dim = cfg.raw_action_dim
            action_pred = action_pred[:, :raw_dim]
            assert action_target is not None
            action_target = action_target[:, :raw_dim]
            if action_pred.shape != action_target.shape:
                raise RuntimeError(
                    "Cosmos3JointStage: action prediction/target shape mismatch: "
                    f"{tuple(action_pred.shape)} != {tuple(action_target.shape)}."
                )

        return Cosmos3JointPrediction(
            vision_pred=vision_pred,
            vision_target=vision_target,
            action_pred=action_pred,
            action_target=action_target,
            sigma=sigma,
        )


def _token_condition(rows: Sequence[Sequence[int]]) -> TextTokenCondition:
    """Per-sample token id lists to right-padded ``[B, L]`` ids plus mask."""
    return TextTokenCondition.concat(
        [
            TextTokenCondition(input_ids=torch.tensor([row]), attention_mask=torch.ones(1, len(row), dtype=torch.long))
            for row in rows
        ]
    )


class Cosmos3Pipeline(Pipeline):
    """Cosmos3 pipeline: packed joint stage for SFT plus single-frame text-to-image generation."""

    def __init__(
        self,
        *,
        bundle: Cosmos3Bundle,
        strategy: Optional[StepStrategy] = None,
        shift: Optional[float] = None,
        trajectory_precision: str = "bf16",
        logprob_precision: str = "fp32",
    ) -> None:
        super().__init__()
        self.bundle = bundle
        self.joint = Cosmos3JointStage(bundle)
        self.diffusion = Cosmos3DiffusionStage(
            model=bundle,
            step=Cosmos3DiffusionStep(self.joint.pipe),
            strategy=strategy if strategy is not None else FlowSDEStrategy(),
            trajectory_precision=trajectory_precision,
            logprob_precision=logprob_precision,
        )
        self.shift = shift

    @classmethod
    def latent_shape(cls, *, model_config: Any, sampling_spec: Any) -> tuple:
        """Per-sample latent shape ``(48, 1, H/16, W/16)`` for driver-side initial noise."""
        return (48, 1, sampling_spec.height // 16, sampling_spec.width // 16)

    def build_schedule_policy(self) -> FlowMatchSchedulePolicy:
        """Static flow shift; the checkpoint's UniPC/Karras scheduler config is not used (README # Gotchas)."""
        if self.shift is None:
            raise ValueError("Cosmos3Pipeline.build_schedule_policy: generation needs an explicit `shift`.")
        return FlowMatchSchedulePolicy.static_only(self.shift)

    def build_conditions(self, texts: Texts, *, height: int, width: int) -> Cosmos3Conditions:
        """Tokenize prompts into cond / uncond ids with the upstream single-frame templates."""
        cfg = self.bundle.config
        cond_rows, uncond_rows = [], []
        for prompt in texts.texts:
            cond_ids, uncond_ids = self.joint.pipe.tokenize_prompt(
                prompt,
                None,
                num_frames=1,
                height=height,
                width=width,
                fps=cfg.fps,
                use_system_prompt=cfg.use_system_prompt,
                add_resolution_template=cfg.add_resolution_template,
                add_duration_template=cfg.add_duration_template,
            )
            cond_rows.append(cond_ids)
            uncond_rows.append(uncond_ids)
        return Cosmos3Conditions(text=_token_condition(cond_rows), negative_text=_token_condition(uncond_rows))

    @torch.no_grad()
    def decode(self, segment: LatentSegment) -> Images:
        """Decode final latents ``[B, C, 1, H_lat, W_lat]`` to images in ``[0, 1]``."""
        vae, pipe = self.bundle.vae, self.joint.pipe
        latents = segment.latents[:, -1].to(device=vae.device, dtype=vae.dtype)
        mean = pipe._vae_latents_mean.to(latents).view(1, -1, 1, 1, 1)
        inv_std = pipe._vae_latents_inv_std.to(latents).view(1, -1, 1, 1, 1)
        decoded = vae.decode(latents / inv_std + mean).sample  # [B, 3, 1, H, W] in [-1, 1]
        return Images.from_dense(((decoded[:, :, 0].float() + 1.0) / 2.0).clamp(0.0, 1.0))

    def generate(self, sample: Sample) -> Sample:
        """Run single-frame text-to-image, filling the frontier (pre-forked) gen Part."""
        frontier = sample.parts[-1]
        params = frontier.sampling_params
        if not isinstance(params, DiffusionSamplingParams):
            raise TypeError(
                f"Cosmos3Pipeline.generate: frontier gen Part must carry DiffusionSamplingParams, "
                f"got {type(params).__name__ if params is not None else 'None'}"
            )
        if params.sigmas is None:
            raise ValueError(
                "Cosmos3Pipeline.generate: gen part sampling_params.sigmas is None; the hosting engine "
                "must pin sigmas before invoking pipeline.generate."
            )
        conditioning = sample.conditioning()
        texts = conditioning[0] if conditioning else None
        if not isinstance(texts, Texts):
            raise TypeError(
                f"Cosmos3Pipeline.generate: expected a Texts prompt from sample.conditioning()[0], "
                f"got {type(texts).__name__ if texts is not None else 'None'}"
            )
        if params.init_same_noise and not params.noise_group_ids:
            params = dataclasses.replace(params, noise_group_ids=frontier.group_ids)

        conditions = self.build_conditions(texts, height=params.height, width=params.width)
        segment = self.diffusion.diffuse(
            conditions,
            schedule=params.sigmas.to(self.bundle.device),
            params=params,
            initial_latents=NoiseRecipe.from_sample(sample).resolve(),
        )
        images = self.decode(segment)
        filled = frontier.fill(segment=segment, primitives={"image": images}, conditions=conditions.to_dict())
        return Sample(parts=[*sample.parts[:-1], filled], reward_compute_s=sample.reward_compute_s)


__all__ = ["Cosmos3JointPrediction", "Cosmos3JointStage", "Cosmos3Pipeline"]
