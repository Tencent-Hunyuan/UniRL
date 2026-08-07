"""Block-wise causal WAN rollout used by Self-Forcing training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .causal import WAN21CausalCache
from .conditions import WAN21Conditions


@dataclass
class WAN21SelfForcingOutput:
    latents: torch.Tensor
    gradient_mask: torch.Tensor
    exit_step: int


class WAN21SelfForcingStage:
    """Few-step block rollout with detached generated history.

    The implementation follows the core Self-Forcing exposure pattern:
    only initial noise and text are provided, while every later temporal
    block conditions on K/V from the model's own earlier blocks.
    """

    def __init__(
        self,
        *,
        diffusion: Any,
        frames_per_block: int = 1,
        denoising_sigmas: Sequence[float] = (1.0, 0.75, 0.5, 0.25),
        context_sigma: float = 0.0,
        guidance_scale: float = 1.0,
    ) -> None:
        if frames_per_block < 1:
            raise ValueError(f"frames_per_block must be >= 1; got {frames_per_block}.")
        sigmas = tuple(float(value) for value in denoising_sigmas)
        if not sigmas or any(not 0.0 < value <= 1.0 for value in sigmas):
            raise ValueError(f"denoising_sigmas must lie in (0, 1]; got {sigmas}.")
        if any(left <= right for left, right in zip(sigmas, sigmas[1:])):
            raise ValueError(f"denoising_sigmas must be strictly decreasing; got {sigmas}.")
        if not 0.0 <= context_sigma < 1.0:
            raise ValueError(f"context_sigma must lie in [0, 1); got {context_sigma}.")
        self.diffusion = diffusion
        self.frames_per_block = int(frames_per_block)
        self.denoising_sigmas = sigmas
        self.context_sigma = float(context_sigma)
        self.guidance_scale = float(guidance_scale)

    def _predict(
        self,
        conditions: WAN21Conditions,
        sample: torch.Tensor,
        sigma: float,
        *,
        cache: WAN21CausalCache,
        frame_offset: int,
        commit_cache: bool,
    ) -> torch.Tensor:
        sigma_tensor = torch.tensor(sigma, device=sample.device, dtype=torch.float32)
        return self.diffusion.step.predict_noise(
            self.diffusion.model,
            sample,
            sigma_tensor,
            conditions,
            guidance_scale=self.guidance_scale,
            attention_kwargs={
                "frames_per_block": self.frames_per_block,
                "frame_offset": frame_offset,
                "kv_cache": cache,
                "commit_cache": commit_cache,
            },
        )

    def rollout(
        self,
        conditions: WAN21Conditions,
        *,
        initial_noise: torch.Tensor,
        exit_step: int | None = None,
        generator: torch.Generator | None = None,
    ) -> WAN21SelfForcingOutput:
        if initial_noise.ndim != 5:
            raise ValueError(f"initial_noise must be [B,C,T,H,W], got {tuple(initial_noise.shape)}.")
        total_frames = int(initial_noise.shape[2])
        if total_frames % self.frames_per_block:
            raise ValueError(
                f"latent frames {total_frames} must be divisible by frames_per_block={self.frames_per_block}."
            )
        if exit_step is None:
            exit_step = int(
                torch.randint(
                    len(self.denoising_sigmas),
                    (1,),
                    device=initial_noise.device,
                    generator=generator,
                ).item()
            )
        if not 0 <= exit_step < len(self.denoising_sigmas):
            raise ValueError(f"exit_step must lie in [0,{len(self.denoising_sigmas)}), got {exit_step}.")

        cache = WAN21CausalCache.empty(len(self.diffusion.model.transformer.blocks))
        blocks = []
        grad_masks = []
        for frame_offset in range(0, total_frames, self.frames_per_block):
            x = initial_noise[:, :, frame_offset : frame_offset + self.frames_per_block]
            x0 = None
            for step_index, sigma in enumerate(self.denoising_sigmas):
                track_grad = step_index == exit_step
                with torch.set_grad_enabled(track_grad):
                    velocity = self._predict(
                        conditions,
                        x,
                        sigma,
                        cache=cache,
                        frame_offset=frame_offset,
                        commit_cache=False,
                    )
                    x0 = x - float(sigma) * velocity
                if track_grad:
                    break
                next_sigma = self.denoising_sigmas[step_index + 1]
                with torch.no_grad():
                    eps = torch.randn(x0.shape, device=x0.device, dtype=torch.float32, generator=generator)
                    x = (1.0 - next_sigma) * x0 + next_sigma * eps

            assert x0 is not None
            blocks.append(x0)
            grad_masks.append(torch.ones_like(x0, dtype=torch.bool))

            with torch.no_grad():
                context = x0.detach()
                if self.context_sigma:
                    eps = torch.randn(
                        context.shape, device=context.device, dtype=torch.float32, generator=generator
                    )
                    context = (1.0 - self.context_sigma) * context + self.context_sigma * eps
                self._predict(
                    conditions,
                    context,
                    self.context_sigma,
                    cache=cache,
                    frame_offset=frame_offset,
                    commit_cache=True,
                )

        return WAN21SelfForcingOutput(
            latents=torch.cat(blocks, dim=2),
            gradient_mask=torch.cat(grad_masks, dim=2),
            exit_step=exit_step,
        )


__all__ = ["WAN21SelfForcingOutput", "WAN21SelfForcingStage"]
