"""Batched replay for stateless SDE stages — S steps stacked step-major as ``[S*B, ...]``, replay-anchor only."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import torch

from unirl.models.types.diffusion import combine_modality_logp
from unirl.models.types.replay_result import ReplayResult

if TYPE_CHECKING:
    from unirl.sde.kernels import StepStrategy
    from unirl.types.segments.latent import LatentSegment


class BatchedStepReplayMixin:
    """Step-major batched replay (``[S*B, ...]``) for single-stream and joint video/audio SDE stages."""

    def batched_step_kwargs(self, segment: "LatentSegment", params: Any) -> Dict[str, Any]:
        """Return model-specific arguments for the batched step call."""
        return {}

    def replay_batched_steps(
        self,
        conditions: Any,
        *,
        segment: "LatentSegment",
        params: Any,
        target: List[int],
        sigmas: torch.Tensor,
        sigma_max: torch.Tensor,
        device: torch.device,
    ) -> ReplayResult:
        """Replay ``target`` steps in one step-major batch; log-probs ``[B, S]``, slot ``s`` is ``target[s]``."""
        S = len(target)
        sample_all = torch.cat([segment.latents_at(i).to(device) for i in target], dim=0)
        prev_all = torch.cat([segment.latents_at(i + 1).to(device) for i in target], dim=0)
        B = sample_all.shape[0] // S
        sigma_all = torch.cat([sigmas[i].to(torch.float32).expand(B) for i in target], dim=0)
        sigma_next_all = torch.cat([sigmas[i + 1].to(torch.float32).expand(B) for i in target], dim=0)
        tiled = type(conditions).concat([conditions] * S)

        _, log_prob_all, prev_mean_all = self.step.step_with_logp(
            self.model,
            tiled,
            strategy=self.strategy,
            sample=sample_all,
            prev_sample=prev_all,
            sigma=sigma_all,
            sigma_next=sigma_next_all,
            guidance_scale=params.guidance_scale,
            eta=params.eta,
            sigma_max=sigma_max,
            step_index=target[0],
            **self.batched_step_kwargs(segment, params),
        )
        if log_prob_all is None:
            raise RuntimeError(
                f"{type(self).__name__}.replay_batched_steps: strategy returned None "
                f"log-prob (deterministic mode); batched replay requires a stochastic "
                f"SDE strategy."
            )
        log_probs_t = log_prob_all.view(S, B).transpose(0, 1).contiguous().to(dtype=self.logprob_dtype)
        tail = prev_mean_all.shape[1:]
        means_t = prev_mean_all.view(S, B, *tail).transpose(0, 1).contiguous()
        return ReplayResult(log_probs=log_probs_t, prev_sample_means=means_t)

    def replay_batched_joint_av_steps(
        self,
        conditions: Any,
        *,
        segment: "LatentSegment",
        params: Any,
        target: List[int],
        sigmas: torch.Tensor,
        sigma_max: float,
        device: torch.device,
        predict_av_noise: Callable[
            [Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor],
        ],
        audio_in_policy: bool,
        audio_strategy: Optional["StepStrategy"] = None,
        audio_sigmas: Optional[torch.Tensor] = None,
        audio_sigma_max: Optional[float] = None,
        concat_audio_mean: bool = False,
        latent_dtype: Optional[torch.dtype] = None,
    ) -> ReplayResult:
        """Replay joint video/audio ``target`` steps in one step-major batch."""
        S = len(target)
        a_strat = self.strategy if audio_strategy is None else audio_strategy
        a_sigmas = sigmas if audio_sigmas is None else audio_sigmas
        a_sigma_max = sigma_max if audio_sigma_max is None else audio_sigma_max
        to_kwargs: Dict[str, Any] = (
            {"device": device} if latent_dtype is None else {"device": device, "dtype": latent_dtype}
        )

        video_all = torch.cat([segment.latents_at(i).to(**to_kwargs) for i in target], dim=0)
        prev_video_all = torch.cat([segment.latents_at(i + 1).to(**to_kwargs) for i in target], dim=0)
        audio_all = torch.cat([segment.aux_latents_at(i).to(**to_kwargs) for i in target], dim=0)
        B = video_all.shape[0] // S

        video_sigma_all = torch.cat([sigmas[i].to(torch.float32).expand(B) for i in target], dim=0)
        video_sigma_next_all = torch.cat([sigmas[i + 1].to(torch.float32).expand(B) for i in target], dim=0)
        audio_sigma_all = torch.cat([a_sigmas[i].to(torch.float32).expand(B) for i in target], dim=0)
        audio_sigma_next_all = torch.cat([a_sigmas[i + 1].to(torch.float32).expand(B) for i in target], dim=0)
        tiled = type(conditions).concat([conditions] * S)

        video_pred, audio_pred = predict_av_noise(tiled, video_all, audio_all, video_sigma_all, audio_sigma_all)
        _, log_prob_all, prev_mean_all = self.strategy.denoise(
            noise_pred=video_pred,
            sample=video_all,
            sigma=video_sigma_all,
            sigma_next=video_sigma_next_all,
            eta=params.eta,
            prev_sample=prev_video_all,
            sigma_max=sigma_max,
            step_index=target[0],
        )
        if log_prob_all is None:
            raise RuntimeError(
                f"{type(self).__name__}.replay_batched_joint_av_steps: strategy returned None "
                f"log-prob (deterministic mode); batched replay requires a stochastic "
                f"SDE strategy."
            )

        if audio_in_policy:
            prev_audio_all = torch.cat([segment.aux_latents_at(i + 1).to(**to_kwargs) for i in target], dim=0)
            _, audio_log_prob_all, audio_prev_mean_all = a_strat.denoise(
                noise_pred=audio_pred,
                sample=audio_all,
                sigma=audio_sigma_all,
                sigma_next=audio_sigma_next_all,
                eta=params.eta,
                prev_sample=prev_audio_all,
                sigma_max=a_sigma_max,
                step_index=target[0],
            )
            if audio_log_prob_all is None:
                raise RuntimeError(
                    f"{type(self).__name__}.replay_batched_joint_av_steps: audio strategy returned None "
                    f"log-prob; joint audio SDE requires a stochastic strategy."
                )
            log_prob_all = combine_modality_logp(
                log_prob_all,
                audio_log_prob_all,
                n_video=video_all[0].numel(),
                n_audio=audio_all[0].numel(),
            )
            if concat_audio_mean:
                prev_mean_all = torch.cat([prev_mean_all, audio_prev_mean_all], dim=1)

        log_probs_t = log_prob_all.view(S, B).transpose(0, 1).contiguous().to(dtype=self.logprob_dtype)
        tail = prev_mean_all.shape[1:]
        means_t = prev_mean_all.view(S, B, *tail).transpose(0, 1).contiguous()
        return ReplayResult(log_probs=log_probs_t, prev_sample_means=means_t)


__all__ = ["BatchedStepReplayMixin"]
