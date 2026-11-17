"""Step-major batched and serial SDE transition replay for single-stream and joint video/audio stages."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from unirl.models.types.diffusion import combine_modality_logp
from unirl.models.types.replay_result import ReplayResult
from unirl.sde.kernels import SDEStrategy

if TYPE_CHECKING:
    from unirl.sde.kernels import StepStrategy
    from unirl.types.segments.latent import LatentSegment


class BatchedStepReplayMixin:
    """Step-major batched replay (``[S*B, ...]``) with automatic serial fallback for SDE stages."""

    def batched_step_kwargs(self, segment: "LatentSegment", params: Any) -> Dict[str, Any]:
        """Return model-specific arguments for the step call."""
        return {}

    def resolve_replay_steps(
        self,
        segment: "LatentSegment",
        step_indices: Optional[Sequence[int]],
        *,
        expected_ndim: Optional[int] = None,
        ndim_hint: str = "",
    ) -> Tuple[List[int], Dict[int, int]]:
        """Validate ``segment`` trajectory fields and return ``(target_steps, step_to_slot)``."""
        owner = type(self).__name__
        if segment.sde_indices is None or segment.latents is None:
            raise ValueError(f"{owner}.replay: segment.sde_indices / latents missing")
        if segment.sigmas is None or segment.sigmas.shape[0] < 2:
            raise ValueError(f"{owner}.replay: segment.sigmas missing or shorter than 2")
        if segment.indices is None:
            raise RuntimeError(f"{owner}.replay: segment.indices missing")
        if expected_ndim is not None and segment.latents.ndim != expected_ndim:
            raise ValueError(f"{owner}.replay: expected latents {ndim_hint}, got {tuple(segment.latents.shape)}")

        sde_list = segment.sde_indices.tolist()
        sde_set = set(sde_list)
        target = list(step_indices) if step_indices is not None else sde_list
        bad = [i for i in target if i not in sde_set]
        if bad:
            raise ValueError(f"{owner}.replay: step_indices {bad} not in segment.sde_indices={sorted(sde_set)}")

        slot_map = {step: slot for slot, step in enumerate(segment.indices.tolist())}
        for i in target:
            if i not in slot_map or (i + 1) not in slot_map:
                raise KeyError(f"{owner}.replay: step_idx={i} or {i + 1} not in stored indices={list(slot_map)}")
        return target, slot_map

    def replay_steps(
        self,
        conditions: Any,
        *,
        segment: "LatentSegment",
        params: Any,
        step_indices: Optional[Sequence[int]] = None,
        device: Optional[torch.device] = None,
        expected_ndim: Optional[int] = None,
        ndim_hint: str = "",
    ) -> ReplayResult:
        """Replay ``step_indices`` (batched when ``batch_replay_steps and len(target) > 1`` on an ``SDEStrategy``)."""
        owner = type(self).__name__
        target, slot_map = self.resolve_replay_steps(
            segment,
            step_indices,
            expected_ndim=expected_ndim,
            ndim_hint=ndim_hint,
        )
        dev = segment.latents.device if device is None else device
        if not target:
            return ReplayResult(
                log_probs=torch.empty(segment.latents.shape[0], 0, device=dev, dtype=self.logprob_dtype),
                prev_sample_means=None,
            )
        sigmas = segment.sigmas.to(dev)
        sigma_max = sigmas[1].float()
        step_kwargs = self.batched_step_kwargs(segment, params)
        groups = (
            [target]
            if self.batch_replay_steps and len(target) > 1 and isinstance(self.strategy, SDEStrategy)
            else [[step_idx] for step_idx in target]
        )

        latents = segment.latents
        B = latents.shape[0]
        log_prob_parts: List[torch.Tensor] = []
        mean_parts: List[torch.Tensor] = []

        for group in groups:
            S = len(group)
            if S == 1:
                step_idx = group[0]
                sample_all = latents[:, slot_map[step_idx]].to(dev)
                prev_all = latents[:, slot_map[step_idx + 1]].to(dev)
                sigma_all = sigmas[step_idx].to(torch.float32)
                sigma_next_all = sigmas[step_idx + 1].to(torch.float32)
                tiled = conditions
            else:
                sample_all = torch.cat([latents[:, slot_map[i]].to(dev) for i in group], dim=0)
                prev_all = torch.cat([latents[:, slot_map[i + 1]].to(dev) for i in group], dim=0)
                idx = torch.tensor(group, dtype=torch.long, device=sigmas.device)
                sigma_all = sigmas.index_select(0, idx).to(torch.float32).repeat_interleave(B)
                sigma_next_all = sigmas.index_select(0, idx + 1).to(torch.float32).repeat_interleave(B)
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
                step_index=group[0],
                **step_kwargs,
            )
            if log_prob_all is None:
                raise RuntimeError(
                    f"{owner}.replay: strategy returned None log-prob "
                    f"at step_index={group[0]} (deterministic mode); replay requires a stochastic SDE strategy."
                )
            log_prob_parts.append(log_prob_all.view(S, B).transpose(0, 1).contiguous().to(dtype=self.logprob_dtype))
            if prev_mean_all is not None:
                tail = prev_mean_all.shape[1:]
                mean_parts.append(prev_mean_all.view(S, B, *tail).transpose(0, 1).contiguous())

        log_probs_t = log_prob_parts[0] if len(log_prob_parts) == 1 else torch.cat(log_prob_parts, dim=1)
        means_t = (mean_parts[0] if len(mean_parts) == 1 else torch.cat(mean_parts, dim=1)) if mean_parts else None
        return ReplayResult(log_probs=log_probs_t, prev_sample_means=means_t)

    def replay_joint_av_steps(
        self,
        conditions: Any,
        *,
        segment: "LatentSegment",
        params: Any,
        step_indices: Optional[Sequence[int]] = None,
        sigmas: torch.Tensor,
        device: torch.device,
        predict_av_noise: Callable[
            [Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor],
        ],
        audio_in_policy: bool,
        audio_strategy: Optional["StepStrategy"] = None,
        audio_sigmas: Optional[torch.Tensor] = None,
        concat_audio_mean: bool = False,
        latent_dtype: Optional[torch.dtype] = None,
    ) -> ReplayResult:
        """Replay joint A/V steps (batched when ``batch_replay_steps and len(target) > 1`` on an ``SDEStrategy``)."""
        owner = type(self).__name__
        target, slot_map = self.resolve_replay_steps(segment, step_indices)
        if not target:
            return ReplayResult(
                log_probs=torch.empty(segment.latents.shape[0], 0, device=device, dtype=self.logprob_dtype),
                prev_sample_means=None,
            )
        sigma_max = sigmas[1].float()
        a_strat = self.strategy if audio_strategy is None else audio_strategy
        a_sigmas = sigmas if audio_sigmas is None else audio_sigmas
        a_sigma_max = sigma_max if audio_sigmas is None else a_sigmas[1].float()
        groups = (
            [target]
            if self.batch_replay_steps and len(target) > 1 and isinstance(self.strategy, SDEStrategy)
            else [[step_idx] for step_idx in target]
        )

        latents = segment.latents
        aux_latents = segment.aux_latents
        B = latents.shape[0]
        to_kwargs: Dict[str, Any] = (
            {"device": device} if latent_dtype is None else {"device": device, "dtype": latent_dtype}
        )
        log_prob_parts: List[torch.Tensor] = []
        mean_parts: List[torch.Tensor] = []

        for group in groups:
            S = len(group)
            if S == 1:
                step_idx = group[0]
                s_cur, s_next = slot_map[step_idx], slot_map[step_idx + 1]
                video_all = latents[:, s_cur].to(**to_kwargs)
                prev_video_all = latents[:, s_next].to(**to_kwargs)
                audio_all = aux_latents[:, s_cur].to(**to_kwargs)
                video_sigma_all = sigmas[step_idx].to(torch.float32).expand(B)
                video_sigma_next_all = sigmas[step_idx + 1].to(torch.float32)
                audio_sigma_all = a_sigmas[step_idx].to(torch.float32).expand(B)
                audio_sigma_next_all = a_sigmas[step_idx + 1].to(torch.float32)
                tiled = conditions
            else:
                video_all = torch.cat([latents[:, slot_map[i]].to(**to_kwargs) for i in group], dim=0)
                prev_video_all = torch.cat([latents[:, slot_map[i + 1]].to(**to_kwargs) for i in group], dim=0)
                audio_all = torch.cat([aux_latents[:, slot_map[i]].to(**to_kwargs) for i in group], dim=0)
                idx = torch.tensor(group, dtype=torch.long, device=sigmas.device)
                video_sigma_all = sigmas.index_select(0, idx).to(torch.float32).repeat_interleave(B)
                video_sigma_next_all = sigmas.index_select(0, idx + 1).to(torch.float32).repeat_interleave(B)
                audio_sigma_all = a_sigmas.index_select(0, idx).to(torch.float32).repeat_interleave(B)
                audio_sigma_next_all = a_sigmas.index_select(0, idx + 1).to(torch.float32).repeat_interleave(B)
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
                step_index=group[0],
            )
            if log_prob_all is None:
                raise RuntimeError(
                    f"{owner}.replay: strategy returned None log-prob "
                    f"at step_index={group[0]} (deterministic mode); replay requires a stochastic SDE strategy."
                )

            if audio_in_policy:
                prev_audio_all = (
                    aux_latents[:, slot_map[group[0] + 1]].to(**to_kwargs)
                    if S == 1
                    else torch.cat([aux_latents[:, slot_map[i + 1]].to(**to_kwargs) for i in group], dim=0)
                )
                _, audio_log_prob_all, audio_prev_mean_all = a_strat.denoise(
                    noise_pred=audio_pred,
                    sample=audio_all,
                    sigma=audio_sigma_all,
                    sigma_next=audio_sigma_next_all,
                    eta=params.eta,
                    prev_sample=prev_audio_all,
                    sigma_max=a_sigma_max,
                    step_index=group[0],
                )
                if audio_log_prob_all is None:
                    raise RuntimeError(
                        f"{owner}.replay: audio strategy returned None "
                        f"log-prob at step_index={group[0]}; joint audio SDE requires a stochastic strategy."
                    )
                log_prob_all = combine_modality_logp(
                    log_prob_all,
                    audio_log_prob_all,
                    n_video=video_all[0].numel(),
                    n_audio=audio_all[0].numel(),
                )
                if concat_audio_mean and prev_mean_all is not None and audio_prev_mean_all is not None:
                    prev_mean_all = torch.cat([prev_mean_all, audio_prev_mean_all], dim=1)

            log_prob_parts.append(log_prob_all.view(S, B).transpose(0, 1).contiguous().to(dtype=self.logprob_dtype))
            if prev_mean_all is not None:
                tail = prev_mean_all.shape[1:]
                mean_parts.append(prev_mean_all.view(S, B, *tail).transpose(0, 1).contiguous())

        log_probs_t = log_prob_parts[0] if len(log_prob_parts) == 1 else torch.cat(log_prob_parts, dim=1)
        means_t = (mean_parts[0] if len(mean_parts) == 1 else torch.cat(mean_parts, dim=1)) if mean_parts else None
        return ReplayResult(log_probs=log_probs_t, prev_sample_means=means_t)


__all__ = ["BatchedStepReplayMixin"]
