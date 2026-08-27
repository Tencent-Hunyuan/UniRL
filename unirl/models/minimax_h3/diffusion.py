"""MiniMax-H3 diffusion step + stage."""

from __future__ import annotations

from copy import deepcopy
from typing import ClassVar, List, Optional, Tuple

import torch

from unirl.config.require import require
from unirl.models.types.diffusion import DiffusionStage, DiffusionStep
from unirl.models.types.replay_result import ReplayResult
from unirl.sde.kernels import StepStrategy
from unirl.sde.noise import make_denoise_step_generators
from unirl.sde.runtime import get_sigma_schedule
from unirl.types.sampling import DiffusionSamplingParams, compute_trajectory_positions
from unirl.types.segments.latent import LatentSegment, make_video_segment
from unirl.utils.dtypes import parse_torch_dtype

from .bundle import MiniMaxH3Bundle
from .conditions import MiniMaxH3Conditions
from .config import MINIMAX_H3_AUDIO_LATENT_CHANNELS
from .packing import MiniMaxH3Geometry, build_t2va_layout, row_timestep_plan


def _combine_modality_logp(
    video_logp: torch.Tensor,
    audio_logp: torch.Tensor,
    n_video: int,
    n_audio: int,
) -> torch.Tensor:
    """Element-weighted mean of the per-step video/audio log-probs."""
    total = n_video + n_audio
    return (video_logp * n_video + audio_logp * n_audio) / total


def pack_dual_streams(video_rows: torch.Tensor, audio_rows: torch.Tensor) -> torch.Tensor:
    """Flatten ``[B, V, Cv]`` + ``[B, A, Ca]`` into one ``[B, V*Cv + A*Ca]`` tensor."""
    batch = video_rows.shape[0]
    return torch.cat([video_rows.reshape(batch, -1), audio_rows.reshape(batch, -1)], dim=1)


def unpack_dual_streams(
    packed: torch.Tensor,
    geometry: MiniMaxH3Geometry,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Inverse of :func:`pack_dual_streams`, using ``geometry`` for the split."""
    batch = packed.shape[0]
    split = geometry.num_video_rows * geometry.video_token_dim
    expected = split + geometry.num_audio_rows * MINIMAX_H3_AUDIO_LATENT_CHANNELS
    require(
        int(packed.shape[1]) == expected,
        f"unpack_dual_streams: packed width {int(packed.shape[1])} != {expected} implied by geometry "
        f"({geometry.num_video_rows} video rows x {geometry.video_token_dim} + {geometry.num_audio_rows} audio rows "
        f"x {MINIMAX_H3_AUDIO_LATENT_CHANNELS}). The packed latent and the resolved geometry disagree.",
    )
    video_rows = packed[:, :split].reshape(batch, geometry.num_video_rows, geometry.video_token_dim)
    audio_rows = packed[:, split:].reshape(batch, geometry.num_audio_rows, MINIMAX_H3_AUDIO_LATENT_CHANNELS)
    return video_rows, audio_rows


class MiniMaxH3DiffusionStep(DiffusionStep[MiniMaxH3Bundle, MiniMaxH3Conditions]):
    """Per-step MiniMax-H3 denoising kernel -- stateless."""

    def __init__(self, bundle: MiniMaxH3Bundle) -> None:
        self.bundle = bundle

    def predict_noise(
        self,
        conditions: MiniMaxH3Conditions,
        *,
        video_sample: torch.Tensor,
        audio_sample: torch.Tensor,
        video_sigma: torch.Tensor,
        audio_sigma: torch.Tensor,
        layout,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One forward -> ``(video_velocity, audio_velocity)``, sign-corrected."""
        unique_timesteps, timestep_indices = row_timestep_plan(layout, video_sigma=video_sigma, audio_sigma=audio_sigma)
        device = video_sample.device
        video_velocity, audio_velocity = self.bundle.transformer(
            hidden_states=video_sample,
            audio_hidden_states=audio_sample,
            encoder_hidden_states=conditions.text.embeds,
            timestep=unique_timesteps.to(device),
            timestep_indices=timestep_indices.to(device),
            token_tags=layout.token_tags.to(device),
            position_ids=layout.position_ids.to(device),
            video_indices=layout.video_indices.to(device),
            audio_indices=layout.audio_indices.to(device),
            text_indices=layout.text_indices.to(device),
            return_dict=False,
        )
        # The one line that reconciles H3's data-ward velocity with the
        # noise-ward convention every downstream SDE path assumes. See module
        # docstring for the algebra.
        return -video_velocity, -audio_velocity


class MiniMaxH3DiffusionStage(DiffusionStage[MiniMaxH3Conditions]):
    """Rollout-level MiniMax-H3 stage: conditions -> ``LatentSegment``."""

    _no_split_modules: ClassVar[List[str]] = ["MiniMaxH3TransformerBlock"]

    def __init__(
        self,
        bundle: MiniMaxH3Bundle,
        strategy: StepStrategy,
        *,
        audio_shift: float,
        audio_joint_sde: bool = True,
        trajectory_precision: str = "fp16",
        logprob_precision: str = "fp32",
    ) -> None:
        self.bundle = bundle
        self.step = MiniMaxH3DiffusionStep(bundle)
        self.video_strategy = strategy
        self.audio_strategy = deepcopy(strategy)
        # StageAlgorithm's shared transition helpers read this conventional slot.
        self.strategy = self.video_strategy
        self.audio_shift = float(audio_shift)
        self.audio_joint_sde = bool(audio_joint_sde)
        self.trajectory_dtype = parse_torch_dtype(trajectory_precision, field_name="trajectory_precision")
        self.logprob_dtype = parse_torch_dtype(logprob_precision, field_name="logprob_precision")

    def trainable_module(self) -> torch.nn.Module:
        return self.bundle.transformer

    def audio_schedule(self, video_schedule: torch.Tensor) -> torch.Tensor:
        """The audio sigma grid, derived from the video one."""
        return get_sigma_schedule(
            num_steps=int(video_schedule.shape[0]) - 1,
            shift=self.audio_shift,
            device=video_schedule.device,
        ).to(dtype=video_schedule.dtype)

    def generate(
        self,
        conditions: MiniMaxH3Conditions,
        *,
        params: DiffusionSamplingParams,
        sigmas: torch.Tensor,
        geometry: MiniMaxH3Geometry,
        initial_latents: torch.Tensor,
        initial_audio_latents: torch.Tensor,
        sde_indices: Optional[List[int]] = None,
        denoise_seed_keys: Optional[List[str]] = None,
        denoise_base_seed: int = 0,
    ) -> LatentSegment:
        """Run the joint video+audio denoising loop and build the segment."""
        require(
            conditions.batch_size == 1,
            f"MiniMaxH3DiffusionStage: batch-1 only (got {conditions.batch_size}). The packed sequence carries "
            f"unbatched per-row metadata, so callers chunk instead: set rollout.forward_batch_size=1 and "
            f"stack.micro_batch_size=1. Mirrors the bagel navit recipe.",
        )
        conditions = conditions.trim_text_padding()
        num_text_tokens = int(conditions.text.embeds.shape[1])
        layout = build_t2va_layout(geometry, num_text_tokens)
        audio_sigmas = self.audio_schedule(sigmas)

        num_steps = int(sigmas.shape[0]) - 1
        require(
            num_steps == int(params.num_inference_steps),
            f"MiniMaxH3DiffusionStage: schedule holds {num_steps} transitions but params.num_inference_steps="
            f"{params.num_inference_steps}. UniRL counts model evaluations, so N steps require N+1 sigma points; "
            f"the vendored MiniMaxH3Scheduler counts those grid points and therefore uses set_timesteps(N+1).",
        )
        self.video_strategy.init_schedule(sigmas)
        self.audio_strategy.init_schedule(audio_sigmas)
        video_sigma_max = float(sigmas[1].item()) if sigmas.numel() > 1 else 0.99
        audio_sigma_max = float(audio_sigmas[1].item()) if audio_sigmas.numel() > 1 else 0.99

        device = self.bundle.device
        x = initial_latents.to(device=device, dtype=self.trajectory_dtype)
        a = initial_audio_latents.to(device=device, dtype=self.trajectory_dtype)

        sde_sorted = sorted(sde_indices) if sde_indices is not None else list(range(num_steps))
        sde_set = set(sde_sorted)
        needed = set(compute_trajectory_positions(sde_sorted, num_steps))
        # An SDE recipe gets the terminal position for free as the `i+1` of its
        # last index; `sde_indices=[]` does not, and stacks an empty list. (sd3)
        needed.add(num_steps)

        stored_pairs: List[Tuple[int, torch.Tensor]] = []
        stored_audio: List[torch.Tensor] = []
        sde_logp_list: List[torch.Tensor] = []
        if 0 in needed:
            stored_pairs.append((0, x.detach().clone()))
            stored_audio.append(a.detach().clone())

        audio_in_policy = self.audio_joint_sde
        for step_idx in range(num_steps):
            step_eta = float(params.eta) if step_idx in sde_set else 0.0
            # Per-STEP generators: the seed tuple is (base_seed, step_index,
            # sample_id), so this must be rebuilt inside the loop. Hoisting
            # it out would make each step draw from a running stream instead
            # of its own seed, and rollout/replay would stop agreeing across
            # engines. Mirrors ltx2.
            step_generators = (
                make_denoise_step_generators(
                    base_seed=int(denoise_base_seed),
                    step_index=step_idx,
                    sample_ids=[str(key) for key in denoise_seed_keys],
                )
                if step_eta > 0.0 and denoise_seed_keys is not None
                else None
            )
            audio_step_generators = (
                make_denoise_step_generators(
                    base_seed=int(denoise_base_seed),
                    step_index=step_idx,
                    sample_ids=[f"{key}::audio" for key in denoise_seed_keys],
                )
                if step_eta > 0.0 and audio_in_policy and denoise_seed_keys is not None
                else None
            )
            video_pred, audio_pred = self.step.predict_noise(
                conditions,
                video_sample=x,
                audio_sample=a,
                video_sigma=sigmas[step_idx],
                audio_sigma=audio_sigmas[step_idx],
                layout=layout,
            )

            x_next, log_prob, _ = self.video_strategy.denoise(
                noise_pred=video_pred,
                sample=x,
                sigma=sigmas[step_idx],
                sigma_next=sigmas[step_idx + 1],
                eta=step_eta,
                generator=step_generators,
                sigma_max=video_sigma_max,
                step_index=step_idx,
            )
            # Audio steps its OWN grid -- the divergence from LTX-2.3.
            audio_eta = step_eta if audio_in_policy else 0.0
            a_next, audio_log_prob, _ = self.audio_strategy.denoise(
                noise_pred=audio_pred,
                sample=a,
                sigma=audio_sigmas[step_idx],
                sigma_next=audio_sigmas[step_idx + 1],
                eta=audio_eta,
                generator=audio_step_generators,
                sigma_max=audio_sigma_max,
                step_index=step_idx,
            )
            x = x_next.to(dtype=self.trajectory_dtype)
            a = a_next.to(dtype=self.trajectory_dtype)

            if (step_idx + 1) in needed:
                stored_pairs.append((step_idx + 1, x.detach().clone()))
                stored_audio.append(a.detach().clone())
            if log_prob is not None:
                if audio_in_policy and audio_log_prob is not None:
                    log_prob = _combine_modality_logp(
                        log_prob, audio_log_prob, n_video=x[0].numel(), n_audio=a[0].numel()
                    )
                sde_logp_list.append(log_prob.to(dtype=self.logprob_dtype))

        positions = [p for p, _ in stored_pairs]
        return make_video_segment(
            latents=torch.stack([t for _, t in stored_pairs], dim=1),
            indices=torch.tensor(positions, dtype=torch.long, device=device),
            sigmas=sigmas.detach().clone(),
            sde_logp=torch.stack(sde_logp_list, dim=1) if sde_logp_list else None,
            sde_indices=(torch.tensor(sde_sorted, dtype=torch.long, device=device) if sde_sorted else None),
            aux_latents=torch.stack(stored_audio, dim=1),
        )

    def replay(
        self,
        conditions: MiniMaxH3Conditions,
        *,
        segment: LatentSegment,
        params: DiffusionSamplingParams,
        geometry: Optional[MiniMaxH3Geometry] = None,
        step_indices: Optional[List[int]] = None,
    ) -> ReplayResult:
        """Recompute log-probs for the stored transitions (training path)."""
        require(
            segment.sde_indices is not None and segment.latents is not None and segment.sigmas is not None,
            "MiniMaxH3DiffusionStage.replay: segment.sde_indices / latents / sigmas missing",
        )
        require(
            segment.aux_latents is not None,
            "MiniMaxH3DiffusionStage.replay: segment.aux_latents (audio trajectory) missing -- the packed forward "
            "couples video to the per-step audio state, so replay needs it. Was the segment produced by generate()?",
        )
        # The training path reaches replay through ``StageAlgorithm``, whose
        # contract is ``replay(conditions, segment=, params=, step_indices=)`` --
        # there is no geometry kwarg to pass. That is fine: geometry is a pure
        # function of the SHARED (height, width, num_frames) on ``params``, the
        # same values ``generate`` resolved from, so rebuilding it here yields
        # the identical row layout. The explicit argument stays as an override
        # for callers that already hold one.
        if geometry is None:
            geometry = MiniMaxH3Geometry.from_params(params)

        conditions = conditions.trim_text_padding()
        sigmas = segment.sigmas.to(self.bundle.device)
        audio_sigmas = self.audio_schedule(sigmas)
        self.video_strategy.init_schedule(sigmas)
        self.audio_strategy.init_schedule(audio_sigmas)
        video_sigma_max = float(sigmas[1].item()) if sigmas.numel() > 1 else 0.99
        audio_sigma_max = float(audio_sigmas[1].item()) if audio_sigmas.numel() > 1 else 0.99
        num_text_tokens = int(conditions.text.embeds.shape[1])
        layout = build_t2va_layout(geometry, num_text_tokens)

        stored = [int(i) for i in segment.sde_indices.tolist()]
        targets = [int(i) for i in (step_indices if step_indices is not None else stored)]

        log_probs: List[torch.Tensor] = []
        means: List[torch.Tensor] = []
        for step_idx in targets:
            x = segment.latents_at(step_idx).to(self.bundle.device)
            a = segment.aux_latents_at(step_idx).to(self.bundle.device)
            prev_x = segment.latents_at(step_idx + 1).to(self.bundle.device)
            prev_a = segment.aux_latents_at(step_idx + 1).to(self.bundle.device)

            video_pred, audio_pred = self.step.predict_noise(
                conditions,
                video_sample=x,
                audio_sample=a,
                video_sigma=sigmas[step_idx],
                audio_sigma=audio_sigmas[step_idx],
                layout=layout,
            )
            _, log_prob, mean = self.video_strategy.denoise(
                noise_pred=video_pred,
                sample=x,
                sigma=sigmas[step_idx],
                sigma_next=sigmas[step_idx + 1],
                eta=float(params.eta),
                prev_sample=prev_x,
                sigma_max=video_sigma_max,
                step_index=step_idx,
            )
            if self.audio_joint_sde:
                _, audio_log_prob, _ = self.audio_strategy.denoise(
                    noise_pred=audio_pred,
                    sample=a,
                    sigma=audio_sigmas[step_idx],
                    sigma_next=audio_sigmas[step_idx + 1],
                    eta=float(params.eta),
                    prev_sample=prev_a,
                    sigma_max=audio_sigma_max,
                    step_index=step_idx,
                )
                if audio_log_prob is not None:
                    log_prob = _combine_modality_logp(
                        log_prob, audio_log_prob, n_video=x[0].numel(), n_audio=a[0].numel()
                    )
            log_probs.append(log_prob.to(dtype=self.logprob_dtype))
            means.append(mean)

        return ReplayResult(
            log_probs=torch.stack(log_probs, dim=1),
            prev_sample_means=torch.stack(means, dim=1) if means else None,
        )

    def nft_clean_latents(self, segment: LatentSegment) -> torch.Tensor:
        """The clean ``x0`` a forward-process algorithm should train on."""
        require(
            segment.latents is not None and segment.aux_latents is not None,
            "MiniMaxH3DiffusionStage.nft_clean_latents: segment.latents / aux_latents missing -- the rollout must "
            "store the terminal position of both streams.",
        )
        return pack_dual_streams(segment.latents[:, -1], segment.aux_latents[:, -1])

    def predict_noise_at_step(
        self,
        conditions: MiniMaxH3Conditions,
        *,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        params: DiffusionSamplingParams,
    ) -> torch.Tensor:
        """One packed forward at an arbitrary ``(xt, sigma)`` -- no SDE iteration."""
        require(
            int(sample.shape[0]) == 1,
            f"MiniMaxH3DiffusionStage.predict_noise_at_step: batch-1 only (got {int(sample.shape[0])}). The packed "
            f"sequence carries unbatched per-row metadata, so callers chunk instead: set stack.micro_batch_size=1 "
            f"and rollout.forward_batch_size=1. Same discipline as generate().",
        )
        conditions = conditions.trim_text_padding()
        geometry = MiniMaxH3Geometry.from_params(params)
        video_sample, audio_sample = unpack_dual_streams(sample, geometry)
        layout = build_t2va_layout(geometry, int(conditions.text.embeds.shape[1]))
        shared_sigma = sigma.detach().reshape(-1)[0]
        video_pred, audio_pred = self.step.predict_noise(
            conditions,
            video_sample=video_sample.to(dtype=self.trajectory_dtype),
            audio_sample=audio_sample.to(dtype=self.trajectory_dtype),
            video_sigma=shared_sigma,
            audio_sigma=shared_sigma,
            layout=layout,
        )
        return pack_dual_streams(video_pred, audio_pred)


__all__ = [
    "MiniMaxH3DiffusionStage",
    "MiniMaxH3DiffusionStep",
    "pack_dual_streams",
    "unpack_dual_streams",
]
