"""LTX2 diffusion: per-step kernel + rollout-level stage.

Two classes:
- ``LTX2DiffusionStep`` — stateless per-step kernel.
- ``LTX2DiffusionStage`` — implements ``DiffusionStage[LTX2Conditions]``.

LTX2-specific deviations from other models:
- Unified video+audio latent space: video and audio are concatenated on the
  sequence dimension before the transformer, split after.
- Video uses SDE (stochastic, log_prob for RL gradients).
- Audio uses ODE (deterministic, no gradients) — trained jointly but not
  directly optimized by the RL signal.
- The transformer takes ``hidden_states`` (patchified latents) +
  ``encoder_hidden_states`` (text embeddings) + ``encoder_attention_mask``.
- Timestep is scaled by 1000 (flow matching convention).
- RoPE is computed internally by the transformer from spatial/temporal shapes.
"""

from __future__ import annotations

import inspect
from contextlib import nullcontext
from typing import ClassVar, List, Optional, Set, Tuple

import torch

from unirl.models.types.diffusion import DiffusionStage, DiffusionStep
from unirl.models.types.replay_result import ReplayResult
from unirl.sde.kernels import StepStrategy
from unirl.sde.noise import make_denoise_step_generators
from unirl.types.sampling import DiffusionSamplingParams, compute_trajectory_positions
from unirl.types.segments.latent import LatentSegment, make_video_segment
from unirl.utils.dtypes import parse_torch_dtype

from .bundle import LTX2Bundle
from .conditions import LTX2Conditions
from .config import LTX2_SPATIAL_COMPRESSION, LTX2_TEMPORAL_COMPRESSION

_LTX2_TIMESTEP_SCALE: float = 1000.0

_FORWARD_PARAMS_CACHE: dict = {}


def _filter_forward_kwargs(transformer, kwargs):
    cls = type(transformer)
    params = _FORWARD_PARAMS_CACHE.get(cls)
    if params is None:
        sig = inspect.signature(cls.forward)
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
            params = True
        else:
            params = set(sig.parameters)
        _FORWARD_PARAMS_CACHE[cls] = params
    if params is True:
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


# Co-denoise audio because LTX-2 feeds audio cross-attention into every video layer.
_LTX2_FRAME_RATE: float = 24.0

_LTX2_AUDIO_SAMPLING_RATE: int = 16000
_LTX2_AUDIO_HOP_LENGTH: int = 160
_LTX2_AUDIO_TEMPORAL_COMPRESSION: int = 4
_LTX2_AUDIO_MEL_BINS: int = 64
_LTX2_AUDIO_MEL_COMPRESSION: int = 4
_LTX2_AUDIO_LATENT_CHANNELS: int = 8


def _audio_num_frames(num_pixel_frames: int, fps: float) -> int:
    """Number of audio LATENT frames for a clip, matching diffusers:
    ``round(duration_s * sampling_rate / hop_length / temporal_compression)``.
    """
    duration_s = float(num_pixel_frames) / float(fps)
    per_s = _LTX2_AUDIO_SAMPLING_RATE / _LTX2_AUDIO_HOP_LENGTH / float(_LTX2_AUDIO_TEMPORAL_COMPRESSION)
    return max(1, int(round(duration_s * per_s)))


def _audio_packed_feature_dim() -> int:
    """Packed audio feature dim: ``latent_channels * (mel_bins // mel_compress)``
    (== ``transformer.config.audio_in_channels`` = 128 for LTX-2)."""
    return _LTX2_AUDIO_LATENT_CHANNELS * (_LTX2_AUDIO_MEL_BINS // _LTX2_AUDIO_MEL_COMPRESSION)


def audio_latent_shape(params: DiffusionSamplingParams) -> Tuple[int, int]:
    """Per-sample PACKED audio-latent shape ``(audio_t, 128)`` — the geometry the
    LTX-2 audio stream is allocated with. Exposed so the pipeline can resolve a
    matching audio x_T from the video NoiseRecipe
    (``recipe.resolve(salt="audio", latent_shape=…)``).
    """
    audio_t = _audio_num_frames(int(params.num_frames), _LTX2_FRAME_RATE)
    return (audio_t, _audio_packed_feature_dim())


def _combine_modality_logp(
    video_logp: torch.Tensor,
    audio_logp: torch.Tensor,
    n_video: int,
    n_audio: int,
) -> torch.Tensor:
    """Element-weighted mean of the per-step video/audio log-probs.

    Video and audio are stepped by the SAME stateless strategy, each returning a
    per-sample log-prob already meaned over its own latent dims. Weighting by the
    element counts reproduces the mean a single SDE over the concatenated
    ``[video|audio]`` latent would produce, so the joint log-prob keeps the same
    scale as the video-only path. Mirrors Flow-Factory ``combine_modality_log_prob``.
    """
    total = n_video + n_audio
    return (video_logp * n_video + audio_logp * n_audio) / total


class LTX2DiffusionStep(DiffusionStep[LTX2Bundle, LTX2Conditions]):
    """Per-step LTX2 denoising kernel — stateless.

    Handles the video-only forward (SDE path for RL). Audio is handled
    separately via ODE in the stage.
    """

    def predict_noise(
        self,
        model: LTX2Bundle,
        sample: torch.Tensor,
        sigma: torch.Tensor,
        conditions: LTX2Conditions,
        *,
        guidance_scale: float,
        latent_num_frames: int,
        latent_height: int,
        latent_width: int,
        audio_sample: torch.Tensor,
        audio_num_frames: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the LTX2 audiovisual transformer with optional CFG, returning
        BOTH the video and audio velocity predictions.

        LTX-2 co-denoises video + audio: the video forward depends on the
        current audio state via the per-layer audio→video cross-attention
        residual (``isolate_modalities=False``). So we feed the real audio
        latent every step and return both predictions; the stage ODE-steps the
        audio and SDE-steps the video.

        Args:
            sample: Patchified video latents (B, seq_v, C_v).
            audio_sample: Packed audio latents (B, seq_a, C_a=128).
            sigma: Current noise level (B,).
            latent_num_frames / latent_height / latent_width: Video LATENT grid
                dims (post-VAE-compression) for video RoPE coords.
            audio_num_frames: Audio LATENT frame count for audio RoPE coords.

        Returns:
            ``(video_velocity [B, seq_v, C_v], audio_velocity [B, seq_a, C_a])``.
        """
        transformer = model.transformer
        timestep = (sigma * _LTX2_TIMESTEP_SCALE).to(sample.device)

        text_cond = conditions.text
        encoder_hidden_states = text_cond.embeds
        encoder_attention_mask = text_cond.attn_mask
        audio_text_cond = conditions.audio_text if conditions.audio_text is not None else text_cond
        audio_encoder_hidden_states = audio_text_cond.embeds
        audio_encoder_attention_mask = audio_text_cond.attn_mask

        def _run(v_in, a_in, ts_in, enc_hs, enc_mask, a_enc_hs, a_enc_mask):
            _fwd = dict(
                hidden_states=v_in,
                audio_hidden_states=a_in,
                encoder_hidden_states=enc_hs,
                audio_encoder_hidden_states=a_enc_hs,
                timestep=ts_in,
                audio_timestep=ts_in,
                sigma=ts_in,
                audio_sigma=ts_in,
                encoder_attention_mask=enc_mask,
                audio_encoder_attention_mask=a_enc_mask,
                num_frames=latent_num_frames,
                height=latent_height,
                width=latent_width,
                fps=_LTX2_FRAME_RATE,
                audio_num_frames=audio_num_frames,
                isolate_modalities=False,
                return_dict=False,
            )
            out = transformer(**_filter_forward_kwargs(transformer, _fwd))
            return out[0], out[1]

        if guidance_scale > 1.0 and conditions.negative_text is not None:
            neg = conditions.negative_text
            neg_audio = conditions.negative_audio_text if conditions.negative_audio_text is not None else neg

            def _cat_masks(negative_mask, positive_mask):
                if negative_mask is None or positive_mask is None:
                    if negative_mask is not None or positive_mask is not None:
                        raise ValueError("LTX-2 CFG attention masks must be both present or both absent")
                    return None
                return torch.cat([negative_mask, positive_mask], dim=0)

            v_cfg = torch.cat([sample, sample], dim=0)
            a_cfg = torch.cat([audio_sample, audio_sample], dim=0)
            ts_cfg = torch.cat([timestep, timestep], dim=0)
            enc_hs = torch.cat([neg.embeds, encoder_hidden_states], dim=0)
            enc_mask = _cat_masks(neg.attn_mask, encoder_attention_mask)
            a_enc_hs = torch.cat([neg_audio.embeds, audio_encoder_hidden_states], dim=0)
            a_enc_mask = _cat_masks(neg_audio.attn_mask, audio_encoder_attention_mask)

            v_pred, a_pred = _run(v_cfg, a_cfg, ts_cfg, enc_hs, enc_mask, a_enc_hs, a_enc_mask)
            v_u, v_c = v_pred.chunk(2, dim=0)
            a_u, a_c = a_pred.chunk(2, dim=0)
            video_pred = v_u + guidance_scale * (v_c - v_u)
            audio_pred = a_u + guidance_scale * (a_c - a_u)
        else:
            video_pred, audio_pred = _run(
                sample,
                audio_sample,
                timestep,
                encoder_hidden_states,
                encoder_attention_mask,
                audio_encoder_hidden_states,
                audio_encoder_attention_mask,
            )

        return video_pred, audio_pred


class LTX2DiffusionStage(DiffusionStage[LTX2Conditions]):
    """LTX2 diffusion stage — owns the denoising loop and replay.

    FSDP wrapping hint: the transformer's block class is
    ``LTX2VideoTransformerBlock``.
    """

    _no_split_modules: ClassVar[List[str]] = ["LTX2VideoTransformerBlock"]

    def __init__(
        self,
        bundle: LTX2Bundle,
        *,
        strategy: StepStrategy,
        autocast_precision: str = "bf16",
        trajectory_precision: str = "fp16",
        logprob_precision: str = "fp32",
        audio_joint_sde: bool = True,
    ) -> None:
        self.bundle = bundle
        self.step_kernel = LTX2DiffusionStep()
        self.strategy = strategy
        self.autocast_dtype = parse_torch_dtype(autocast_precision, field_name="autocast_precision")
        self.trajectory_dtype = parse_torch_dtype(trajectory_precision, field_name="trajectory_precision")
        self.logprob_dtype = parse_torch_dtype(logprob_precision, field_name="logprob_precision")
        self.audio_joint_sde = bool(audio_joint_sde)
        self._audio_in_policy = self.audio_joint_sde and bool(getattr(bundle, "has_audio", False))

    def trainable_module(self) -> torch.nn.Module:
        """The trainable transformer (for FSDP wrapping)."""
        return self.bundle.transformer

    @staticmethod
    def _latent_geometry(params: DiffusionSamplingParams) -> tuple[int, int, int]:
        """Video LATENT grid ``(T_lat, H_lat, W_lat)`` from pixel-space params.

        Mirrors ``LTX2Pipeline.latent_shape``: 32x spatial, 8x temporal (causal,
        so ``T_lat = (num_frames - 1) // 8 + 1``). The transformer needs these
        to build video RoPE coords inside ``predict_noise``.
        """
        latent_t = (int(params.num_frames) - 1) // LTX2_TEMPORAL_COMPRESSION + 1
        latent_h = int(params.height) // LTX2_SPATIAL_COMPRESSION
        latent_w = int(params.width) // LTX2_SPATIAL_COMPRESSION
        return latent_t, latent_h, latent_w

    def generate(
        self,
        conditions: LTX2Conditions,
        *,
        params: DiffusionSamplingParams,
        sigmas: torch.Tensor,
        initial_latents: torch.Tensor,
        initial_audio_latents: Optional[torch.Tensor] = None,
        sde_indices: Optional[List[int]] = None,
        denoise_seed_keys: Optional[List[str]] = None,
        denoise_base_seed: int = 0,
    ) -> LatentSegment:
        """Run the full denoising loop, collecting trajectory for RL.

        Args:
            conditions: Text/image conditioning.
            params: Sampling parameters (guidance_scale, eta, etc.).
            sigmas: Sigma schedule (T+1,) from high → 0.
            initial_latents: Starting noise (B, seq, C) or (B, C, T, H, W).
            initial_audio_latents: Driver-authoritative audio x_T (B, audio_t,
                128) resolved by the pipeline from the same NoiseRecipe as video.
                ``None`` falls back to a bare randn (non-pipeline/test callers).
            sde_indices: Which steps to use SDE (stochastic) for RL.
            denoise_seed_keys: Stable per-sample keys used to derive the same
                per-step SDE noise as SGLang.
            denoise_base_seed: Base seed paired with ``denoise_seed_keys``.

        Returns:
            LatentSegment with trajectory and log-probs at SDE steps.
        """
        guidance_scale = float(params.guidance_scale)
        eta = float(params.eta)
        latent_t, latent_h, latent_w = self._latent_geometry(params)
        audio_t = _audio_num_frames(int(params.num_frames), _LTX2_FRAME_RATE)

        device = initial_latents.device
        if denoise_seed_keys is not None and len(denoise_seed_keys) != int(initial_latents.shape[0]):
            raise ValueError(
                "LTX2DiffusionStage.generate: denoise_seed_keys length "
                f"{len(denoise_seed_keys)} != batch size {int(initial_latents.shape[0])}"
            )
        num_steps = len(sigmas) - 1
        sigmas = sigmas.to(device)
        self.strategy.init_schedule(sigmas)

        sde_set: Set[int] = set(int(i) for i in sde_indices) if sde_indices else set(range(num_steps))
        sde_sorted: List[int] = sorted(sde_set)

        needed: Set[int] = set(compute_trajectory_positions(sde_set, num_steps))
        needed.add(num_steps)

        x = initial_latents.to(dtype=self.trajectory_dtype)
        if initial_audio_latents is not None:
            a = initial_audio_latents.to(device=device, dtype=self.trajectory_dtype)
        else:
            a = torch.randn(
                (int(x.shape[0]), audio_t, _audio_packed_feature_dim()),
                device=device,
                dtype=self.trajectory_dtype,
            )
        stored_pairs: List[tuple] = []
        stored_audio: List[torch.Tensor] = []
        if 0 in needed:
            stored_pairs.append((0, x.detach().clone()))
            stored_audio.append(a.detach().clone())
        sde_logp_list: List[torch.Tensor] = []

        autocast_ctx = (
            torch.autocast("cuda", dtype=self.autocast_dtype) if self.autocast_dtype != torch.float32 else nullcontext()
        )
        sigma_max = float(sigmas[1].item()) if int(sigmas.shape[0]) > 1 else 0.99

        with autocast_ctx:
            for step_idx in range(num_steps):
                sigma = sigmas[step_idx].to(device)
                sigma_next = sigmas[step_idx + 1].to(device)
                step_eta = eta if step_idx in sde_set else 0.0
                step_generators = (
                    make_denoise_step_generators(
                        base_seed=int(denoise_base_seed),
                        step_index=step_idx,
                        sample_ids=[str(key) for key in denoise_seed_keys],
                    )
                    if step_eta > 0.0 and denoise_seed_keys is not None
                    else None
                )

                video_pred, audio_pred = self.step_kernel.predict_noise(
                    self.bundle,
                    x,
                    sigma.expand(x.shape[0]),
                    conditions,
                    guidance_scale=guidance_scale,
                    latent_num_frames=latent_t,
                    latent_height=latent_h,
                    latent_width=latent_w,
                    audio_sample=a,
                    audio_num_frames=audio_t,
                )

                x_next, log_prob, _ = self.strategy.denoise(
                    noise_pred=video_pred,
                    sample=x,
                    sigma=sigma,
                    sigma_next=sigma_next,
                    eta=step_eta,
                    generator=step_generators,
                    sigma_max=sigma_max,
                    step_index=step_idx,
                )
                x = x_next.to(dtype=self.trajectory_dtype)

                audio_eta = step_eta if self._audio_in_policy else 0.0
                a_next, audio_log_prob, _ = self.strategy.denoise(
                    noise_pred=audio_pred,
                    sample=a,
                    sigma=sigma,
                    sigma_next=sigma_next,
                    eta=audio_eta,
                    sigma_max=sigma_max,
                    step_index=step_idx,
                )
                a = a_next.to(dtype=self.trajectory_dtype)

                if (step_idx + 1) in needed:
                    stored_pairs.append((step_idx + 1, x.detach().clone()))
                    stored_audio.append(a.detach().clone())
                if log_prob is not None:
                    if self._audio_in_policy and audio_log_prob is not None:
                        log_prob = _combine_modality_logp(
                            log_prob,
                            audio_log_prob,
                            n_video=x[0].numel(),
                            n_audio=a[0].numel(),
                        )
                    sde_logp_list.append(log_prob.to(dtype=self.logprob_dtype))

        positions = [p for p, _ in stored_pairs]
        latents_stacked = torch.stack([t for _, t in stored_pairs], dim=1)
        aux_stacked = torch.stack(stored_audio, dim=1)
        sde_logp = torch.stack(sde_logp_list, dim=1) if sde_logp_list else None
        sde_indices_t = torch.tensor(sde_sorted, dtype=torch.long, device=device) if sde_sorted else None
        indices_t = torch.tensor(positions, dtype=torch.long, device=device)

        return make_video_segment(
            latents=latents_stacked,
            aux_latents=aux_stacked,
            sigmas=sigmas,
            indices=indices_t,
            sde_logp=sde_logp,
            sde_indices=sde_indices_t,
        )

    def replay(
        self,
        conditions: LTX2Conditions,
        *,
        segment: LatentSegment,
        params: DiffusionSamplingParams,
        step_indices: Optional[List[int]] = None,
    ) -> ReplayResult:
        """Segment-based log-prob replay over the rollout's SDE transitions.

        For each target SDE step ``k`` we re-run the model at the stored
        ``sample = latents_at(k)`` and evaluate the log-prob of the stored
        transition to ``prev_sample = latents_at(k+1)`` (no fresh noise —
        ``strategy.denoise`` with ``prev_sample`` set is replay mode). Used by
        FlowGRPO for both the frozen π_old anchor and the trainable new_logp.
        Returns ``log_probs`` ``[B, len(target)]`` and ``prev_sample_means``
        for the KL penalty. Mirrors WAN21.
        """
        if segment.sde_indices is None or segment.latents is None or segment.sigmas is None:
            raise ValueError("LTX2DiffusionStage.replay: segment.sde_indices / latents / sigmas missing")
        if segment.aux_latents is None:
            raise ValueError(
                "LTX2DiffusionStage.replay: segment.aux_latents (audio trajectory) missing — "
                "the video forward cross-attends to the per-step audio state, so replay needs it. "
                "Was the segment produced by this stage's generate()?"
            )

        guidance_scale = float(params.guidance_scale)
        eta = float(params.eta)
        latent_t, latent_h, latent_w = self._latent_geometry(params)
        audio_t = _audio_num_frames(int(params.num_frames), _LTX2_FRAME_RATE)

        sde_set = set(int(i) for i in segment.sde_indices.tolist())
        target = [int(i) for i in (step_indices if step_indices is not None else segment.sde_indices.tolist())]
        bad = [i for i in target if i not in sde_set]
        if bad:
            raise ValueError(
                f"LTX2DiffusionStage.replay: step_indices {bad} not in segment.sde_indices={sorted(sde_set)}"
            )

        device = segment.latents.device
        sigmas = segment.sigmas.to(device)
        sigma_max = float(sigmas[1].item()) if int(sigmas.shape[0]) > 1 else 0.99

        log_probs: List[torch.Tensor] = []
        prev_sample_means: List[torch.Tensor] = []
        autocast_ctx = (
            torch.autocast("cuda", dtype=self.autocast_dtype) if self.autocast_dtype != torch.float32 else nullcontext()
        )

        with autocast_ctx:
            for step_idx in target:
                sigma = sigmas[step_idx].to(dtype=torch.float32)
                sigma_next = sigmas[step_idx + 1].to(dtype=torch.float32)
                sample = segment.latents_at(step_idx).to(device=device, dtype=self.trajectory_dtype)
                prev_sample = segment.latents_at(step_idx + 1).to(device=device, dtype=self.trajectory_dtype)
                audio_sample = segment.aux_latents_at(step_idx).to(device=device, dtype=self.trajectory_dtype)

                video_pred, audio_pred = self.step_kernel.predict_noise(
                    self.bundle,
                    sample,
                    sigma.expand(sample.shape[0]),
                    conditions,
                    guidance_scale=guidance_scale,
                    latent_num_frames=latent_t,
                    latent_height=latent_h,
                    latent_width=latent_w,
                    audio_sample=audio_sample,
                    audio_num_frames=audio_t,
                )

                _, log_prob, prev_mean = self.strategy.denoise(
                    noise_pred=video_pred,
                    sample=sample,
                    sigma=sigma,
                    sigma_next=sigma_next,
                    eta=eta,
                    prev_sample=prev_sample,
                    sigma_max=sigma_max,
                    step_index=step_idx,
                )
                if log_prob is None:
                    raise RuntimeError(
                        f"LTX2DiffusionStage.replay: strategy returned None log-prob at step_index={step_idx} "
                        f"(deterministic mode); replay requires a stochastic SDE strategy."
                    )

                if self._audio_in_policy:
                    audio_prev = segment.aux_latents_at(step_idx + 1).to(device=device, dtype=self.trajectory_dtype)
                    _, audio_log_prob, audio_prev_mean = self.strategy.denoise(
                        noise_pred=audio_pred,
                        sample=audio_sample,
                        sigma=sigma,
                        sigma_next=sigma_next,
                        eta=eta,
                        prev_sample=audio_prev,
                        sigma_max=sigma_max,
                        step_index=step_idx,
                    )
                    if audio_log_prob is None:
                        raise RuntimeError(
                            f"LTX2DiffusionStage.replay: audio strategy returned None log-prob at "
                            f"step_index={step_idx}; joint audio SDE requires a stochastic strategy."
                        )
                    log_prob = _combine_modality_logp(
                        log_prob,
                        audio_log_prob,
                        n_video=sample[0].numel(),
                        n_audio=audio_sample[0].numel(),
                    )
                    if prev_mean is not None and audio_prev_mean is not None:
                        prev_mean = torch.cat([prev_mean, audio_prev_mean], dim=1)

                log_probs.append(log_prob)
                if prev_mean is not None:
                    prev_sample_means.append(prev_mean)

        log_probs_t = torch.stack(log_probs, dim=1).to(dtype=self.logprob_dtype)
        means_t = torch.stack(prev_sample_means, dim=1).to(dtype=self.trajectory_dtype) if prev_sample_means else None
        return ReplayResult(log_probs=log_probs_t, prev_sample_means=means_t)


__all__ = ["LTX2DiffusionStep", "LTX2DiffusionStage"]
