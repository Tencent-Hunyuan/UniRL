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

from contextlib import nullcontext
from typing import ClassVar, List, Optional, Set, Tuple

import torch

from unirl.models.types.diffusion import DiffusionStage, DiffusionStep
from unirl.models.types.replay_result import ReplayResult
from unirl.sde.kernels import StepStrategy
from unirl.types.sampling import DiffusionSamplingParams
from unirl.types.segments.latent import LatentSegment, make_video_segment
from unirl.types.trajectory_store import compute_trajectory_positions
from unirl.utils.dtypes import parse_torch_dtype

from .bundle import LTX2Bundle
from .conditions import LTX2Conditions
from .config import LTX2_SPATIAL_COMPRESSION, LTX2_TEMPORAL_COMPRESSION

_LTX2_TIMESTEP_SCALE: float = 1000.0

# LTX-2 is a UNIFIED audiovisual transformer: ``forward`` always runs both the
# video and audio branches AND, by design, injects an audio→video cross-attn
# residual into the video stream at every layer (``hidden_states += a2v_gate *
# a2v_attn``). diffusers' default T2V path co-denoises a real audio latent
# stream with ``isolate_modalities=False`` — so to match the training/inference
# distribution we MUST do the same: maintain an audio latent alongside video,
# feed it each step, and keep that residual. (The earlier "1-frame zero audio +
# isolate_modalities=True" shortcut deleted the residual at all 48 layers →
# residual blur even after the schedule fix.) The audio branch runs ODE (no RL
# gradient); only video carries the SDE log-prob.
_LTX2_FRAME_RATE: float = 24.0

# Audio-latent geometry fallbacks (used when no audio_vae is loaded, i.e.
# enable_audio=False). Mirror diffusers ``Flux2KleinPipeline``/LTX2Pipeline
# defaults: 16kHz / hop 160 / temporal-compress 4 → 25 audio-latent frames per
# second; 8 latent channels, 64 mel bins, mel-compress 4 → packed feature dim
# 8 * (64/4) = 128 == transformer.config.audio_in_channels.
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

        # Text conditioning (video + audio share the connector's text embeds;
        # the audio branch has its own projection inside the transformer).
        text_cond = conditions.text
        encoder_hidden_states = text_cond.embeds
        encoder_attention_mask = text_cond.attn_mask
        audio_text_cond = conditions.audio_text if conditions.audio_text is not None else text_cond
        audio_encoder_hidden_states = audio_text_cond.embeds
        audio_encoder_attention_mask = audio_text_cond.attn_mask

        def _run(v_in, a_in, ts_in, enc_hs, enc_mask, a_enc_hs, a_enc_mask):
            out = transformer(
                hidden_states=v_in,
                audio_hidden_states=a_in,
                encoder_hidden_states=enc_hs,
                audio_encoder_hidden_states=a_enc_hs,
                timestep=ts_in,
                audio_timestep=ts_in,
                # ``sigma``/``audio_sigma`` are consumed only by LTX-2.3 prompt
                # modulation; harmless for 2.0 and required by 2.3 — pass them.
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
            # forward returns (video_out, audio_out).
            return out[0], out[1]

        if guidance_scale > 1.0 and conditions.negative_text is not None:
            # CFG: batch [uncond, cond] for both modalities.
            neg = conditions.negative_text
            neg_audio = conditions.negative_audio_text if conditions.negative_audio_text is not None else neg
            v_cfg = torch.cat([sample, sample], dim=0)
            a_cfg = torch.cat([audio_sample, audio_sample], dim=0)
            ts_cfg = torch.cat([timestep, timestep], dim=0)
            enc_hs = torch.cat([neg.embeds, encoder_hidden_states], dim=0)
            enc_mask = torch.cat([neg.attn_mask, encoder_attention_mask], dim=0)
            a_enc_hs = torch.cat([neg_audio.embeds, audio_encoder_hidden_states], dim=0)
            a_enc_mask = torch.cat([neg_audio.attn_mask, audio_encoder_attention_mask], dim=0)

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
    ) -> None:
        self.bundle = bundle
        self.step_kernel = LTX2DiffusionStep()
        self.strategy = strategy
        self.autocast_dtype = parse_torch_dtype(autocast_precision, field_name="autocast_precision")
        self.trajectory_dtype = parse_torch_dtype(trajectory_precision, field_name="trajectory_precision")
        self.logprob_dtype = parse_torch_dtype(logprob_precision, field_name="logprob_precision")

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
        sde_indices: Optional[List[int]] = None,
    ) -> LatentSegment:
        """Run the full denoising loop, collecting trajectory for RL.

        Args:
            conditions: Text/image conditioning.
            params: Sampling parameters (guidance_scale, eta, etc.).
            sigmas: Sigma schedule (T+1,) from high → 0.
            initial_latents: Starting noise (B, seq, C) or (B, C, T, H, W).
            sde_indices: Which steps to use SDE (stochastic) for RL.

        Returns:
            LatentSegment with trajectory and log-probs at SDE steps.
        """
        guidance_scale = float(params.guidance_scale)
        eta = float(params.eta)
        latent_t, latent_h, latent_w = self._latent_geometry(params)
        audio_t = _audio_num_frames(int(params.num_frames), _LTX2_FRAME_RATE)

        device = initial_latents.device
        num_steps = len(sigmas) - 1
        sigmas = sigmas.to(device)
        self.strategy.init_schedule(sigmas)

        # SDE step set: which steps record log-probs (default: all).
        sde_set: Set[int] = set(int(i) for i in sde_indices) if sde_indices else set(range(num_steps))
        sde_sorted: List[int] = sorted(sde_set)

        # Sparse trajectory storage: SDE transition endpoints (k, k+1) plus the
        # final step T so VAE decode always has the clean latent. Stored as a
        # (position, latent) list → packed into LatentSegment.{latents,indices},
        # which ``latents_at`` / ``replay`` index by step. Mirrors WAN21. The
        # audio trajectory is stored in parallel (aux_latents) so replay can
        # reproduce the per-step audio that the video forward cross-attends to.
        needed: Set[int] = set(compute_trajectory_positions(sde_set, num_steps))
        needed.add(num_steps)

        x = initial_latents.to(dtype=self.trajectory_dtype)
        # Audio latent stream: fresh N(0,1) noise, packed shape (B, audio_t, 128).
        # ODE-denoised in lockstep with video (no RL gradient). diffusers seeds
        # this from randn too (prepare_audio_latents); we don't need byte-exact
        # cross-engine audio since it's an internal conditioning signal.
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

                # Video: SDE (RL) step. strategy.denoise →
                # (prev_sample, log_prob, prev_sample_mean); log_prob None on ODE.
                x_next, log_prob, _ = self.strategy.denoise(
                    noise_pred=video_pred,
                    sample=x,
                    sigma=sigma,
                    sigma_next=sigma_next,
                    eta=step_eta,
                    sigma_max=sigma_max,
                    step_index=step_idx,
                )
                x = x_next.to(dtype=self.trajectory_dtype)

                # Audio: ODE step (eta=0) — no RL gradient, just track the state
                # the video branch conditions on. eta=0 → euler velocity step.
                a_next, _, _ = self.strategy.denoise(
                    noise_pred=audio_pred,
                    sample=a,
                    sigma=sigma,
                    sigma_next=sigma_next,
                    eta=0.0,
                    sigma_max=sigma_max,
                    step_index=step_idx,
                )
                a = a_next.to(dtype=self.trajectory_dtype)

                if (step_idx + 1) in needed:
                    stored_pairs.append((step_idx + 1, x.detach().clone()))
                    stored_audio.append(a.detach().clone())
                if log_prob is not None:
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
                # Feed the model/strategy the SAME dtype generate() used
                # (trajectory_dtype, the dtype the rollout latents were actually
                # stored in). Upcasting to autocast_dtype here would make the
                # replay model input differ from rollout (fp16 vs bf16 residual
                # stream) → step-0 importance ratio != 1 and a biased FlowGRPO
                # gradient. Matches WAN21 (which never re-casts at the replay
                # call site). autocast still runs the matmuls in autocast_dtype.
                sample = segment.latents_at(step_idx).to(device=device, dtype=self.trajectory_dtype)
                prev_sample = segment.latents_at(step_idx + 1).to(device=device, dtype=self.trajectory_dtype)
                # Reuse the audio state stored at this step from the rollout, so
                # the video prediction matches what generate() produced (the
                # video forward cross-attends to audio). Only the video pred is
                # used for the log-prob; we discard the audio pred.
                audio_sample = segment.aux_latents_at(step_idx).to(device=device, dtype=self.trajectory_dtype)

                video_pred, _ = self.step_kernel.predict_noise(
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
                log_probs.append(log_prob)
                if prev_mean is not None:
                    prev_sample_means.append(prev_mean)

        log_probs_t = torch.stack(log_probs, dim=1).to(dtype=self.logprob_dtype)
        means_t = torch.stack(prev_sample_means, dim=1).to(dtype=self.trajectory_dtype) if prev_sample_means else None
        return ReplayResult(log_probs=log_probs_t, prev_sample_means=means_t)


__all__ = ["LTX2DiffusionStep", "LTX2DiffusionStage"]
