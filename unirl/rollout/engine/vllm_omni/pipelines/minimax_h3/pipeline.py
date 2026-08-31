"""MiniMax-H3 RL worker pipeline with joint video/audio trajectory capture."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Optional

import torch
from vllm_omni.diffusion.forward_context import set_forward_context_denoise_step_idx
from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as h3_module
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
    MiniMaxH3Pipeline,
    minimax_h3_patchify_video_latent,
)
from vllm_omni.diffusion.registry import _apply_sequence_parallel_if_enabled

from unirl.rollout.engine.vllm_omni.pipelines._shared.interception import (
    set_payload,
    stamp_capture,
)
from unirl.rollout.engine.vllm_omni.pipelines.minimax_h3.profile import RolloutProfile
from unirl.sde.kernels import CPSSDEStrategy
from unirl.sde.noise import make_denoise_step_generators
from unirl.types.noise_recipe import NoiseRecipe
from unirl.types.sampling import compute_trajectory_positions

#: Capture key carrying the whole H3 replay bundle; :mod:`adapters.minimax_h3` is the only reader.
CAPTURE_KEY = "minimax_h3_capture"


class MiniMaxH3RLPipeline(MiniMaxH3Pipeline):
    """Upstream H3 execution plus deterministic CPS-SDE and sparse replay state."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Memory history must start before super().__init__ loads weights so the snapshot
        # can attribute the resident weight allocations.
        self._h3_profile = RolloutProfile.from_env()
        if self._h3_profile is not None:
            self._h3_profile.enable_memory_history()
        super().__init__(*args, **kwargs)
        # vLLM-Omni's custom-pipeline loader constructs the class directly, bypassing
        # registry.initialize_model(), which normally installs the _sp_plan hooks. Install
        # them before HSDP wraps the transformer, or Ulysses all-to-all receives a full
        # sequence from every rank.
        _apply_sequence_parallel_if_enabled(self, self.od_config)
        # The custom-pipeline loader also bypasses registry VAE setup.
        vae_parallel_size = int(self.od_config.parallel_config.vae_patch_parallel_size)
        self.video_vae.use_slicing = bool(self.od_config.vae_use_slicing)
        self.video_vae.use_tiling = bool(self.od_config.vae_use_tiling or vae_parallel_size > 1)
        self.video_vae.set_parallel_size(
            vae_parallel_size,
            mode=self.od_config.parallel_config.vae_parallel_mode,
        )
        self._rl_strategy = CPSSDEStrategy()
        self._rl_recipe: Optional[NoiseRecipe] = None
        self._rl_sde_sample_key = "0"
        self._rl_eta = 0.0
        self._rl_sde_indices: list[int] = []
        self._rl_video_states: list[torch.Tensor] = []
        self._rl_audio_states: list[torch.Tensor] = []
        self._rl_trajectory_indices: list[int] = []
        self._rl_video_sigmas: Optional[torch.Tensor] = None
        self._rl_audio_sigmas: Optional[torch.Tensor] = None
        self._rl_text_embeddings: Optional[torch.Tensor] = None
        self._rl_log_probs: list[torch.Tensor] = []
        self._rl_video_means: list[torch.Tensor] = []
        self._rl_capture_transition_means = False
        self._rl_audio_joint_sde = True

    @staticmethod
    def _request_span(request: Any) -> tuple[dict[str, Any], int, int]:
        """This request's ``[start, end)`` slice of the driver's grouped batch."""
        sampling = request.sampling_params
        extra = getattr(sampling, "extra_args", None) or {}
        rid = str(getattr(request, "request_id", "") or "")
        try:
            request_index = int(rid.split("_", 1)[0])
        except ValueError as exc:
            raise RuntimeError(f"MiniMax-H3 cannot parse request index from request_id={rid!r}") from exc
        outputs_per_prompt = int(getattr(sampling, "num_outputs_per_prompt", 1) or 1)
        start = request_index * outputs_per_prompt
        end = start + outputs_per_prompt
        return extra, start, end

    @classmethod
    def _recipe_for_request(cls, request: Any) -> NoiseRecipe:
        """The driver-authored x_T recipe for this request's slice."""
        extra, start, end = cls._request_span(request)
        gids = list(extra.get("init_noise_group_ids") or [])
        if gids and (start < 0 or end > len(gids)):
            raise IndexError(f"MiniMax-H3 x_T recipe slice [{start}:{end}) exceeds {len(gids)} group ids")
        return NoiseRecipe(
            noise_group_ids=[str(gid) for gid in gids[start:end]],
            base_seed=int(extra.get("init_noise_seed", 0)),
            latent_shape=tuple(extra["init_noise_latent_shape"]) if extra.get("init_noise_latent_shape") else None,
        )

    @classmethod
    def _sde_sample_key_for_request(cls, request: Any) -> str:
        """The stable per-sibling identity that seeds this request's exploration noise."""
        extra, start, end = cls._request_span(request)
        sample_ids = list(extra.get("sde_sample_ids") or [])
        if start < 0 or end > len(sample_ids):
            raise RuntimeError(
                "MiniMax-H3 training request is missing stable per-sibling SDE identities: "
                f"slice=[{start}:{end}) available={len(sample_ids)}"
            )
        selected = [str(sample_id) for sample_id in sample_ids[start:end]]
        if len(selected) != 1 or not selected[0]:
            raise RuntimeError(f"MiniMax-H3 requires exactly one non-empty SDE sample id, got {selected!r}")
        return selected[0]

    def _initial_noise(
        self,
        *,
        seed: int,
        latent_t: int,
        latent_h: int,
        latent_w: int,
        audio_t: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Replace the engine's x_T with the driver's, so trainer and rollout share it."""
        recipe = self._rl_recipe
        if recipe is None or not recipe.noise_group_ids:
            return super()._initial_noise(
                seed=seed,
                latent_t=latent_t,
                latent_h=latent_h,
                latent_w=latent_w,
                audio_t=audio_t,
            )
        video = recipe.resolve(
            latent_shape=(24, latent_t, latent_h, latent_w),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        audio = recipe.resolve(
            salt="audio",
            latent_shape=(audio_t * 2, 32),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        if video is None or audio is None or int(video.shape[0]) != 1 or int(audio.shape[0]) != 1:
            raise RuntimeError("MiniMax-H3 requires exactly one driver-authored x_T per worker request")
        return (
            minimax_h3_patchify_video_latent(video, patch_size=(1, 2, 2)),
            audio[0],
        )

    def encode_prompt(self, *args: Any, **kwargs: Any) -> Any:
        """Tap the text embeddings the trainside replay needs to re-run a step."""
        result = super().encode_prompt(*args, **kwargs)
        self._rl_text_embeddings = result[0].detach().to("cpu")
        return result

    def _capture_state(
        self,
        index: int,
        *,
        needed: set[int],
        video_rows: torch.Tensor,
        audio_rows: torch.Tensor,
        positive: Any,
    ) -> None:
        """Record the denoised rows at ``index`` when the replay needs that position."""
        if index not in needed:
            return
        self._rl_trajectory_indices.append(int(index))
        self._rl_video_states.append(
            video_rows[positive.update_mask_dev].detach().to(device="cpu", dtype=torch.float32)
        )
        self._rl_audio_states.append(
            audio_rows[positive.audio_update_mask_dev].detach().to(device="cpu", dtype=torch.float32)
        )

    def _cps_step(
        self,
        sample: torch.Tensor,
        velocity: torch.Tensor,
        *,
        sigma: float,
        sigma_next: float,
        eta: float,
        generator: list[torch.Generator],
        step_index: int,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """One CPS transition, returning the new sample plus its log-prob and mean."""
        updated, log_prob, mean = self._rl_strategy.denoise(
            noise_pred=(-velocity).unsqueeze(0),
            sample=sample.unsqueeze(0),
            sigma=torch.tensor(float(sigma), device=sample.device),
            sigma_next=torch.tensor(float(sigma_next), device=sample.device),
            eta=float(eta),
            generator=generator,
            step_index=int(step_index),
        )
        return updated[0], log_prob, (mean[0] if mean is not None else None)

    def _rl_denoise_loop(
        self,
        *,
        model: Any,
        positive: Any,
        initial_video_rows: torch.Tensor,
        initial_audio_rows: torch.Tensor,
        keyframe_cond_rows: Optional[torch.Tensor],
        audio_ref_rows: Optional[torch.Tensor] = None,
        sigmas_video: list[float],
        sigmas_audio: list[float],
        device: torch.device,
        imgvid_cond_noise_aug_for_inference: float = 0.999,
        audio_cond_noise_aug_for_inference: float = 1.0,
        on_step_end: Any = None,
        on_step_start: Any = None,
        step_profiler: Any = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Replaces the upstream loop: CPS transitions on the selected steps, plus replay capture."""
        if len(sigmas_video) != len(sigmas_audio) or len(sigmas_video) < 2:
            raise ValueError("MiniMax-H3 video/audio sigma schedules must have equal positive length")
        num_steps = len(sigmas_video) - 1

        update = positive.update_mask_dev
        audio_update = positive.audio_update_mask_dev
        video_rows = initial_video_rows.to(device=device, dtype=torch.float32).clone()
        audio_rows = initial_audio_rows.to(device=device, dtype=torch.float32).clone()
        if int(video_rows.shape[0]) != int(positive.img_pos.shape[0]):
            raise ValueError(
                f"MiniMax-H3 initial video rows {int(video_rows.shape[0])} "
                f"!= packed video rows {int(positive.img_pos.shape[0])}"
            )
        if int(audio_rows.shape[0]) != int(positive.audio_pos.shape[0]):
            raise ValueError(
                f"MiniMax-H3 initial audio rows {int(audio_rows.shape[0])} "
                f"!= packed audio rows {int(positive.audio_pos.shape[0])}"
            )

        n_video_cond = int((~positive.update_mask).sum())
        if keyframe_cond_rows is None:
            if n_video_cond:
                raise ValueError(f"MiniMax-H3 layout has {n_video_cond} video condition rows but no anchor")
            video_anchor = None
        else:
            if int(keyframe_cond_rows.shape[0]) != n_video_cond:
                raise ValueError(
                    f"MiniMax-H3 keyframe rows {int(keyframe_cond_rows.shape[0])} "
                    f"!= layout condition rows {n_video_cond}"
                )
            video_anchor = keyframe_cond_rows.to(device=device, dtype=torch.float32)
            video_rows[~update] = video_anchor

        n_audio_cond = int((~positive.audio_update_mask).sum())
        if audio_ref_rows is None:
            if n_audio_cond:
                raise ValueError(f"MiniMax-H3 layout has {n_audio_cond} audio condition rows but no anchor")
            audio_anchor = None
        else:
            if int(audio_ref_rows.shape[0]) != n_audio_cond:
                raise ValueError(
                    f"MiniMax-H3 audio reference rows {int(audio_ref_rows.shape[0])} "
                    f"!= layout condition rows {n_audio_cond}"
                )
            audio_anchor = audio_ref_rows.to(device=device, dtype=torch.float32)
            audio_rows[~audio_update] = audio_anchor

        sde_indices = sorted({int(index) for index in self._rl_sde_indices})
        if any(index < 0 or index >= num_steps for index in sde_indices):
            raise ValueError(f"MiniMax-H3 sde_indices out of range for {num_steps} steps: {sde_indices}")
        needed = set(compute_trajectory_positions(set(sde_indices), num_steps))
        self._rl_video_states = []
        self._rl_audio_states = []
        self._rl_trajectory_indices = []
        self._rl_log_probs = []
        self._rl_video_means = []
        self._rl_video_sigmas = torch.as_tensor(sigmas_video, dtype=torch.float32)
        self._rl_audio_sigmas = torch.as_tensor(sigmas_audio, dtype=torch.float32)
        self._capture_state(
            0,
            needed=needed,
            video_rows=video_rows,
            audio_rows=audio_rows,
            positive=positive,
        )

        sample_id = self._rl_sde_sample_key

        try:
            for step_index in range(num_steps):
                context = step_profiler(step_index) if step_profiler is not None else nullcontext()
                with context:
                    set_forward_context_denoise_step_idx(step_index)
                    video_sigma = float(sigmas_video[step_index])
                    audio_sigma = float(sigmas_audio[step_index])
                    if on_step_start is not None:
                        on_step_start(step_index, video_sigma, audio_sigma)

                    video_t = 1.0 - video_sigma
                    audio_t = 1.0 - audio_sigma
                    forward_kwargs = positive.forward_kwargs(
                        video_rows=video_rows,
                        audio_rows=audio_rows,
                        t_video=video_t,
                        t_audio=audio_t,
                        imgvid_cond_timestep=max(video_t, float(imgvid_cond_noise_aug_for_inference)),
                        audio_ref_cond_timestep=max(audio_t, float(audio_cond_noise_aug_for_inference)),
                    )
                    with torch.inference_mode():
                        video_velocity, audio_velocity = model(**forward_kwargs)
                    video_velocity = video_velocity.float()[update]
                    audio_velocity = audio_velocity.float()[audio_update]

                    step_eta = self._rl_eta if step_index in sde_indices else 0.0
                    video_rows = video_rows.clone()
                    next_video, video_log_prob, video_mean = self._cps_step(
                        video_rows[update],
                        video_velocity,
                        sigma=video_sigma,
                        sigma_next=float(sigmas_video[step_index + 1]),
                        eta=step_eta,
                        generator=make_denoise_step_generators(
                            base_seed=int(self._rl_recipe.base_seed if self._rl_recipe else 0),
                            step_index=step_index,
                            sample_ids=[sample_id],
                        ),
                        step_index=step_index,
                    )
                    video_rows[update] = next_video
                    audio_rows = audio_rows.clone()
                    audio_eta = step_eta if self._rl_audio_joint_sde else 0.0
                    next_audio, audio_log_prob, _ = self._cps_step(
                        audio_rows[audio_update],
                        audio_velocity,
                        sigma=audio_sigma,
                        sigma_next=float(sigmas_audio[step_index + 1]),
                        eta=audio_eta,
                        generator=make_denoise_step_generators(
                            base_seed=int(self._rl_recipe.base_seed if self._rl_recipe else 0),
                            step_index=step_index,
                            sample_ids=[f"{sample_id}::audio"],
                        ),
                        step_index=step_index,
                    )
                    audio_rows[audio_update] = next_audio
                    if step_index in sde_indices:
                        if video_log_prob is None:
                            raise RuntimeError(f"MiniMax-H3 CPS step {step_index} produced no video rollout log-prob")
                        if self._rl_audio_joint_sde and audio_log_prob is None:
                            raise RuntimeError(f"MiniMax-H3 CPS step {step_index} produced no audio rollout log-prob")
                        if self._rl_capture_transition_means:
                            if video_mean is None:
                                raise RuntimeError(
                                    f"MiniMax-H3 CPS step {step_index} produced no rollout transition mean"
                                )
                            self._rl_video_means.append(video_mean.detach().to(device="cpu", dtype=torch.float32))
                        policy_log_prob = video_log_prob
                        if self._rl_audio_joint_sde:
                            assert audio_log_prob is not None
                            # One joint transition, so the two modalities' log-probs combine
                            # weighted by element count rather than being averaged evenly.
                            video_numel = int(video_velocity.numel())
                            audio_numel = int(audio_velocity.numel())
                            policy_log_prob = (video_log_prob * video_numel + audio_log_prob * audio_numel) / (
                                video_numel + audio_numel
                            )
                        self._rl_log_probs.append(policy_log_prob.detach().to(device="cpu", dtype=torch.float32))
                    if video_anchor is not None:
                        video_rows[~update] = video_anchor
                    if audio_anchor is not None:
                        audio_rows[~audio_update] = audio_anchor
                    if on_step_end is not None:
                        on_step_end(step_index, video_rows, audio_rows)
                    self._capture_state(
                        step_index + 1,
                        needed=needed,
                        video_rows=video_rows,
                        audio_rows=audio_rows,
                        positive=positive,
                    )
        finally:
            set_forward_context_denoise_step_idx(None)
        return video_rows, audio_rows

    @torch.inference_mode()
    def forward(self, request: Any) -> Any:
        """Profiled wrapper around :meth:`_forward_impl`."""
        profile = self._h3_profile
        if profile is not None and profile.active():
            with profile.request_profiler():
                try:
                    return self._forward_impl(request)
                finally:
                    profile.finish_request()
        return self._forward_impl(request)

    def _arm(self, request: Any) -> dict[str, Any]:
        """Read this request's RL knobs onto ``self`` for the denoise loop to consume."""
        sampling = request.sampling_params
        extra = getattr(sampling, "extra_args", None) or {}
        self._rl_recipe = self._recipe_for_request(request)
        self._rl_eta = float(getattr(sampling, "eta", 0.0) or 0.0)
        self._rl_sde_indices = [int(index) for index in extra.get("sde_indices", [])]
        # x_T may be shared by group, but policy exploration stays unique per sibling and
        # reproducible across retries and resume.
        self._rl_sde_sample_key = self._sde_sample_key_for_request(request) if self._rl_sde_indices else "0"
        self._rl_audio_joint_sde = bool(extra.get("audio_joint_sde", True))
        self._rl_capture_transition_means = bool(extra.get("capture_transition_means", False))
        self._rl_text_embeddings = None
        return extra

    def _reward_media(self, video: torch.Tensor, audio: torch.Tensor, extra: dict[str, Any]) -> dict[str, Any]:
        """Subsample the decode down to what the reward models actually score."""
        reward_num_frames = min(
            max(1, int(extra.get("reward_num_frames", 9))),
            int(video.shape[2]),
        )
        reward_indices = (
            torch.linspace(0, int(video.shape[2]) - 1, steps=reward_num_frames, device=video.device).round().long()
        )
        reward_video = video.index_select(2, reward_indices).detach().to("cpu")
        reward_audio = audio[0].transpose(0, 1).contiguous().detach().to(device="cpu", dtype=torch.float32)
        return {
            "reward_video": reward_video.clamp(0, 1).mul(255).round().to(torch.uint8),
            "reward_audio": reward_audio,
        }

    def _replay_bundle(self, extra: dict[str, Any]) -> dict[str, Any]:
        """The sparse trajectory the trainside forward replays, empty when no step was stochastic."""
        del extra
        if not self._rl_sde_indices:
            return {}
        if not self._rl_video_states or self._rl_text_embeddings is None:
            raise RuntimeError("MiniMax-H3 worker produced no replay trajectory or text embeddings")
        bundle: dict[str, Any] = {
            "video": torch.stack(self._rl_video_states, dim=0).unsqueeze(0),
            "audio": torch.stack(self._rl_audio_states, dim=0).unsqueeze(0),
            "indices": torch.as_tensor(self._rl_trajectory_indices, dtype=torch.long),
            "text_embeddings": self._rl_text_embeddings.unsqueeze(0),
            "video_sigmas": self._rl_video_sigmas,
            "audio_sigmas": self._rl_audio_sigmas,
            "sde_indices": torch.as_tensor(self._rl_sde_indices, dtype=torch.long),
        }
        if self._rl_capture_transition_means:
            if len(self._rl_video_means) != len(self._rl_sde_indices):
                raise RuntimeError(
                    "MiniMax-H3 worker transition-mean count mismatch: "
                    f"means={len(self._rl_video_means)} sde_indices={len(self._rl_sde_indices)}"
                )
            bundle["video_means"] = torch.stack(self._rl_video_means, dim=0).unsqueeze(0)
        if self._rl_log_probs:
            bundle["log_probs"] = torch.stack(self._rl_log_probs, dim=1)
        return bundle

    def _forward_impl(self, request: Any) -> Any:
        if len(request.prompts) != 1:
            raise ValueError("MiniMaxH3RLPipeline requires one prompt per worker request")
        extra = self._arm(request)

        original_loop = h3_module.minimax_h3_denoise_loop
        h3_module.minimax_h3_denoise_loop = self._rl_denoise_loop
        try:
            output = super().forward(request)
            video, audio = output.output
            capture = self._reward_media(video, audio, extra)
            capture.update(self._replay_bundle(extra))
            stamp_capture(output, CAPTURE_KEY, capture)
            # Keep the wire payload bounded — the reward media rides the capture bag — while
            # leaving postprocess the (video, audio) pair it formats.
            set_payload(output, (capture["reward_video"], audio[..., :1].detach().to("cpu")))
            return output
        finally:
            h3_module.minimax_h3_denoise_loop = original_loop
            self._rl_recipe = None


__all__ = ["CAPTURE_KEY", "MiniMaxH3RLPipeline"]
