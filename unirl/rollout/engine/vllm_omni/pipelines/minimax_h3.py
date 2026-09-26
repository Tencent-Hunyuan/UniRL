"""MiniMax-H3 RL worker pipeline with joint video/audio trajectory capture."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Optional

import torch
from vllm_omni.diffusion.models.minimax_h3 import pipeline_minimax_h3 as h3_module
from vllm_omni.diffusion.models.minimax_h3.denoise_loop import (
    minimax_h3_prepare_denoise_rows,
    minimax_h3_publish_denoise_progress,
)
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import (
    MiniMaxH3Pipeline,
    minimax_h3_patchify_video_latent,
)
from vllm_omni.diffusion.registry import _apply_sequence_parallel_if_enabled

from unirl.sde.kernels import CPSSDEStrategy
from unirl.sde.noise import make_denoise_step_generators
from unirl.types.noise_recipe import NoiseRecipe
from unirl.types.sampling import compute_trajectory_positions

# The serving DiT keeps SwiGLU gate/up fused in ``mlp.fc1``; vLLM-Omni 0.28 only
# names the fused QKV sublayers, so trainer LoRA on the FFN would otherwise bind
# to nothing (vllm-project/vllm-omni#6351 adds the same mapping upstream).
_FC1_SUBLAYER_MAPPING = (
    (".mlp.fc1", ".mlp.gate_proj", 0),
    (".mlp.fc1", ".mlp.up_proj", 1),
)


class MiniMaxH3RLPipeline(MiniMaxH3Pipeline):
    """Upstream H3 execution plus deterministic CPS-SDE and sparse replay state."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.od_config.step_execution:
            raise ValueError("MiniMaxH3RLPipeline captures trajectories in forward(); step_execution is unsupported")
        # vLLM-Omni's custom-pipeline loader constructs the class directly,
        # bypassing registry.initialize_model(), which normally installs _sp_plan
        # hooks. Install them here before HSDP wraps the transformer; otherwise
        # Ulysses all-to-all receives a full sequence from every rank.
        _apply_sequence_parallel_if_enabled(self, self.od_config)
        # The custom-pipeline loader also bypasses registry VAE setup.
        vae_parallel_size = int(self.od_config.parallel_config.vae_patch_parallel_size)
        self.video_vae.use_slicing = bool(self.od_config.vae_use_slicing)
        self.video_vae.use_tiling = bool(self.od_config.vae_use_tiling or vae_parallel_size > 1)
        self.video_vae.set_parallel_size(
            vae_parallel_size,
            mode=self.od_config.parallel_config.vae_parallel_mode,
        )
        # The diffusion LoRA manager derives packed sublayers from this
        # attribute when it is built, which happens after the pipeline loads.
        self.transformer.stacked_params_mapping = (
            *type(self.transformer).stacked_params_mapping,
            *_FC1_SUBLAYER_MAPPING,
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
        self._rl_audio_joint_sde = True

    def _validate_diffusion_lora_binding(self, *, lora_model: Any, bound_lora_names: frozenset[str]) -> None:
        """Fail when any trainer LoRA module was left unbound by the serving layout."""
        super()._validate_diffusion_lora_binding(lora_model=lora_model, bound_lora_names=bound_lora_names)
        missing = sorted(set(lora_model.loras) - bound_lora_names)
        if missing:
            raise RuntimeError(
                f"MiniMax-H3 LoRA binding is incomplete: bound={len(bound_lora_names)}/{len(lora_model.loras)}, "
                f"missing={missing[:5]}"
            )

    @staticmethod
    def _request_span(request: Any) -> tuple[dict[str, Any], int, int]:
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
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        updated, log_prob, _ = self._rl_strategy.denoise(
            noise_pred=(-velocity).unsqueeze(0),
            sample=sample.unsqueeze(0),
            sigma=torch.tensor(float(sigma), device=sample.device),
            sigma_next=torch.tensor(float(sigma_next), device=sample.device),
            eta=float(eta),
            generator=generator,
            step_index=int(step_index),
        )
        return updated[0], log_prob

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
        imgvid_cond_noise_aug_for_inference: float,
        audio_cond_noise_aug_for_inference: float,
        on_step: Any = None,
        step_profiler: Any = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Drop-in for ``minimax_h3_denoise_loop`` that swaps Euler for CPS-SDE and records replay state."""
        if len(sigmas_video) != len(sigmas_audio) or len(sigmas_video) < 2:
            raise ValueError("MiniMax-H3 video/audio sigma schedules must have equal length >= 2")
        num_steps = len(sigmas_video) - 1
        video_rows, audio_rows, video_anchor, audio_anchor = minimax_h3_prepare_denoise_rows(
            positive=positive,
            initial_video_rows=initial_video_rows,
            initial_audio_rows=initial_audio_rows,
            keyframe_cond_rows=keyframe_cond_rows,
            audio_ref_rows=audio_ref_rows,
            device=device,
        )
        update = positive.update_mask_dev
        audio_update = positive.audio_update_mask_dev

        sde_indices = sorted({int(index) for index in self._rl_sde_indices})
        if any(index < 0 or index >= num_steps for index in sde_indices):
            raise ValueError(f"MiniMax-H3 sde_indices out of range for {num_steps} steps: {sde_indices}")
        needed = set(compute_trajectory_positions(set(sde_indices), num_steps))
        self._rl_video_states = []
        self._rl_audio_states = []
        self._rl_trajectory_indices = []
        self._rl_log_probs = []
        self._rl_video_sigmas = torch.as_tensor(sigmas_video, dtype=torch.float32)
        self._rl_audio_sigmas = torch.as_tensor(sigmas_audio, dtype=torch.float32)
        self._capture_state(0, needed=needed, video_rows=video_rows, audio_rows=audio_rows, positive=positive)

        sample_id = self._rl_sde_sample_key
        base_seed = int(self._rl_recipe.base_seed if self._rl_recipe else 0)
        try:
            for step_index in range(num_steps):
                context = step_profiler(step_index) if step_profiler is not None else nullcontext()
                with context:
                    video_sigma = float(sigmas_video[step_index])
                    audio_sigma = float(sigmas_audio[step_index])
                    minimax_h3_publish_denoise_progress(step_index, video_sigma, num_steps)
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
                    next_video, video_log_prob = self._cps_step(
                        video_rows[update],
                        video_velocity,
                        sigma=video_sigma,
                        sigma_next=float(sigmas_video[step_index + 1]),
                        eta=step_eta,
                        generator=make_denoise_step_generators(
                            base_seed=base_seed, step_index=step_index, sample_ids=[sample_id]
                        ),
                        step_index=step_index,
                    )
                    video_rows[update] = next_video
                    audio_rows = audio_rows.clone()
                    next_audio, audio_log_prob = self._cps_step(
                        audio_rows[audio_update],
                        audio_velocity,
                        sigma=audio_sigma,
                        sigma_next=float(sigmas_audio[step_index + 1]),
                        eta=step_eta if self._rl_audio_joint_sde else 0.0,
                        generator=make_denoise_step_generators(
                            base_seed=base_seed, step_index=step_index, sample_ids=[f"{sample_id}::audio"]
                        ),
                        step_index=step_index,
                    )
                    audio_rows[audio_update] = next_audio
                    if step_index in sde_indices:
                        if video_log_prob is None:
                            raise RuntimeError(f"MiniMax-H3 CPS step {step_index} produced no video rollout log-prob")
                        policy_log_prob = video_log_prob
                        if self._rl_audio_joint_sde:
                            if audio_log_prob is None:
                                raise RuntimeError(
                                    f"MiniMax-H3 CPS step {step_index} produced no audio rollout log-prob"
                                )
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
                    if on_step is not None:
                        on_step(step_index, video_rows, audio_rows)
                    self._capture_state(
                        step_index + 1,
                        needed=needed,
                        video_rows=video_rows,
                        audio_rows=audio_rows,
                        positive=positive,
                    )
        finally:
            minimax_h3_publish_denoise_progress(None, None, None)
        return video_rows, audio_rows

    @torch.inference_mode()
    def forward(self, request: Any) -> Any:
        if len(request.prompts) != 1:
            raise ValueError("MiniMaxH3RLPipeline requires one prompt per worker request")
        sampling = request.sampling_params
        extra = getattr(sampling, "extra_args", None) or {}
        self._rl_recipe = self._recipe_for_request(request)
        self._rl_eta = float(getattr(sampling, "eta", 0.0) or 0.0)
        self._rl_sde_indices = [int(index) for index in extra.get("sde_indices", [])]
        # x_T may be shared by group, but policy exploration remains unique per
        # sibling and reproducible across retries/resume.
        self._rl_sde_sample_key = self._sde_sample_key_for_request(request) if self._rl_sde_indices else "0"
        self._rl_audio_joint_sde = bool(extra.get("audio_joint_sde", True))
        self._rl_text_embeddings = None

        original_loop = h3_module.minimax_h3_denoise_loop
        h3_module.minimax_h3_denoise_loop = self._rl_denoise_loop
        try:
            output = super().forward(request)
            video, audio = output.output
            reward_num_frames = min(max(1, int(extra.get("reward_num_frames", 9))), int(video.shape[2]))
            reward_indices = (
                torch.linspace(0, int(video.shape[2]) - 1, steps=reward_num_frames, device=video.device).round().long()
            )
            reward_video = video.index_select(2, reward_indices).detach().to("cpu")
            reward_audio = audio[0].transpose(0, 1).contiguous().detach().to(device="cpu", dtype=torch.float32)
            # vLLM-Omni serializes only declared DiffusionOutput fields; its
            # trajectory fields accept dictionaries, so the joint H3 replay
            # payload travels there rather than on an undeclared attribute.
            payload = {
                "reward_video": reward_video.clamp(0, 1).mul(255).round().to(torch.uint8),
                "reward_audio": reward_audio,
            }
            if self._rl_sde_indices:
                if not self._rl_video_states or self._rl_text_embeddings is None:
                    raise RuntimeError("MiniMax-H3 worker produced no replay trajectory or text embeddings")
                payload.update(
                    video=torch.stack(self._rl_video_states, dim=0).unsqueeze(0),
                    audio=torch.stack(self._rl_audio_states, dim=0).unsqueeze(0),
                    indices=torch.as_tensor(self._rl_trajectory_indices, dtype=torch.long),
                    text_embeddings=self._rl_text_embeddings.unsqueeze(0),
                )
                output.trajectory_timesteps = {
                    "video": self._rl_video_sigmas,
                    "audio": self._rl_audio_sigmas,
                    "sde_indices": torch.as_tensor(self._rl_sde_indices, dtype=torch.long),
                }
            else:
                output.trajectory_timesteps = None
            output.trajectory_latents = payload
            output.trajectory_log_probs = torch.stack(self._rl_log_probs, dim=1) if self._rl_log_probs else None
            # Keep the wire payload bounded while preserving temporal and audio
            # signals for multi-frame/video-audio reward models.
            output.output = (reward_video, audio[..., :1].detach().to("cpu"))
            return output
        finally:
            h3_module.minimax_h3_denoise_loop = original_loop
            self._rl_recipe = None


__all__ = ["MiniMaxH3RLPipeline"]
