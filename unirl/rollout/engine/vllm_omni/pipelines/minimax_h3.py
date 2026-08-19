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

from unirl.sde.kernels import CPSSDEStrategy
from unirl.sde.noise import make_denoise_step_generators
from unirl.types.noise_recipe import NoiseRecipe
from unirl.types.sampling import compute_trajectory_positions


class MiniMaxH3RLPipeline(MiniMaxH3Pipeline):
    """Upstream H3 execution plus deterministic CPS-SDE and sparse replay state."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # vLLM-Omni's custom-pipeline loader constructs the class directly,
        # bypassing registry.initialize_model(), which normally installs _sp_plan
        # hooks. Install them here before HSDP wraps the transformer; otherwise
        # Ulysses all-to-all receives a full sequence from every rank.
        _apply_sequence_parallel_if_enabled(self, self.od_config)
        self._rl_strategy = CPSSDEStrategy()
        self._rl_recipe: Optional[NoiseRecipe] = None
        self._rl_eta = 0.0
        self._rl_sde_indices: list[int] = []
        self._rl_video_states: list[torch.Tensor] = []
        self._rl_audio_states: list[torch.Tensor] = []
        self._rl_trajectory_indices: list[int] = []
        self._rl_video_sigmas: Optional[torch.Tensor] = None
        self._rl_audio_sigmas: Optional[torch.Tensor] = None
        self._rl_text_embeddings: Optional[torch.Tensor] = None

    @staticmethod
    def _recipe_for_request(request: Any) -> NoiseRecipe:
        sampling = request.sampling_params
        extra = getattr(sampling, "extra_args", None) or {}
        gids = list(extra.get("init_noise_group_ids") or [])
        rid = str(getattr(request, "request_id", "") or "")
        try:
            request_index = int(rid.split("_", 1)[0])
        except ValueError as exc:
            raise RuntimeError(f"MiniMax-H3 cannot parse request index from request_id={rid!r}") from exc
        outputs_per_prompt = int(getattr(sampling, "num_outputs_per_prompt", 1) or 1)
        start = request_index * outputs_per_prompt
        end = start + outputs_per_prompt
        if gids and (start < 0 or end > len(gids)):
            raise IndexError(f"MiniMax-H3 x_T recipe slice [{start}:{end}) exceeds {len(gids)} group ids")
        return NoiseRecipe(
            noise_group_ids=[str(gid) for gid in gids[start:end]],
            base_seed=int(extra.get("init_noise_seed", 0)),
            latent_shape=tuple(extra["init_noise_latent_shape"]) if extra.get("init_noise_latent_shape") else None,
        )

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
            video_rows[positive.update_mask_dev].detach().to(device="cpu", dtype=torch.float16)
        )
        self._rl_audio_states.append(
            audio_rows[positive.audio_update_mask_dev].detach().to(device="cpu", dtype=torch.float16)
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
    ) -> torch.Tensor:
        updated, _, _ = self._rl_strategy.denoise(
            noise_pred=(-velocity).unsqueeze(0),
            sample=sample.unsqueeze(0),
            sigma=torch.tensor(float(sigma), device=sample.device),
            sigma_next=torch.tensor(float(sigma_next), device=sample.device),
            eta=float(eta),
            generator=generator,
            step_index=int(step_index),
        )
        return updated[0]

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
        self._rl_video_sigmas = torch.as_tensor(sigmas_video, dtype=torch.float32)
        self._rl_audio_sigmas = torch.as_tensor(sigmas_audio, dtype=torch.float32)
        self._capture_state(
            0,
            needed=needed,
            video_rows=video_rows,
            audio_rows=audio_rows,
            positive=positive,
        )

        sample_id = self._rl_recipe.noise_group_ids[0] if self._rl_recipe and self._rl_recipe.noise_group_ids else "0"

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
                    video_rows[update] = self._cps_step(
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
                    audio_rows = audio_rows.clone()
                    audio_rows[audio_update] = self._cps_step(
                        audio_rows[audio_update],
                        audio_velocity,
                        sigma=audio_sigma,
                        sigma_next=float(sigmas_audio[step_index + 1]),
                        eta=step_eta,
                        generator=make_denoise_step_generators(
                            base_seed=int(self._rl_recipe.base_seed if self._rl_recipe else 0),
                            step_index=step_index,
                            sample_ids=[f"{sample_id}::audio"],
                        ),
                        step_index=step_index,
                    )
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
        if len(request.prompts) != 1:
            raise ValueError("MiniMaxH3RLPipeline requires one prompt per worker request")
        sampling = request.sampling_params
        extra = getattr(sampling, "extra_args", None) or {}
        self._rl_recipe = self._recipe_for_request(request)
        self._rl_eta = float(getattr(sampling, "eta", 0.0) or 0.0)
        self._rl_sde_indices = [int(index) for index in extra.get("sde_indices", [])]
        self._rl_text_embeddings = None

        original_loop = h3_module.minimax_h3_denoise_loop
        h3_module.minimax_h3_denoise_loop = self._rl_denoise_loop
        try:
            output = super().forward(request)
            if not self._rl_video_states or self._rl_text_embeddings is None:
                raise RuntimeError("MiniMax-H3 worker produced no replay trajectory or text embeddings")

            video, audio = output.output
            middle = int(video.shape[2]) // 2
            reward_frame = video[:, :, middle : middle + 1].detach().to("cpu")
            # Current vLLM-Omni serializes only declared DiffusionOutput fields.
            # Its trajectory fields explicitly accept dictionaries, so keep the
            # joint H3 replay payload there rather than on a runtime-only
            # ``custom_output`` attribute that gets dropped at the stage wire.
            output.trajectory_latents = {
                "video": torch.stack(self._rl_video_states, dim=0).unsqueeze(0),
                "audio": torch.stack(self._rl_audio_states, dim=0).unsqueeze(0),
                "indices": torch.as_tensor(self._rl_trajectory_indices, dtype=torch.long),
                "text_embeddings": self._rl_text_embeddings.unsqueeze(0),
                "reward_frame": reward_frame.clamp(0, 1).mul(255).round().to(torch.uint8),
            }
            output.trajectory_timesteps = {
                "video": self._rl_video_sigmas,
                "audio": self._rl_audio_sigmas,
                "sde_indices": torch.as_tensor(self._rl_sde_indices, dtype=torch.long),
            }
            output.trajectory_log_probs = None
            # Avoid sending the full decoded 124-frame video/audio through Omni IPC.
            output.output = (reward_frame, audio[..., :1].detach().to("cpu"))
            return output
        finally:
            h3_module.minimax_h3_denoise_loop = original_loop
            self._rl_recipe = None


__all__ = ["MiniMaxH3RLPipeline"]
