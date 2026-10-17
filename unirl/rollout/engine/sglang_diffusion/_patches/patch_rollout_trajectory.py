"""Per-sample rollout-trajectory preservation across the grouped forward."""

from __future__ import annotations

import torch

_MERGE_SENTINEL = "_unirl_rtd_concat"
_RESULT_SENTINEL = "_unirl_rtd_slice"


def _require_uniform_presence(values: list, *, field: str) -> list | None:
    """Return present values, rejecting partial optional-field population."""
    if not values or all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(f"Grouped rollout field {field!r} is present for only some outputs")
    return values


def _cat0(values: list, *, field: str) -> object:
    """Concatenate aligned batch-dim-0 tensors while preserving optional absence."""
    present = _require_uniform_presence(values, field=field)
    if present is None:
        return None
    if not all(isinstance(value, torch.Tensor) for value in present):
        raise TypeError(f"Grouped rollout field {field!r} must contain only tensors")
    if any(value.dim() < 1 for value in present):
        raise ValueError(f"Grouped rollout field {field!r} must have a batch dimension")
    return torch.cat(present, dim=0)


def _shared_tensor(values: list, *, field: str) -> object:
    """Return shared schedule metadata after verifying every output agrees."""
    present = _require_uniform_presence(values, field=field)
    if present is None:
        return None
    if not all(isinstance(value, torch.Tensor) for value in present):
        raise TypeError(f"Grouped rollout field {field!r} must contain only tensors")
    first = present[0]
    if any(not torch.equal(first, value) for value in present[1:]):
        raise ValueError(f"Grouped rollout field {field!r} differs across outputs")
    return first


def _slice_row_keepdim(t, idx: int):
    """Return one grouped-output row, preserving its batch dimension."""
    if t is None:
        return None
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"Grouped rollout value must be a tensor or None; got {type(t).__name__}")
    if t.dim() < 1:
        raise ValueError("Grouped rollout tensor must have a batch dimension")
    if not 0 <= idx < t.shape[0]:
        raise IndexError(f"Grouped rollout row {idx} is outside batch size {t.shape[0]}")
    return t[idx : idx + 1].contiguous()


def patch_rollout_trajectory() -> None:
    """Concat per-output trajectories in the merge + slice them per output result."""
    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
        DiffGenerator,
    )
    from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker
    from sglang.multimodal_gen.runtime.post_training.rl_dataclasses import (
        RolloutDebugTensors,
        RolloutDitTrajectory,
        RolloutTrajectoryData,
    )

    raw_merge = GPUWorker._merge_expanded_output_batches
    if not getattr(raw_merge, _MERGE_SENTINEL, False):

        def _merge_expanded_output_batches(output_batches):
            merged = raw_merge(output_batches)
            rtds = _require_uniform_presence(
                [ob.rollout_trajectory_data for ob in output_batches],
                field="rollout_trajectory_data",
            )
            if rtds is None:
                return merged
            if len(rtds) == 1:
                merged.rollout_trajectory_data = rtds[0]
                return merged

            new_dit = None
            dit_trajectories = _require_uniform_presence(
                [rtd.dit_trajectory for rtd in rtds],
                field="dit_trajectory",
            )
            if dit_trajectories is not None:
                new_dit = RolloutDitTrajectory(
                    latents=_cat0(
                        [trajectory.latents for trajectory in dit_trajectories],
                        field="dit_trajectory.latents",
                    ),
                    timesteps=_shared_tensor(
                        [trajectory.timesteps for trajectory in dit_trajectories],
                        field="dit_trajectory.timesteps",
                    ),
                    sigmas=_shared_tensor(
                        [trajectory.sigmas for trajectory in dit_trajectories],
                        field="dit_trajectory.sigmas",
                    ),
                )
                audio_latents = _cat0(
                    [getattr(trajectory, "audio_latents", None) for trajectory in dit_trajectories],
                    field="dit_trajectory.audio_latents",
                )
                if audio_latents is not None:
                    new_dit.audio_latents = audio_latents

            new_debug = None
            debug_tensors = _require_uniform_presence(
                [rtd.rollout_debug_tensors for rtd in rtds],
                field="rollout_debug_tensors",
            )
            if debug_tensors is not None:

                def _dbg(field: str):
                    return _cat0(
                        [getattr(debug, field) for debug in debug_tensors],
                        field=f"rollout_debug_tensors.{field}",
                    )

                new_debug = RolloutDebugTensors(
                    rollout_variance_noises=_dbg("rollout_variance_noises"),
                    rollout_prev_sample_means=_dbg("rollout_prev_sample_means"),
                    rollout_noise_std_devs=_dbg("rollout_noise_std_devs"),
                    rollout_model_outputs=_dbg("rollout_model_outputs"),
                )

            denoising_envs = _require_uniform_presence(
                [rtd.denoising_env for rtd in rtds],
                field="denoising_env",
            )
            if denoising_envs is not None:
                raise ValueError(
                    "Grouped rollout denoising_env is unsupported because its nested "
                    "tensors do not carry explicit batch-axis metadata"
                )

            merged.rollout_trajectory_data = RolloutTrajectoryData(
                rollout_log_probs=_cat0(
                    [rtd.rollout_log_probs for rtd in rtds],
                    field="rollout_log_probs",
                ),
                rollout_debug_tensors=new_debug,
                denoising_env=None,
                dit_trajectory=new_dit,
            )
            return merged

        setattr(_merge_expanded_output_batches, _MERGE_SENTINEL, True)
        GPUWorker._merge_expanded_output_batches = staticmethod(_merge_expanded_output_batches)

    raw_result_common = DiffGenerator._result_common
    if not getattr(raw_result_common, _RESULT_SENTINEL, False):

        def _result_common(req, output_batch, generation_time, output_index=None):
            d = raw_result_common(req, output_batch, generation_time, output_index)
            if output_index is not None and isinstance(d, dict):
                rtd = d.get("rollout_trajectory_data")
                if rtd is not None:
                    if rtd.denoising_env is not None:
                        raise ValueError("Per-output slicing of grouped denoising_env is unsupported")

                    new_dit = None
                    if rtd.dit_trajectory is not None:
                        new_dit = RolloutDitTrajectory(
                            latents=_slice_row_keepdim(rtd.dit_trajectory.latents, output_index),
                            timesteps=rtd.dit_trajectory.timesteps,
                            sigmas=rtd.dit_trajectory.sigmas,
                        )
                        _aud = getattr(rtd.dit_trajectory, "audio_latents", None)
                        if _aud is not None:
                            new_dit.audio_latents = _slice_row_keepdim(_aud, output_index)

                    new_debug = None
                    if rtd.rollout_debug_tensors is not None:
                        dbg = rtd.rollout_debug_tensors
                        new_debug = RolloutDebugTensors(
                            rollout_variance_noises=_slice_row_keepdim(dbg.rollout_variance_noises, output_index),
                            rollout_prev_sample_means=_slice_row_keepdim(dbg.rollout_prev_sample_means, output_index),
                            rollout_noise_std_devs=_slice_row_keepdim(dbg.rollout_noise_std_devs, output_index),
                            rollout_model_outputs=_slice_row_keepdim(dbg.rollout_model_outputs, output_index),
                        )

                    d["rollout_trajectory_data"] = RolloutTrajectoryData(
                        rollout_log_probs=_slice_row_keepdim(rtd.rollout_log_probs, output_index),
                        rollout_debug_tensors=new_debug,
                        denoising_env=None,
                        dit_trajectory=new_dit,
                    )
            return d

        setattr(_result_common, _RESULT_SENTINEL, True)
        DiffGenerator._result_common = staticmethod(_result_common)


__all__ = ["patch_rollout_trajectory"]
