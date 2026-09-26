"""Per-sample rollout-trajectory preservation across the grouped forward."""

from __future__ import annotations

import torch

_MERGE_SENTINEL = "_unirl_rtd_concat"
_RESULT_SENTINEL = "_unirl_rtd_slice"


def _rl_dataclasses():
    from sglang.multimodal_gen.runtime.post_training.rl_dataclasses import (
        RolloutDebugTensors,
        RolloutDitTrajectory,
        RolloutTrajectoryData,
    )

    return RolloutTrajectoryData, RolloutDitTrajectory, RolloutDebugTensors


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


def _concat_rollout_trajectory_data(output_batches: list):
    """Build ONE ``RolloutTrajectoryData`` concatenated across the per-output batches."""
    RolloutTrajectoryData, RolloutDitTrajectory, RolloutDebugTensors = _rl_dataclasses()

    rtds = _require_uniform_presence(
        [getattr(ob, "rollout_trajectory_data", None) for ob in output_batches],
        field="rollout_trajectory_data",
    )
    if rtds is None:
        return None
    if len(rtds) == 1:
        return rtds[0]

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

    return RolloutTrajectoryData(
        rollout_log_probs=_cat0(
            [rtd.rollout_log_probs for rtd in rtds],
            field="rollout_log_probs",
        ),
        rollout_debug_tensors=new_debug,
        denoising_env=None,
        dit_trajectory=new_dit,
    )


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


def _slice_rollout_trajectory_keepdim(rtd, idx: int):
    """Per-output slice of a concatenated ``[K, ...]`` trajectory, keep-dim."""
    if rtd is None:
        return None
    RolloutTrajectoryData, RolloutDitTrajectory, RolloutDebugTensors = _rl_dataclasses()
    if rtd.denoising_env is not None:
        raise ValueError("Per-output slicing of grouped denoising_env is unsupported")

    new_dit = None
    if rtd.dit_trajectory is not None:
        new_dit = RolloutDitTrajectory(
            latents=_slice_row_keepdim(rtd.dit_trajectory.latents, idx),
            timesteps=rtd.dit_trajectory.timesteps,
            sigmas=rtd.dit_trajectory.sigmas,
        )
        _aud = getattr(rtd.dit_trajectory, "audio_latents", None)
        if _aud is not None:
            new_dit.audio_latents = _slice_row_keepdim(_aud, idx)

    new_debug = None
    if rtd.rollout_debug_tensors is not None:
        d = rtd.rollout_debug_tensors
        new_debug = RolloutDebugTensors(
            rollout_variance_noises=_slice_row_keepdim(d.rollout_variance_noises, idx),
            rollout_prev_sample_means=_slice_row_keepdim(d.rollout_prev_sample_means, idx),
            rollout_noise_std_devs=_slice_row_keepdim(d.rollout_noise_std_devs, idx),
            rollout_model_outputs=_slice_row_keepdim(d.rollout_model_outputs, idx),
        )

    return RolloutTrajectoryData(
        rollout_log_probs=_slice_row_keepdim(rtd.rollout_log_probs, idx),
        rollout_debug_tensors=new_debug,
        denoising_env=None,
        dit_trajectory=new_dit,
    )


def patch_rollout_trajectory() -> None:
    """Concat per-output trajectories in the merge + slice them per output result."""
    _patch_merge()
    _patch_result_common()


def _patch_merge() -> None:
    from sglang.multimodal_gen.runtime.managers.gpu_worker import GPUWorker

    orig_sm = GPUWorker.__dict__.get("_merge_expanded_output_batches")
    if orig_sm is None:
        raise AttributeError("GPUWorker._merge_expanded_output_batches missing upstream")
    raw = orig_sm.__func__ if isinstance(orig_sm, staticmethod) else orig_sm
    if getattr(raw, _MERGE_SENTINEL, False):
        return

    def _merge_expanded_output_batches(output_batches):
        merged = raw(output_batches)
        fixed = _concat_rollout_trajectory_data(output_batches)
        if fixed is not None:
            merged.rollout_trajectory_data = fixed
        return merged

    setattr(_merge_expanded_output_batches, _MERGE_SENTINEL, True)
    GPUWorker._merge_expanded_output_batches = staticmethod(_merge_expanded_output_batches)


def _patch_result_common() -> None:
    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
        DiffGenerator,
    )

    orig_sm = DiffGenerator.__dict__.get("_result_common")
    if orig_sm is None:
        raise AttributeError("DiffGenerator._result_common missing upstream")
    raw = orig_sm.__func__ if isinstance(orig_sm, staticmethod) else orig_sm
    if getattr(raw, _RESULT_SENTINEL, False):
        return

    def _result_common(req, output_batch, generation_time, output_index=None):
        d = raw(req, output_batch, generation_time, output_index)
        if output_index is not None and isinstance(d, dict):
            rtd = d.get("rollout_trajectory_data")
            if rtd is not None:
                d["rollout_trajectory_data"] = _slice_rollout_trajectory_keepdim(rtd, int(output_index))
        return d

    setattr(_result_common, _RESULT_SENTINEL, True)
    DiffGenerator._result_common = staticmethod(_result_common)


__all__ = ["patch_rollout_trajectory"]
