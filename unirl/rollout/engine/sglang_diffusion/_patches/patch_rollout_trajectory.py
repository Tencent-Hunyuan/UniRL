"""Per-sample rollout-trajectory preservation across the grouped forward."""

from __future__ import annotations

import torch

_MERGE_SENTINEL = "_unirl_rtd_concat"
_RESULT_SENTINEL = "_unirl_rtd_slice"


def _rl_dataclasses():
    from sglang.multimodal_gen.runtime.post_training.rl_dataclasses import (
        RolloutDebugTensors,
        RolloutDenoisingEnv,
        RolloutDitTrajectory,
        RolloutTrajectoryData,
    )

    return RolloutTrajectoryData, RolloutDitTrajectory, RolloutDebugTensors, RolloutDenoisingEnv


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


def _merge_env_tree(values: list, *, field: str, current_key: str | None = None):
    """Merge per-output denoising metadata while preserving nested alignment."""
    present = _require_uniform_presence(values, field=field)
    if present is None:
        return None
    first = present[0]
    if isinstance(first, torch.Tensor):
        return _cat0(present, field=field)
    if isinstance(first, dict):
        keys = set(first)
        if any(not isinstance(value, dict) or set(value) != keys for value in present):
            raise ValueError(f"Grouped rollout field {field!r} has mismatched dict keys")
        return {
            key: _merge_env_tree(
                [value[key] for value in present],
                field=f"{field}.{key}",
                current_key=key,
            )
            for key in first
        }
    if isinstance(first, list):
        if any(not isinstance(value, list) for value in present):
            raise TypeError(f"Grouped rollout field {field!r} has mixed container types")
        if current_key == "img_shapes":
            return [item for value in present for item in value]
        if any(len(value) != len(first) for value in present):
            raise ValueError(f"Grouped rollout field {field!r} has mismatched list lengths")
        return [
            _merge_env_tree(
                [value[index] for value in present],
                field=f"{field}[{index}]",
                current_key=current_key,
            )
            for index in range(len(first))
        ]
    if isinstance(first, tuple):
        if any(not isinstance(value, tuple) or len(value) != len(first) for value in present):
            raise ValueError(f"Grouped rollout field {field!r} has mismatched tuples")
        return tuple(
            _merge_env_tree(
                [value[index] for value in present],
                field=f"{field}[{index}]",
                current_key=current_key,
            )
            for index in range(len(first))
        )
    if any(value != first for value in present[1:]):
        raise ValueError(f"Grouped rollout field {field!r} differs across outputs")
    return first


def _slice_env_tree(value, idx: int, batch_size: int, *, field: str, current_key: str | None = None):
    """Slice one output from nested denoising metadata, keeping tensor batch dims."""
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.dim() >= 1 and int(value.shape[0]) == batch_size:
            return value[idx : idx + 1].contiguous()
        return value
    if isinstance(value, dict):
        return {
            key: _slice_env_tree(
                child,
                idx,
                batch_size,
                field=f"{field}.{key}",
                current_key=key,
            )
            for key, child in value.items()
        }
    if isinstance(value, list):
        if current_key == "img_shapes" and len(value) == batch_size:
            return [value[idx]]
        return [
            _slice_env_tree(
                child,
                idx,
                batch_size,
                field=f"{field}[{index}]",
                current_key=current_key,
            )
            for index, child in enumerate(value)
        ]
    if isinstance(value, tuple):
        return tuple(
            _slice_env_tree(
                child,
                idx,
                batch_size,
                field=f"{field}[{index}]",
                current_key=current_key,
            )
            for index, child in enumerate(value)
        )
    return value


def _concat_rollout_trajectory_data(output_batches: list):
    """Build ONE ``RolloutTrajectoryData`` concatenated across the per-output batches."""
    RolloutTrajectoryData, RolloutDitTrajectory, RolloutDebugTensors, RolloutDenoisingEnv = _rl_dataclasses()

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

    new_env = None
    denoising_envs = _require_uniform_presence(
        [rtd.denoising_env for rtd in rtds],
        field="denoising_env",
    )
    if denoising_envs is not None:
        new_env = RolloutDenoisingEnv(
            image_kwargs=_merge_env_tree(
                [env.image_kwargs for env in denoising_envs],
                field="denoising_env.image_kwargs",
            ),
            pos_cond_kwargs=_merge_env_tree(
                [env.pos_cond_kwargs for env in denoising_envs],
                field="denoising_env.pos_cond_kwargs",
            ),
            neg_cond_kwargs=_merge_env_tree(
                [env.neg_cond_kwargs for env in denoising_envs],
                field="denoising_env.neg_cond_kwargs",
            ),
            guidance=_merge_env_tree(
                [env.guidance for env in denoising_envs],
                field="denoising_env.guidance",
            ),
        )

    return RolloutTrajectoryData(
        rollout_log_probs=_cat0(
            [rtd.rollout_log_probs for rtd in rtds],
            field="rollout_log_probs",
        ),
        rollout_debug_tensors=new_debug,
        denoising_env=new_env,
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


def _first_tensor_batch_size(value) -> int | None:
    if isinstance(value, torch.Tensor) and value.dim() >= 1:
        return int(value.shape[0])
    if isinstance(value, dict):
        children = value.values()
    elif isinstance(value, (list, tuple)):
        children = value
    else:
        return None
    for child in children:
        size = _first_tensor_batch_size(child)
        if size is not None:
            return size
    return None


def _trajectory_batch_size(rtd) -> int:
    candidates = [rtd.rollout_log_probs]
    if rtd.dit_trajectory is not None:
        candidates.append(rtd.dit_trajectory.latents)
    if rtd.rollout_debug_tensors is not None:
        candidates.extend(
            [
                rtd.rollout_debug_tensors.rollout_variance_noises,
                rtd.rollout_debug_tensors.rollout_prev_sample_means,
                rtd.rollout_debug_tensors.rollout_noise_std_devs,
                rtd.rollout_debug_tensors.rollout_model_outputs,
            ]
        )
    if rtd.denoising_env is not None:
        candidates.extend(
            [
                rtd.denoising_env.guidance,
                rtd.denoising_env.image_kwargs,
                rtd.denoising_env.pos_cond_kwargs,
                rtd.denoising_env.neg_cond_kwargs,
            ]
        )
    for candidate in candidates:
        size = _first_tensor_batch_size(candidate)
        if size is not None:
            return size
    raise ValueError("Grouped rollout data has no batched tensor to determine output count")


def _slice_rollout_trajectory_keepdim(rtd, idx: int):
    """Per-output slice of a concatenated ``[K, ...]`` trajectory, keep-dim."""
    if rtd is None:
        return None
    RolloutTrajectoryData, RolloutDitTrajectory, RolloutDebugTensors, RolloutDenoisingEnv = _rl_dataclasses()

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

    new_env = None
    if rtd.denoising_env is not None:
        env = rtd.denoising_env
        batch_size = _trajectory_batch_size(rtd)
        new_env = RolloutDenoisingEnv(
            image_kwargs=_slice_env_tree(
                env.image_kwargs,
                idx,
                batch_size,
                field="denoising_env.image_kwargs",
            ),
            pos_cond_kwargs=_slice_env_tree(
                env.pos_cond_kwargs,
                idx,
                batch_size,
                field="denoising_env.pos_cond_kwargs",
            ),
            neg_cond_kwargs=_slice_env_tree(
                env.neg_cond_kwargs,
                idx,
                batch_size,
                field="denoising_env.neg_cond_kwargs",
            ),
            guidance=_slice_env_tree(
                env.guidance,
                idx,
                batch_size,
                field="denoising_env.guidance",
            ),
        )

    return RolloutTrajectoryData(
        rollout_log_probs=_slice_row_keepdim(rtd.rollout_log_probs, idx),
        rollout_debug_tensors=new_debug,
        denoising_env=new_env,
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
