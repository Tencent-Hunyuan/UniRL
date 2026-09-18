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


def _cat0(values: list) -> object:
    """Concat a list of batch-dim-0 tensors; fall back to the first if not all are tensors."""
    if not values or not all(isinstance(v, torch.Tensor) for v in values):
        return values[0] if values else None
    return torch.cat([v if v.dim() >= 1 else v.unsqueeze(0) for v in values], dim=0)


def _concat_rollout_trajectory_data(output_batches: list):
    """Build ONE ``RolloutTrajectoryData`` concatenated across the per-output batches."""
    RolloutTrajectoryData, RolloutDitTrajectory, RolloutDebugTensors = _rl_dataclasses()

    rtds = [
        getattr(ob, "rollout_trajectory_data", None)
        for ob in output_batches
        if getattr(ob, "rollout_trajectory_data", None) is not None
    ]
    if not rtds:
        return None
    if len(rtds) == 1:
        return rtds[0]

    first = rtds[0]

    new_dit = None
    if first.dit_trajectory is not None:
        new_dit = RolloutDitTrajectory(
            latents=_cat0([r.dit_trajectory.latents for r in rtds if r.dit_trajectory is not None]),
            timesteps=first.dit_trajectory.timesteps,
        )
        _auds = [getattr(r.dit_trajectory, "audio_latents", None) for r in rtds if r.dit_trajectory is not None]
        if _auds and all(a is not None for a in _auds):
            new_dit.audio_latents = _cat0(_auds)

    new_debug = None
    if first.rollout_debug_tensors is not None:

        def _dbg(field: str):
            return _cat0([getattr(r.rollout_debug_tensors, field) for r in rtds if r.rollout_debug_tensors is not None])

        new_debug = RolloutDebugTensors(
            rollout_variance_noises=_dbg("rollout_variance_noises"),
            rollout_prev_sample_means=_dbg("rollout_prev_sample_means"),
            rollout_noise_std_devs=_dbg("rollout_noise_std_devs"),
            rollout_model_outputs=_dbg("rollout_model_outputs"),
        )

    return RolloutTrajectoryData(
        rollout_log_probs=_cat0([r.rollout_log_probs for r in rtds]),
        rollout_debug_tensors=new_debug,
        denoising_env=first.denoising_env,
        dit_trajectory=new_dit,
    )


def _slice_row_keepdim(t, idx: int):
    """Row ``idx`` of a group tensor, KEEP-DIM ``[1, ...]``; pass a non-batched tensor through unchanged."""
    if isinstance(t, torch.Tensor) and t.dim() >= 1 and t.shape[0] > 1 and idx < t.shape[0]:
        return t[idx : idx + 1].contiguous()
    return t


def _slice_rollout_trajectory_keepdim(rtd, idx: int):
    """Per-output slice of a concatenated ``[K, ...]`` trajectory, keep-dim."""
    if rtd is None:
        return None
    RolloutTrajectoryData, RolloutDitTrajectory, RolloutDebugTensors = _rl_dataclasses()

    new_dit = None
    if rtd.dit_trajectory is not None:
        new_dit = RolloutDitTrajectory(
            latents=_slice_row_keepdim(rtd.dit_trajectory.latents, idx),
            timesteps=rtd.dit_trajectory.timesteps,
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
        denoising_env=rtd.denoising_env,
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
