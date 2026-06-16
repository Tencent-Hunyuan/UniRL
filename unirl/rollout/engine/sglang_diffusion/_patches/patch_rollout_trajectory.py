"""Per-sample rollout-trajectory preservation across the grouped forward (LIN-365).

THE GRADIENT-KILLER for SD3 GRPO on patched-upstream sglang. Symptom:
``grad_norm≈0`` / flat reward despite the SAME recipe training on the fork and
vLLM-Omni. Root cause is an upstream **expanded-output merge** bug, not the SDE
math, conditions, advantages, or x_T recipe (all ruled out).

Upstream runs a ``num_outputs_per_prompt=K`` GRPO group as K per-output ``B=1``
forwards (``GPUWorker._forward_group`` -> ``pipeline.forward_batch``), each
producing its OWN distinct ``batch.rollout_trajectory_data`` (distinct x_T slice
+ distinct per-step SDE noise -> distinct trajectory latents + native log-probs).
Two upstream sites then collapse those K distinct trajectories into K copies of
output 0's:

1. ``GPUWorker._merge_expanded_singletons`` keeps ``rollout_trajectory_data``
   from only the FIRST per-output batch (``if merged.rollout_trajectory_data is
   None``), discarding outputs 1..K-1.
2. ``DiffGenerator._result_common`` slices ``samples`` per ``output_index`` but
   passes the WHOLE (now singleton) ``rollout_trajectory_data`` to EVERY
   per-output ``GenerationResult`` -- it never slices the trajectory.

So all K results carry output 0's trajectory. ``response.py`` cats them ->
byte-identical per-sample trajectories within the group -> identical
``grad(logp_i)`` -> the mean-zero GRPO advantages cancel (~667x) -> zero
gradient -> flat reward. The fork never hit this: it ran the whole group as ONE
``batch_size=K`` forward, so ``rollout_trajectory_data`` was natively ``[K, ...]``.

Fix (mirrors how ``samples`` are already handled), two AROUND-wraps:

* ``_merge_expanded_output_batches`` -> after the upstream merge, REPLACE the
  collapsed ``rollout_trajectory_data`` with one CONCATENATED (dim 0) across all
  K per-output batches: ``dit_trajectory.latents`` ``[K, T+1, ...]`` +
  ``rollout_log_probs`` ``[K, T]`` (+ debug tensors). ``timesteps`` /
  ``denoising_env`` are group-shared, kept from output 0. This also fixes the
  HTTP rollout path, whose ``_slice_rollout_trajectory_for_sample`` was likewise
  broadcasting output 0 (its ``_extract_single_sample_tensor`` returns the whole
  tensor when ``shape[0] != batch_size``).
* ``_result_common`` -> slice the (now ``[K, ...]``) trajectory to this output's
  ``[output_index:output_index+1]`` KEEP-DIM row, so each ``GenerationResult``
  carries its own ``[1, T+1, ...]`` -- the shape ``response.py`` cats per result.

Idempotent; AROUND-wrap of two staticmethods only -- no sglang source edits.
"""

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
    """Concat a list of batch-dim-0 tensors; fall back to the first if not all
    are tensors (a missing field must not crash the merge)."""
    if not values or not all(isinstance(v, torch.Tensor) for v in values):
        return values[0] if values else None
    return torch.cat([v if v.dim() >= 1 else v.unsqueeze(0) for v in values], dim=0)


def _concat_rollout_trajectory_data(output_batches: list):
    """Build ONE ``RolloutTrajectoryData`` concatenated across the per-output batches.

    Preserves every output's distinct trajectory (the whole point); ``timesteps``
    and ``denoising_env`` are group-shared so output 0's copy is kept.
    """
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
    """Row ``idx`` of a group tensor, KEEP-DIM (``[1, ...]``); pass through a
    ``[1, ...]`` / non-batched tensor unchanged."""
    if isinstance(t, torch.Tensor) and t.dim() >= 1 and t.shape[0] > 1 and idx < t.shape[0]:
        return t[idx : idx + 1].contiguous()
    return t


def _slice_rollout_trajectory_keepdim(rtd, idx: int):
    """Per-output slice of a concatenated ``[K, ...]`` trajectory, keep-dim.

    ``response.py`` reads one trajectory per result and cats them, so each result
    must carry its own ``[1, T+1, ...]`` (NOT the squeezed ``[T+1, ...]`` the
    upstream HTTP slicer produces). No-op when the field is already ``[1, ...]``.
    """
    if rtd is None:
        return None
    RolloutTrajectoryData, RolloutDitTrajectory, RolloutDebugTensors = _rl_dataclasses()

    new_dit = None
    if rtd.dit_trajectory is not None:
        new_dit = RolloutDitTrajectory(
            latents=_slice_row_keepdim(rtd.dit_trajectory.latents, idx),
            timesteps=rtd.dit_trajectory.timesteps,
        )

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
