"""Helpers for building structured WandB metrics in train loops."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch


def _coerce_scalar(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if torch.is_tensor(value):
        tensor = value.detach()
        if tensor.numel() == 0:
            return None
        if tensor.numel() == 1:
            return float(tensor.item())
        return float(tensor.to(dtype=torch.float32).mean().item())
    return None


def flatten_numeric_metrics(
    payload: Dict[str, Any],
    *,
    prefix: str = "",
) -> Dict[str, float]:
    """Flatten nested dict payload into numeric metrics only."""
    output: Dict[str, float] = {}

    def _walk(node: Dict[str, Any], node_prefix: str) -> None:
        for key, value in node.items():
            metric_key = f"{node_prefix}{key}" if node_prefix else str(key)
            if isinstance(value, dict):
                _walk(value, f"{metric_key}/")
                continue
            scalar = _coerce_scalar(value)
            if scalar is not None:
                output[metric_key] = scalar

    _walk(payload, prefix)
    return output


def _tensor_stats(prefix: str, tensor: Optional[torch.Tensor]) -> Dict[str, float]:
    if tensor is None or (not torch.is_tensor(tensor)) or tensor.numel() == 0:
        return {}
    flat = tensor.detach().to(dtype=torch.float32).reshape(-1).cpu()
    return {
        f"{prefix}_mean": float(flat.mean().item()),
        f"{prefix}_std": float(flat.std(unbiased=False).item()),
        f"{prefix}_min": float(flat.min().item()),
        f"{prefix}_max": float(flat.max().item()),
    }


def _zero_std_group_counts_from_ids(
    rewards: torch.Tensor,
    group_ids: Optional[List[str]],
) -> tuple[int, int]:
    if not isinstance(group_ids, list) or len(group_ids) != int(rewards.shape[0]):
        return 0, 0
    ordered: Dict[str, List[float]] = {}
    rewards_f = rewards.to(dtype=torch.float32).reshape(-1)
    for sample_idx, raw_group_id in enumerate(group_ids):
        group_id = str(raw_group_id).strip()
        if not group_id:
            continue
        ordered.setdefault(group_id, []).append(float(rewards_f[sample_idx].item()))
    if not ordered:
        return 0, 0
    zero_std = 0
    for values in ordered.values():
        if len(values) <= 1:
            continue
        std = torch.tensor(values, dtype=torch.float32).std(unbiased=False)
        if float(std.item()) <= 1e-8:
            zero_std += 1
    return zero_std, len(ordered)


def compute_rollout_resp_metrics(*, resp: Any, trunc_len: Optional[int] = None) -> Dict[str, float]:
    """Build rollout metrics directly from a :class:`RolloutResp`.

    Walks ``resp.tracks`` and emits per-track metrics under the
    ``rollout/`` prefix:

    - ``num_samples`` (sum across tracks for the resp batch_size)
    - For each track: ``reward_{mean,std,min,max}``,
      ``advantage_{mean,std,min,max}``,
      ``reward_<component>_{mean,std,min,max}`` per
      ``track.component_rewards`` entry (``/`` flattened to ``_``),
      ``group_count``, ``zero_std_group_ratio``,
      ``zero_std_group_count`` when the track's ``group_ids`` is
      populated.

    For single-track resps (the common case today: one diffusion
    track), keys are emitted unprefixed. For multi-track resps each
    track's metrics are namespaced under its track name (e.g.
    ``image_reward_mean``, ``refined_reward_mean``).
    """
    metrics: Dict[str, float] = {}

    metrics["num_samples"] = float(int(getattr(resp, "batch_size", 0)))

    tracks = getattr(resp, "tracks", None)
    if not isinstance(tracks, dict):
        return metrics

    multi = len(tracks) > 1
    for name, track in tracks.items():
        prefix = f"{name}_" if multi else ""
        rewards = getattr(track, "rewards", None)
        if torch.is_tensor(rewards) and rewards.numel() > 0:
            rewards_f = rewards.detach().to(dtype=torch.float32).reshape(-1).cpu()
            metrics.update(_tensor_stats(f"{prefix}reward", rewards_f))
            zero_cnt, group_cnt = _zero_std_group_counts_from_ids(
                rewards_f,
                getattr(track, "group_ids", None),
            )
            if group_cnt > 0:
                metrics[f"{prefix}zero_std_group_ratio"] = float(zero_cnt) / float(group_cnt)
                metrics[f"{prefix}zero_std_group_count"] = float(zero_cnt)
                metrics[f"{prefix}group_count"] = float(group_cnt)

        advantages = getattr(track, "advantages", None)
        if torch.is_tensor(advantages) and advantages.numel() > 0:
            adv_f = advantages.detach().to(dtype=torch.float32).reshape(-1).cpu()
            metrics.update(_tensor_stats(f"{prefix}advantage", adv_f))

        # Response-length stats from the packed varlen segment (AR tracks):
        # `segment.lengths[i]` = generated tokens for sample i. `trunc_ratio` is
        # the fraction that hit the generation budget (= truncated, usually no
        # final answer -> reward 0) — mirrors verl's response_length/{mean,clip_ratio}.
        segment = getattr(track, "segment", None)
        lengths = getattr(segment, "lengths", None) if segment is not None else None
        if torch.is_tensor(lengths) and lengths.numel() > 0:
            len_f = lengths.detach().to(dtype=torch.float32).reshape(-1).cpu()
            metrics.update(_tensor_stats(f"{prefix}response_len", len_f))
            if trunc_len is not None and int(trunc_len) > 0:
                metrics[f"{prefix}trunc_ratio"] = float(
                    (len_f >= float(int(trunc_len))).to(dtype=torch.float32).mean().item()
                )

        component_rewards = getattr(track, "component_rewards", None)
        if isinstance(component_rewards, dict):
            for cname, tensor in component_rewards.items():
                if not torch.is_tensor(tensor) or tensor.numel() == 0:
                    continue
                safe_name = str(cname).replace("/", "_")
                cat = tensor.detach().to(dtype=torch.float32).reshape(-1).cpu()
                metrics.update(_tensor_stats(f"{prefix}reward_{safe_name}", cat))

    return metrics


def build_sync_metrics(
    sync_result: Any,
    prefix: str = "sync/",
) -> Dict[str, float]:
    """Flatten weight-sync result into numeric metrics."""
    if sync_result is None:
        return {}

    metrics: Dict[str, float] = {}
    for key in ("elapsed_ms", "version", "rollout_id"):
        scalar = _coerce_scalar(getattr(sync_result, key, None))
        if scalar is not None:
            metrics[f"{prefix}{key}"] = scalar

    extra = getattr(sync_result, "extra", None)
    if isinstance(extra, dict):
        metrics.update(flatten_numeric_metrics(extra, prefix=f"{prefix}extra/"))
    return metrics
