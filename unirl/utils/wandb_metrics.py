"""Helpers for building structured WandB metrics in train loops."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

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
    # Non-finite marks "not scored" rows (per-domain component rewards use NaN).
    flat = flat[torch.isfinite(flat)]
    if flat.numel() == 0:
        return {}
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


def compute_rollout_sample_metrics(*, sample: Any, trunc_len: Optional[int] = None) -> Dict[str, float]:
    """Build rollout metrics directly from a :class:`Sample`.

    Walks the **gen Parts** of ``sample`` (those with ``sampling_params``
    set) and emits per-part metrics under the ``rollout/`` prefix:

    - ``num_samples`` (the sample's ``batch_size``)
    - For each gen Part: ``reward_{mean,std,min,max}``,
      ``advantage_{mean,std,min,max}``,
      ``reward_<component>_{mean,std,min,max}`` per
      ``part.component_rewards`` entry (``/`` flattened to ``_``),
      ``group_count``, ``zero_std_group_ratio``,
      ``zero_std_group_count`` when the part's ``group_ids`` is
      populated.

    Each gen Part is named ``"ar"`` when its ``sampling_params`` is an
    :class:`ARSamplingParams`, else ``"image"`` (matching
    ``BaseTrainer._drop_decoded``). For a single gen-part sample (the
    common case today: one diffusion or one AR part) keys are emitted
    unprefixed. With multiple gen Parts each part's metrics are
    namespaced under its derived name (e.g. ``image_reward_mean``,
    ``ar_reward_mean``).
    """
    from unirl.types.sampling import ARSamplingParams

    metrics: Dict[str, float] = {}

    parts = getattr(sample, "parts", None)
    if not isinstance(parts, list):
        metrics["num_samples"] = float(int(getattr(sample, "batch_size", 0)))
        return metrics

    gen_parts = [p for p in parts if getattr(p, "is_gen", False)]
    metrics["num_samples"] = (
        float(int(gen_parts[-1].batch_size)) if gen_parts else float(int(getattr(sample, "batch_size", 0)))
    )
    multi = len(gen_parts) > 1
    for part in gen_parts:
        name = "ar" if isinstance(part.sampling_params, ARSamplingParams) else "diffusion"
        prefix = f"{name}_" if multi else ""
        rewards = getattr(part, "rewards", None)
        if torch.is_tensor(rewards) and rewards.numel() > 0:
            rewards_f = rewards.detach().to(dtype=torch.float32).reshape(-1).cpu()
            metrics.update(_tensor_stats(f"{prefix}reward", rewards_f))
            zero_cnt, group_cnt = _zero_std_group_counts_from_ids(
                rewards_f,
                getattr(part, "group_ids", None),
            )
            if group_cnt > 0:
                metrics[f"{prefix}zero_std_group_ratio"] = float(zero_cnt) / float(group_cnt)
                metrics[f"{prefix}zero_std_group_count"] = float(zero_cnt)
                metrics[f"{prefix}group_count"] = float(group_cnt)

        advantages = getattr(part, "advantages", None)
        if torch.is_tensor(advantages) and advantages.numel() > 0:
            adv_f = advantages.detach().to(dtype=torch.float32).reshape(-1).cpu()
            metrics.update(_tensor_stats(f"{prefix}advantage", adv_f))

        segment = getattr(part, "segment", None)
        lengths = getattr(segment, "lengths", None) if segment is not None else None
        if torch.is_tensor(lengths) and lengths.numel() > 0:
            len_f = lengths.detach().to(dtype=torch.float32).reshape(-1).cpu()
            metrics.update(_tensor_stats(f"{prefix}response_len", len_f))
            if trunc_len is not None and int(trunc_len) > 0:
                metrics[f"{prefix}trunc_ratio"] = float(
                    (len_f >= float(int(trunc_len))).to(dtype=torch.float32).mean().item()
                )

        component_rewards = getattr(part, "component_rewards", None)
        if isinstance(component_rewards, dict):
            for cname, tensor in component_rewards.items():
                if not torch.is_tensor(tensor) or tensor.numel() == 0:
                    continue
                safe_name = str(cname).replace("/", "_")
                cat = tensor.detach().to(dtype=torch.float32).reshape(-1).cpu()
                metrics.update(_tensor_stats(f"{prefix}reward_{safe_name}", cat))

    return metrics


def pooled_window_reward_metrics(parts: Sequence[Any]) -> Dict[str, float]:
    """Reward metrics pooled over an accumulation window's gen Parts.

    Emits the same keys as the single-gen-part branch of
    :func:`compute_rollout_sample_metrics` (``num_samples``, ``reward_*``,
    ``reward_<component>_*``, group stats), so the trainer can merge them over
    the final rollout's partial view via ``extra_metrics``. With per-domain
    scorers each rollout carries NaN outside its own domain; pooling plus the
    finite-filter in :func:`_tensor_stats` gives every domain a finite point.
    """
    metrics: Dict[str, float] = {"num_samples": float(sum(int(p.batch_size) for p in parts))}
    rewards = [getattr(p, "rewards", None) for p in parts]
    if all(torch.is_tensor(r) and r.numel() > 0 for r in rewards):
        pooled = torch.cat([r.detach().to(dtype=torch.float32).reshape(-1).cpu() for r in rewards])
        metrics.update(_tensor_stats("reward", pooled))
        group_ids = [g for p in parts for g in (getattr(p, "group_ids", None) or [])]
        zero_cnt, group_cnt = _zero_std_group_counts_from_ids(pooled, group_ids)
        if group_cnt > 0:
            metrics["zero_std_group_ratio"] = float(zero_cnt) / float(group_cnt)
            metrics["zero_std_group_count"] = float(zero_cnt)
            metrics["group_count"] = float(group_cnt)
    components = [getattr(p, "component_rewards", None) for p in parts]
    if components and all(isinstance(c, dict) for c in components):
        for cname in sorted(set.intersection(*(set(c) for c in components))):
            tensors = [c[cname] for c in components]
            if all(torch.is_tensor(t) and t.numel() > 0 for t in tensors):
                pooled_c = torch.cat([t.detach().to(dtype=torch.float32).reshape(-1).cpu() for t in tensors])
                metrics.update(_tensor_stats(f"reward_{str(cname).replace('/', '_')}", pooled_c))
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
