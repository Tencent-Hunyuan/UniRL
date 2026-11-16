"""Reduce numeric metrics across repeated metric dictionaries."""

from typing import Any, Dict, List, Mapping, Sequence

import torch


def aggregate_numeric_metrics(metrics_list: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    """Sum entropy statistics, take extrema for ``*_min``/``*_max``, and average other numeric metrics."""
    aggregated: Dict[str, float] = {}
    if not metrics_list:
        return aggregated

    all_keys = set()
    for metrics in metrics_list:
        all_keys.update(metrics.keys())

    for key in all_keys:
        if key == "policy_entropy" and "policy_entropy_count" in all_keys:
            continue
        if key in ("policy_entropy_sum", "policy_entropy_count"):
            total = sum(metrics[key] for metrics in metrics_list if key in metrics)
            aggregated[key] = float(total)
            continue
        values: List[float] = []
        for metrics in metrics_list:
            if key not in metrics:
                continue
            value = metrics[key]
            if isinstance(value, torch.Tensor):
                value = value.item() if value.numel() == 1 else value.mean().item()
            if isinstance(value, bool):
                values.append(float(value))
            elif isinstance(value, (int, float)):
                values.append(float(value))
        if values:
            # torch max/min propagate NaN like the mean does; builtin max/min drop it order-dependently.
            if key.endswith("_max"):
                aggregated[key] = torch.tensor(values, dtype=torch.float64).max().item()
            elif key.endswith("_min"):
                aggregated[key] = torch.tensor(values, dtype=torch.float64).min().item()
            else:
                aggregated[key] = sum(values) / len(values)

    if "policy_entropy_count" in aggregated:
        count = aggregated["policy_entropy_count"]
        aggregated["policy_entropy"] = aggregated["policy_entropy_sum"] / count if count else 0.0
    return aggregated
