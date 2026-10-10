"""Reduce numeric metrics, preserving additive policy entropy statistics."""

from typing import Any, Dict, List

import torch


def aggregate_numeric_metrics(metrics_list: List[Dict[str, Any]]) -> Dict[str, float]:
    """Average numeric keys; sum entropy statistics before computing their token-weighted mean."""
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
            aggregated[key] = sum(values) / len(values)

    if "policy_entropy_count" in aggregated:
        count = aggregated["policy_entropy_count"]
        aggregated["policy_entropy"] = aggregated["policy_entropy_sum"] / count if count else 0.0
    return aggregated
