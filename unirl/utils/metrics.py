"""Semantically aggregate numeric metrics across repeated dictionaries."""

from typing import Any, Dict, List

import torch


def aggregate_numeric_metrics(metrics_list: List[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate canonical counters/maxima exactly; average other metrics."""
    aggregated: Dict[str, float] = {}
    if not metrics_list:
        return aggregated

    all_keys = set()
    for metrics in metrics_list:
        all_keys.update(metrics.keys())

    for key in all_keys:
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
            if key in {"token_count", "mismatch_count"}:
                aggregated[key] = sum(values)
            elif key == "torch_equal_fp32":
                aggregated[key] = min(values)
            elif key in {"max_absdiff_fp32", "k3_max"}:
                aggregated[key] = max(values)
            elif key == "k3_mean":
                weighted = []
                for metrics in metrics_list:
                    value = metrics.get(key)
                    if isinstance(value, torch.Tensor):
                        value = value.item() if value.numel() == 1 else None
                    if not isinstance(value, (int, float)):
                        continue
                    count = metrics.get("token_count")
                    if isinstance(count, torch.Tensor):
                        count = count.item() if count.numel() == 1 else None
                    if isinstance(count, (int, float)) and float(count) >= 0:
                        weighted.append((value, float(count)))
                total_weight = sum(weight for _, weight in weighted)
                aggregated[key] = (
                    sum(value * weight for value, weight in weighted) / total_weight
                    if total_weight > 0
                    else sum(values) / len(values)
                )
            else:
                aggregated[key] = sum(values) / len(values)

    return aggregated
