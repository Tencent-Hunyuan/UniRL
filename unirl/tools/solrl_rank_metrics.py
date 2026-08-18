"""Compare Sol-RL scout reward traces against a BF16 full-step oracle."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Dict, Iterable, List, Tuple


def _average_ranks(values: List[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        stop = cursor + 1
        while stop < len(order) and values[order[stop]] == values[order[cursor]]:
            stop += 1
        rank = (cursor + stop - 1) / 2.0
        for position in range(cursor, stop):
            ranks[order[position]] = rank
        cursor = stop
    return ranks


def _pearson(left: List[float], right: List[float]) -> float:
    left_mean, right_mean = fmean(left), fmean(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    left_norm = sum((x - left_mean) ** 2 for x in left)
    right_norm = sum((y - right_mean) ** 2 for y in right)
    denominator = math.sqrt(left_norm * right_norm)
    return numerator / denominator if denominator else 0.0


def _kendall_tau_b(left: List[float], right: List[float]) -> float:
    concordant = discordant = tie_left = tie_right = 0
    for i in range(len(left)):
        for j in range(i + 1, len(left)):
            dx = (left[i] > left[j]) - (left[i] < left[j])
            dy = (right[i] > right[j]) - (right[i] < right[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tie_left += 1
            elif dy == 0:
                tie_right += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt((concordant + discordant + tie_left) * (concordant + discordant + tie_right))
    return (concordant - discordant) / denominator if denominator else 0.0


def _load_traces(path: Path) -> Dict[Tuple[int, str], dict]:
    traces: Dict[Tuple[int, str], dict] = {}
    for file in sorted(path.glob("rollout_*.json")):
        with file.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        rollout_id = int(payload["rollout_id"])
        for group in payload["groups"]:
            traces[(rollout_id, str(group["group_id"]))] = group
    return traces


def _selected_ids(candidates: Iterable[dict], kind: str) -> set[str]:
    return {str(item["sample_id"]) for item in candidates if item.get("selection") == kind}


def compare(proxy_dir: Path, oracle_dir: Path) -> dict:
    proxy = _load_traces(proxy_dir)
    oracle = _load_traces(oracle_dir)
    keys = sorted(set(proxy) & set(oracle))
    if not keys:
        raise ValueError("No matching (rollout_id, group_id) traces found.")

    spearman: List[float] = []
    kendall: List[float] = []
    top_overlap: List[float] = []
    bottom_overlap: List[float] = []
    top_true_gap: List[float] = []
    bottom_true_gap: List[float] = []
    for key in keys:
        proxy_candidates = proxy[key]["candidates"]
        oracle_candidates = oracle[key]["candidates"]
        proxy_by_id = {str(item["sample_id"]): item for item in proxy_candidates}
        oracle_by_id = {str(item["sample_id"]): item for item in oracle_candidates}
        ids = [sample_id for sample_id in proxy_by_id if sample_id in oracle_by_id]
        if len(ids) != len(proxy_candidates) or len(ids) != len(oracle_candidates):
            raise ValueError(f"Candidate id mismatch for trace group {key}.")
        proxy_rewards = [float(proxy_by_id[sample_id]["reward"]) for sample_id in ids]
        oracle_rewards = [float(oracle_by_id[sample_id]["reward"]) for sample_id in ids]
        spearman.append(_pearson(_average_ranks(proxy_rewards), _average_ranks(oracle_rewards)))
        kendall.append(_kendall_tau_b(proxy_rewards, oracle_rewards))

        oracle_reward_by_id = {sample_id: float(oracle_by_id[sample_id]["reward"]) for sample_id in ids}
        for kind, overlaps, gaps in (
            ("top", top_overlap, top_true_gap),
            ("bottom", bottom_overlap, bottom_true_gap),
        ):
            proxy_ids = _selected_ids(proxy_candidates, kind)
            oracle_ids = _selected_ids(oracle_candidates, kind)
            if not proxy_ids or len(proxy_ids) != len(oracle_ids):
                raise ValueError(f"Selection labels for {kind} do not align in trace group {key}.")
            overlaps.append(len(proxy_ids & oracle_ids) / len(oracle_ids))
            proxy_true = fmean(oracle_reward_by_id[sample_id] for sample_id in proxy_ids)
            oracle_true = fmean(oracle_reward_by_id[sample_id] for sample_id in oracle_ids)
            gaps.append(proxy_true - oracle_true)

    return {
        "groups": len(keys),
        "spearman_mean": fmean(spearman),
        "kendall_tau_b_mean": fmean(kendall),
        "top_overlap_mean": fmean(top_overlap),
        "bottom_overlap_mean": fmean(bottom_overlap),
        "top_selected_true_reward_gap": fmean(top_true_gap),
        "bottom_selected_true_reward_gap": fmean(bottom_true_gap),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy-dir", type=Path, required=True)
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    metrics = compare(args.proxy_dir, args.oracle_dir)
    text = json.dumps(metrics, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{text}\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
