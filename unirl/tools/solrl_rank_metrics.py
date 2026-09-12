"""Compare Sol-RL scout reward traces against a BF16 full-step oracle."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Dict, Iterable, List, Tuple


def _require_sha256(value: object, *, field: str, context: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{context} has invalid {field}; expected a lowercase SHA-256 hex digest.")
    return value


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


def _expected_selection(rewards: List[float], *, top_k: int, bottom_k: int) -> Dict[int, str]:
    """Recompute the trainer's stable, disjoint top/bottom labels."""
    descending = sorted(range(len(rewards)), key=lambda index: (-rewards[index], index))
    top = descending[:top_k]
    top_set = set(top)
    ascending = sorted(range(len(rewards)), key=lambda index: (rewards[index], index))
    bottom = [index for index in ascending if index not in top_set][:bottom_k]
    return {**{index: "top" for index in top}, **{index: "bottom" for index in bottom}}


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


def _load_traces(path: Path, *, rollout_id: int) -> Dict[Tuple[int, str], dict]:
    traces: Dict[Tuple[int, str], dict] = {}
    sources: Dict[Tuple[int, str], Path] = {}
    for file in sorted(path.glob("rollout_*.json")):
        with file.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        file_rollout_id = payload.get("rollout_id")
        if isinstance(file_rollout_id, bool) or not isinstance(file_rollout_id, int) or file_rollout_id < 0:
            raise ValueError(f"Invalid rollout_id in {file}: {file_rollout_id!r}.")
        if file_rollout_id != rollout_id:
            continue
        if payload.get("schema_version") != 1:
            raise ValueError(f"Unsupported or missing trace schema_version in {file}.")
        policy_version = payload.get("policy_version")
        if isinstance(policy_version, bool) or not isinstance(policy_version, int) or policy_version < 0:
            raise ValueError(f"Invalid policy_version in {file}: {policy_version!r}.")
        policy_snapshot_id = payload.get("policy_snapshot_id")
        if not isinstance(policy_snapshot_id, str) or not policy_snapshot_id:
            raise ValueError(f"Missing policy_snapshot_id in {file}.")
        model_fingerprint = _require_sha256(
            payload.get("model_fingerprint"),
            field="model_fingerprint",
            context=file,
        )
        pipeline_fingerprint = _require_sha256(
            payload.get("pipeline_fingerprint"),
            field="pipeline_fingerprint",
            context=file,
        )
        reward_fingerprint = _require_sha256(
            payload.get("reward_fingerprint"),
            field="reward_fingerprint",
            context=file,
        )
        comparison_fingerprint = _require_sha256(
            payload.get("comparison_fingerprint"),
            field="comparison_fingerprint",
            context=file,
        )
        top_k = payload.get("top_k")
        bottom_k = payload.get("bottom_k")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (top_k, bottom_k)):
            raise ValueError(f"Invalid top_k/bottom_k in {file}: {top_k!r}/{bottom_k!r}.")
        groups = payload.get("groups")
        if not isinstance(groups, list) or not groups:
            raise ValueError(f"Trace {file} must contain at least one group.")
        trace_meta = {
            "schema_version": 1,
            "policy_version": policy_version,
            "policy_snapshot_id": policy_snapshot_id,
            "model_fingerprint": model_fingerprint,
            "pipeline_fingerprint": pipeline_fingerprint,
            "reward_fingerprint": reward_fingerprint,
            "comparison_fingerprint": comparison_fingerprint,
            "top_k": top_k,
            "bottom_k": bottom_k,
        }
        for group in groups:
            group_id = group.get("group_id")
            if not isinstance(group_id, str) or not group_id:
                raise ValueError(f"Trace {file} has an invalid group_id {group_id!r}.")
            key = (file_rollout_id, group_id)
            if key in traces:
                raise ValueError(f"Duplicate trace group {key} in {sources[key]} and {file}.")
            for name in ("prompt_sha256", "noise_sha256"):
                _require_sha256(group.get(name), field=name, context=f"trace group {key}")
            candidates = group.get("candidates")
            if not isinstance(candidates, list) or not candidates:
                raise ValueError(f"Trace group {key} must contain candidates.")
            sample_ids: set[str] = set()
            selection_counts = {"top": 0, "bottom": 0}
            rewards: List[float] = []
            for candidate in candidates:
                sample_id = candidate.get("sample_id")
                if not isinstance(sample_id, str) or not sample_id or sample_id in sample_ids:
                    raise ValueError(f"Trace group {key} has an empty or duplicate sample_id {sample_id!r}.")
                sample_ids.add(sample_id)
                raw_reward = candidate.get("reward")
                if isinstance(raw_reward, bool) or not isinstance(raw_reward, (int, float)):
                    raise ValueError(f"Trace group {key} has non-numeric reward for {sample_id!r}.")
                reward = float(raw_reward)
                if not math.isfinite(reward):
                    raise ValueError(f"Trace group {key} has non-finite reward for {sample_id!r}.")
                candidate["reward"] = reward
                rewards.append(reward)
                selection = candidate.get("selection")
                if selection not in {None, "top", "bottom"}:
                    raise ValueError(f"Trace group {key} has unknown selection label {selection!r}.")
                if selection is not None:
                    selection_counts[selection] += 1
            if selection_counts != {"top": top_k, "bottom": bottom_k}:
                raise ValueError(
                    f"Trace group {key} selection counts {selection_counts} "
                    f"do not match top_k/bottom_k={top_k}/{bottom_k}."
                )
            expected_selection = _expected_selection(rewards, top_k=top_k, bottom_k=bottom_k)
            actual_selection = {
                index: candidate["selection"]
                for index, candidate in enumerate(candidates)
                if candidate["selection"] is not None
            }
            if actual_selection != expected_selection:
                raise ValueError(f"Trace group {key} selection labels do not match its recorded rewards.")
            traces[key] = {**group, "_trace_meta": trace_meta}
            sources[key] = file
    return traces


def _selected_ids(candidates: Iterable[dict], kind: str) -> set[str]:
    return {item["sample_id"] for item in candidates if item.get("selection") == kind}


def compare(proxy_dir: Path, oracle_dir: Path, *, rollout_id: int = 0) -> dict:
    if rollout_id != 0:
        raise ValueError(
            "Only rollout_id=0 is safe for cross-run comparison: independently trained arms diverge after update zero. "
            "Use a future paired-oracle trace mode for later policy versions."
        )
    proxy = _load_traces(proxy_dir, rollout_id=rollout_id)
    oracle = _load_traces(oracle_dir, rollout_id=rollout_id)
    proxy_keys = set(proxy)
    oracle_keys = set(oracle)
    if proxy_keys != oracle_keys:
        raise ValueError(
            "Trace groups differ: "
            f"missing from proxy={sorted(oracle_keys - proxy_keys)[:8]}, "
            f"missing from oracle={sorted(proxy_keys - oracle_keys)[:8]}."
        )
    keys = sorted(proxy_keys)
    if not keys:
        raise ValueError("No matching (rollout_id, group_id) traces found.")

    spearman: List[float] = []
    kendall: List[float] = []
    top_overlap: List[float] = []
    bottom_overlap: List[float] = []
    top_true_gap: List[float] = []
    bottom_true_gap: List[float] = []
    for key in keys:
        proxy_group = proxy[key]
        oracle_group = oracle[key]
        proxy_meta = proxy_group["_trace_meta"]
        oracle_meta = oracle_group["_trace_meta"]
        for field in (
            "schema_version",
            "policy_version",
            "policy_snapshot_id",
            "model_fingerprint",
            "pipeline_fingerprint",
            "reward_fingerprint",
            "comparison_fingerprint",
            "top_k",
            "bottom_k",
        ):
            if proxy_meta[field] != oracle_meta[field]:
                raise ValueError(f"Trace metadata {field} differs for group {key}.")
        if proxy_meta["policy_version"] != 0:
            raise ValueError(f"Trace group {key} is not from the initial shared policy snapshot.")
        for field in ("prompt_sha256", "noise_sha256"):
            if proxy_group[field] != oracle_group[field]:
                raise ValueError(f"Trace identity {field} differs for group {key}.")

        proxy_candidates = proxy_group["candidates"]
        oracle_candidates = oracle_group["candidates"]
        proxy_ids_in_order = [item["sample_id"] for item in proxy_candidates]
        oracle_ids_in_order = [item["sample_id"] for item in oracle_candidates]
        if proxy_ids_in_order != oracle_ids_in_order:
            raise ValueError(f"Candidate id order differs for trace group {key}.")
        proxy_by_id = {item["sample_id"]: item for item in proxy_candidates}
        oracle_by_id = {item["sample_id"]: item for item in oracle_candidates}
        ids = [sample_id for sample_id in proxy_by_id if sample_id in oracle_by_id]
        if len(ids) != len(proxy_candidates) or len(ids) != len(oracle_candidates):
            raise ValueError(f"Candidate id mismatch for trace group {key}.")
        proxy_rewards = [proxy_by_id[sample_id]["reward"] for sample_id in ids]
        oracle_rewards = [oracle_by_id[sample_id]["reward"] for sample_id in ids]
        if len(set(proxy_rewards)) < 2 or len(set(oracle_rewards)) < 2:
            raise ValueError(f"Trace group {key} has undefined rank correlation because all rewards are tied.")
        spearman.append(_pearson(_average_ranks(proxy_rewards), _average_ranks(oracle_rewards)))
        kendall.append(_kendall_tau_b(proxy_rewards, oracle_rewards))

        oracle_reward_by_id = {sample_id: oracle_by_id[sample_id]["reward"] for sample_id in ids}
        for kind, overlaps, gaps in (
            ("top", top_overlap, top_true_gap),
            ("bottom", bottom_overlap, bottom_true_gap),
        ):
            proxy_ids = _selected_ids(proxy_candidates, kind)
            oracle_ids = _selected_ids(oracle_candidates, kind)
            if not proxy_ids:
                continue
            overlaps.append(len(proxy_ids & oracle_ids) / len(oracle_ids))
            proxy_true = fmean(oracle_reward_by_id[sample_id] for sample_id in proxy_ids)
            oracle_true = fmean(oracle_reward_by_id[sample_id] for sample_id in oracle_ids)
            gaps.append(proxy_true - oracle_true)

    return {
        "rollout_id": rollout_id,
        "policy_version": 0,
        "groups": len(keys),
        "spearman_mean": fmean(spearman),
        "kendall_tau_b_mean": fmean(kendall),
        "top_overlap_mean": fmean(top_overlap) if top_overlap else None,
        "bottom_overlap_mean": fmean(bottom_overlap) if bottom_overlap else None,
        "top_selected_true_reward_gap": fmean(top_true_gap) if top_true_gap else None,
        "bottom_selected_true_reward_gap": fmean(bottom_true_gap) if bottom_true_gap else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proxy-dir", type=Path, required=True)
    parser.add_argument("--oracle-dir", type=Path, required=True)
    parser.add_argument("--rollout-id", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    metrics = compare(args.proxy_dir, args.oracle_dir, rollout_id=args.rollout_id)
    text = json.dumps(metrics, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{text}\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
