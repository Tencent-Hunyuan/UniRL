"""Validate fixed TTS eval schemas and aggregate trustworthy reward results.

This CPU-only tool does not load reward or generation models. It joins a fixed
eval manifest with model-scored JSONL rows and emits overall plus stratified
WER/CER/SIM/MOS, EOS/decode/anomaly rates, and reward distributions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, List, Mapping, Optional, Sequence

TTS_EVAL_MANIFEST_SCHEMA = "unirl.tts_eval_manifest.v1"
TTS_EVAL_RESULT_SCHEMA = "unirl.tts_eval_result.v1"
LENGTH_BUCKETS = frozenset({"short", "medium", "long"})
STRATIFICATION_FIELDS = ("language", "length_bucket", "contains_number", "speaker_id")


def _require(row: Mapping[str, Any], key: str, expected_type: type, *, source: str) -> Any:
    if key not in row:
        raise ValueError(f"{source}: missing required field {key!r}")
    value = row[key]
    if not isinstance(value, expected_type) or (expected_type is str and not value.strip()):
        raise TypeError(f"{source}: {key!r} must be a non-empty {expected_type.__name__}")
    return value


def validate_manifest_row(row: Mapping[str, Any], *, source: str = "manifest row") -> Dict[str, Any]:
    if row.get("schema") != TTS_EVAL_MANIFEST_SCHEMA:
        raise ValueError(f"{source}: schema must be {TTS_EVAL_MANIFEST_SCHEMA!r}")
    normalized = dict(row)
    _require(row, "sample_id", str, source=source)
    _require(row, "text", str, source=source)
    _require(row, "language", str, source=source)
    _require(row, "speaker_id", str, source=source)
    bucket = _require(row, "length_bucket", str, source=source)
    if bucket not in LENGTH_BUCKETS:
        raise ValueError(f"{source}: length_bucket must be one of {sorted(LENGTH_BUCKETS)}")
    _require(row, "contains_number", bool, source=source)
    if not row.get("ref_audio") and row.get("speaker_embedding") is None:
        raise ValueError(f"{source}: ref_audio or speaker_embedding is required for speaker SIM")
    return normalized


def validate_result_row(row: Mapping[str, Any], *, source: str = "result row") -> Dict[str, Any]:
    if row.get("schema") != TTS_EVAL_RESULT_SCHEMA:
        raise ValueError(f"{source}: schema must be {TTS_EVAL_RESULT_SCHEMA!r}")
    normalized = dict(row)
    _require(row, "sample_id", str, source=source)
    for key in ("has_eos", "decode_failure", "anomaly"):
        _require(row, key, bool, source=source)
    for key in ("wer", "cer", "sim", "mos", "reward"):
        value = row.get(key)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))
        ):
            raise TypeError(f"{source}: {key!r} must be finite numeric or null")
    return normalized


def read_jsonl(path: str, validator) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number}: expected JSON object")
            rows.append(validator(value, source=f"{path}:{line_number}"))
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def _numeric_summary(rows: Sequence[Mapping[str, Any]], key: str) -> Dict[str, Any]:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return {"status": "unavailable", "n": 0, "mean": None}
    ordered = sorted(values)

    def quantile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "status": "available",
        "n": len(values),
        "mean": mean(values),
        "std": pstdev(values),
        "min": ordered[0],
        "p10": quantile(0.10),
        "p50": quantile(0.50),
        "p90": quantile(0.90),
        "max": ordered[-1],
    }


def _summary(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    count = len(rows)
    return {
        "n": count,
        "wer": _numeric_summary(rows, "wer"),
        "cer": _numeric_summary(rows, "cer"),
        "sim": _numeric_summary(rows, "sim"),
        "mos": _numeric_summary(rows, "mos"),
        "reward_distribution": _numeric_summary(rows, "reward"),
        "eos_failure_rate": sum(not bool(row["has_eos"]) for row in rows) / count,
        "decode_failure_rate": sum(bool(row["decode_failure"]) for row in rows) / count,
        "anomaly_rate": sum(bool(row["anomaly"]) for row in rows) / count,
    }


def aggregate_tts_eval(
    manifest_rows: Sequence[Mapping[str, Any]],
    result_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    manifest_by_id: Dict[str, Mapping[str, Any]] = {}
    for row in manifest_rows:
        sample_id = str(row["sample_id"])
        if sample_id in manifest_by_id:
            raise ValueError(f"duplicate manifest sample_id {sample_id!r}")
        manifest_by_id[sample_id] = row
    result_by_id: Dict[str, Mapping[str, Any]] = {}
    for row in result_rows:
        sample_id = str(row["sample_id"])
        if sample_id in result_by_id:
            raise ValueError(f"duplicate result sample_id {sample_id!r}")
        result_by_id[sample_id] = row
    missing = sorted(set(manifest_by_id) - set(result_by_id))
    extra = sorted(set(result_by_id) - set(manifest_by_id))
    if missing or extra:
        raise ValueError(f"manifest/result sample mismatch: missing={missing[:5]}, extra={extra[:5]}")

    joined: List[Dict[str, Any]] = []
    for sample_id, manifest in manifest_by_id.items():
        joined.append({**manifest, **result_by_id[sample_id], "sample_id": sample_id})

    strata: Dict[str, Dict[str, Any]] = {}
    for field in STRATIFICATION_FIELDS:
        groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for row in joined:
            groups[str(row[field]).lower() if isinstance(row[field], bool) else str(row[field])].append(row)
        strata[field] = {key: _summary(values) for key, values in sorted(groups.items())}
    return {
        "schema": "unirl.tts_eval_report.v1",
        "overall": _summary(joined),
        "stratified": strata,
        "rows": joined,
    }


def _atomic_write_json(path: str, payload: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    manifest = read_jsonl(args.manifest, validate_manifest_row)
    results = read_jsonl(args.results, validate_result_row)
    report = aggregate_tts_eval(manifest, results)
    _atomic_write_json(args.output, report)
    print(json.dumps({"n": report["overall"]["n"], "output": args.output}, ensure_ascii=False))


if __name__ == "__main__":
    main()


__all__ = [
    "LENGTH_BUCKETS",
    "STRATIFICATION_FIELDS",
    "TTS_EVAL_MANIFEST_SCHEMA",
    "TTS_EVAL_RESULT_SCHEMA",
    "aggregate_tts_eval",
    "read_jsonl",
    "validate_manifest_row",
    "validate_result_row",
]
