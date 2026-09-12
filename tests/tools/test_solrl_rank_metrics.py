from __future__ import annotations

import json
from pathlib import Path

import pytest

from unirl.tools.solrl_rank_metrics import compare

MODEL_HASH = "a" * 64
PIPELINE_HASH = "f" * 64
PROMPT_HASH = "b" * 64
NOISE_HASH = "c" * 64
REWARD_HASH = "d" * 64
COMPARISON_HASH = "e" * 64


def _write_trace(path: Path, rewards: list[float], *, policy_version: int = 0) -> None:
    path.mkdir()
    candidates = []
    for index, reward in enumerate(rewards):
        selection = "top" if index == 3 else ("bottom" if index == 0 else None)
        candidates.append(
            {
                "sample_id": f"p0/{index}",
                "reward": reward,
                "selection": selection,
            }
        )
    (path / "rollout_000000.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "rollout_id": 0,
                "policy_version": policy_version,
                "policy_snapshot_id": "immutable-checkpoint-v1",
                "model_fingerprint": MODEL_HASH,
                "pipeline_fingerprint": PIPELINE_HASH,
                "reward_fingerprint": REWARD_HASH,
                "comparison_fingerprint": COMPARISON_HASH,
                "top_k": 1,
                "bottom_k": 1,
                "groups": [
                    {
                        "group_id": "p0",
                        "prompt_sha256": PROMPT_HASH,
                        "noise_sha256": NOISE_HASH,
                        "candidates": candidates,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_identical_traces_have_perfect_rank_metrics(tmp_path: Path) -> None:
    proxy, oracle = tmp_path / "proxy", tmp_path / "oracle"
    _write_trace(proxy, [0.0, 0.2, 0.4, 0.6])
    _write_trace(oracle, [0.0, 0.2, 0.4, 0.6])
    metrics = compare(proxy, oracle)
    assert metrics["spearman_mean"] == 1.0
    assert metrics["kendall_tau_b_mean"] == 1.0
    assert metrics["top_overlap_mean"] == 1.0
    assert metrics["bottom_overlap_mean"] == 1.0
    assert metrics["top_selected_true_reward_gap"] == 0.0


def test_trace_group_mismatch_is_rejected(tmp_path: Path) -> None:
    proxy, oracle = tmp_path / "proxy", tmp_path / "oracle"
    _write_trace(proxy, [0.0, 0.2, 0.4, 0.6])
    _write_trace(oracle, [0.0, 0.2, 0.4, 0.6])
    extra = json.loads((oracle / "rollout_000000.json").read_text(encoding="utf-8"))
    second = dict(extra["groups"][0])
    second["group_id"] = "p1"
    second["prompt_sha256"] = "f" * 64
    second["noise_sha256"] = "0" * 64
    second["candidates"] = [
        {**candidate, "sample_id": candidate["sample_id"].replace("p0/", "p1/")} for candidate in second["candidates"]
    ]
    extra["groups"].append(second)
    (oracle / "rollout_000000.json").write_text(json.dumps(extra), encoding="utf-8")

    try:
        compare(proxy, oracle)
    except ValueError as exc:
        assert "Trace groups differ" in str(exc)
    else:
        raise AssertionError("expected incomplete trace comparison to fail")


def test_duplicate_trace_group_is_rejected(tmp_path: Path) -> None:
    proxy, oracle = tmp_path / "proxy", tmp_path / "oracle"
    _write_trace(proxy, [0.0, 0.2, 0.4, 0.6])
    _write_trace(oracle, [0.0, 0.2, 0.4, 0.6])
    duplicate = (proxy / "rollout_000000.json").read_text(encoding="utf-8")
    (proxy / "rollout_000001.json").write_text(duplicate, encoding="utf-8")

    try:
        compare(proxy, oracle)
    except ValueError as exc:
        assert "Duplicate trace group" in str(exc)
    else:
        raise AssertionError("expected duplicate trace group to fail")


def test_one_sided_selection_reports_null_for_missing_side(tmp_path: Path) -> None:
    proxy, oracle = tmp_path / "proxy", tmp_path / "oracle"
    _write_trace(proxy, [0.0, 0.2, 0.4, 0.6])
    _write_trace(oracle, [0.0, 0.2, 0.4, 0.6])
    for path in (proxy / "rollout_000000.json", oracle / "rollout_000000.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for candidate in payload["groups"][0]["candidates"]:
            if candidate["selection"] == "bottom":
                candidate["selection"] = None
        payload["bottom_k"] = 0
        path.write_text(json.dumps(payload), encoding="utf-8")

    metrics = compare(proxy, oracle)
    assert metrics["top_overlap_mean"] == 1.0
    assert metrics["bottom_overlap_mean"] is None
    assert metrics["bottom_selected_true_reward_gap"] is None


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("reward", float("nan"), "non-finite reward"),
        ("selection", "upper", "unknown selection"),
    ],
)
def test_invalid_candidate_trace_is_rejected(tmp_path: Path, field: str, value: object, message: str) -> None:
    proxy, oracle = tmp_path / "proxy", tmp_path / "oracle"
    _write_trace(proxy, [0.0, 0.2, 0.4, 0.6])
    _write_trace(oracle, [0.0, 0.2, 0.4, 0.6])
    payload = json.loads((proxy / "rollout_000000.json").read_text(encoding="utf-8"))
    payload["groups"][0]["candidates"][1][field] = value
    (proxy / "rollout_000000.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        compare(proxy, oracle)


def test_trace_identity_and_later_policy_are_rejected(tmp_path: Path) -> None:
    proxy, oracle = tmp_path / "proxy", tmp_path / "oracle"
    _write_trace(proxy, [0.0, 0.2, 0.4, 0.6])
    _write_trace(oracle, [0.0, 0.2, 0.4, 0.6])
    payload = json.loads((oracle / "rollout_000000.json").read_text(encoding="utf-8"))
    payload["groups"][0]["noise_sha256"] = "1" * 64
    (oracle / "rollout_000000.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="noise_sha256 differs"):
        compare(proxy, oracle)

    with pytest.raises(ValueError, match="Only rollout_id=0"):
        compare(proxy, oracle, rollout_id=1)


def test_incorrect_selection_and_degenerate_ranks_are_rejected(tmp_path: Path) -> None:
    proxy, oracle = tmp_path / "proxy", tmp_path / "oracle"
    _write_trace(proxy, [0.0, 0.2, 0.4, 0.6])
    _write_trace(oracle, [0.0, 0.2, 0.4, 0.6])
    payload = json.loads((proxy / "rollout_000000.json").read_text(encoding="utf-8"))
    candidates = payload["groups"][0]["candidates"]
    candidates[2]["selection"] = "top"
    candidates[3]["selection"] = None
    (proxy / "rollout_000000.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="do not match its recorded rewards"):
        compare(proxy, oracle)

    tied_proxy, tied_oracle = tmp_path / "tied-proxy", tmp_path / "tied-oracle"
    _write_trace(tied_proxy, [1.0, 1.0, 1.0, 1.0])
    _write_trace(tied_oracle, [1.0, 1.0, 1.0, 1.0])
    for path in (tied_proxy / "rollout_000000.json", tied_oracle / "rollout_000000.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        candidates = payload["groups"][0]["candidates"]
        for candidate in candidates:
            candidate["selection"] = None
        candidates[0]["selection"] = "top"
        candidates[1]["selection"] = "bottom"
        path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="undefined rank correlation"):
        compare(tied_proxy, tied_oracle)


def test_candidate_order_mismatch_is_rejected(tmp_path: Path) -> None:
    proxy, oracle = tmp_path / "proxy", tmp_path / "oracle"
    _write_trace(proxy, [0.0, 0.2, 0.4, 0.6])
    _write_trace(oracle, [0.0, 0.2, 0.4, 0.6])
    payload = json.loads((oracle / "rollout_000000.json").read_text(encoding="utf-8"))
    candidates = payload["groups"][0]["candidates"]
    candidates[1], candidates[2] = candidates[2], candidates[1]
    (oracle / "rollout_000000.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Candidate id order differs"):
        compare(proxy, oracle)
