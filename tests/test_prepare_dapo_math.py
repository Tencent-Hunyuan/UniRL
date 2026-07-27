from __future__ import annotations

import json
import sys
from types import SimpleNamespace

from unirl.utils.prepare_dapo_math import _convert


def _install_dataset(monkeypatch, rows: list[dict]) -> None:
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=lambda *_args, **_kwargs: rows),
    )


def test_convert_deduplicates_replicated_source_ids(tmp_path, monkeypatch) -> None:
    rows = [
        {
            "prompt": "same text",
            "reward_model": {"ground_truth": "1"},
            "extra_info": {"index": "source-a"},
        },
        {
            "prompt": "same text",
            "reward_model": {"ground_truth": "1"},
            "extra_info": {"index": "source-b"},
        },
        {
            "prompt": "same text",
            "reward_model": {"ground_truth": "1"},
            "extra_info": {"index": "source-a"},
        },
    ]
    _install_dataset(monkeypatch, rows)
    output = tmp_path / "train.jsonl"

    assert _convert("dataset", "train", str(output), append_boxed=False) == 2
    assert len(output.read_text().splitlines()) == 2


def test_convert_deduplicates_identical_schema_pruned_records(tmp_path, monkeypatch) -> None:
    rows = [
        {"prompt": "problem", "reward_model": {"ground_truth": "42"}},
        {"prompt": "problem", "reward_model": {"ground_truth": "42"}},
        {"prompt": "problem", "reward_model": {"ground_truth": "43"}},
    ]
    _install_dataset(monkeypatch, rows)
    output = tmp_path / "train.jsonl"

    assert _convert("dataset", "train", str(output), append_boxed=False) == 2
    converted = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["metadata"]["answer"] for row in converted] == ["42", "43"]
