"""CPU tests for the generic SFT JSONL data source."""

import json

from unirl.train.sft.data import JsonlSFTDataSource


def _write_manifest(path, n):
    with open(path, "w") as fh:
        for i in range(n):
            fh.write(json.dumps({"sample_id": f"s{i}", "instruction": "t", "frames_path": f"frames/s{i}.pt"}) + "\n")


def test_epoch_cycling_and_root_injection(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, 5)
    src = JsonlSFTDataSource(str(manifest), seed=0, shuffle=True)

    batch = src.get_samples(3)
    assert len(batch) == 3
    assert all(record["_root"] == str(tmp_path) for record in batch)

    # 5 records, batches of 3: the second batch crosses the epoch boundary.
    seen = {record["sample_id"] for record in batch}
    seen |= {record["sample_id"] for record in src.get_samples(3)}
    assert seen == {f"s{i}" for i in range(5)}


def test_shuffle_determinism(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, 8)
    a = JsonlSFTDataSource(str(manifest), seed=1)
    b = JsonlSFTDataSource(str(manifest), seed=1)
    ids = lambda source: [record["sample_id"] for record in source.get_samples(8)]  # noqa: E731
    assert ids(a) == ids(b)


def test_eval_samples_fallback(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, 4)
    src = JsonlSFTDataSource(str(manifest), seed=0)
    assert len(src.eval_samples(2)) == 2  # falls back to train records

    eval_manifest = tmp_path / "eval.jsonl"
    _write_manifest(eval_manifest, 2)
    src = JsonlSFTDataSource(str(manifest), eval_manifest_path=str(eval_manifest), seed=0)
    assert [r["sample_id"] for r in src.eval_samples(2)] == ["s0", "s1"]
