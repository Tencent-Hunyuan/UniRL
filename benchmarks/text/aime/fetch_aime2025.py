"""Fetch AIME 2025 into ``data/aime2025.jsonl``.

AIME 2024 (MIT-tagged upstream) is vendored next to this script; the AIME 2025
dataset card (yentinglin/aime_2025) declares no license, so that file is fetched
on demand instead of being committed: ``python benchmarks/text/aime/fetch_aime2025.py``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"


def _row_id(url: str, index: int) -> str:
    match = re.search(r"2025_AIME_(I{1,2}).*?Problem[_/](\d+)", url or "")
    return f"2025-{match.group(1)}-{match.group(2)}" if match else f"2025-{index}"


def _sort_key(record_id: str):
    match = re.fullmatch(r"2025-(I{1,2})-(\d+)", record_id)
    return (len(match.group(1)), int(match.group(2))) if match else (9, 0)


def main() -> None:
    from huggingface_hub import HfApi, hf_hub_download

    repo = "yentinglin/aime_2025"
    # The repo ships the same 30 problems both combined (data/) and split
    # (part1/ + part2/), so records are deduped by id.
    parquets = [f for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.endswith(".parquet")]
    records = {}
    for filename in sorted(parquets):
        import pyarrow.parquet as pq  # available via the repo's datasets/pyarrow stack

        table = pq.read_table(hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset"))
        for i, row in enumerate(table.to_pylist()):
            record_id = _row_id(row.get("url", ""), i)
            records.setdefault(record_id, {"id": record_id, "problem": row["problem"], "answer": str(row["answer"])})
    DATA_DIR.mkdir(exist_ok=True)
    out_path = DATA_DIR / "aime2025.jsonl"
    with open(out_path, "w") as out:
        for record_id in sorted(records, key=_sort_key):
            out.write(json.dumps(records[record_id]) + "\n")
    print(f"wrote {len(records)} problems to {out_path}")


if __name__ == "__main__":
    main()
