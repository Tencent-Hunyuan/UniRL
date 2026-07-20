"""Merge ``summary.json`` files under a results dir into per-benchmark markdown tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def collect(out_dir: Path) -> List[Dict]:
    return [json.loads(p.read_text()) for p in sorted(out_dir.glob("*/*/summary.json"))]


def render(summaries: List[Dict]) -> str:
    by_benchmark: Dict[str, List[Dict]] = {}
    for s in summaries:
        by_benchmark.setdefault(s["benchmark"], []).append(s)
    lines: List[str] = []
    for benchmark in sorted(by_benchmark):
        rows = by_benchmark[benchmark]
        metric_keys = sorted({k for r in rows for k in r["metrics"]})
        lines.append(f"### {benchmark}")
        lines.append("| ckpt | " + " | ".join(metric_keys) + " | prompts×k | errors |")
        lines.append("|---|" + "---|" * (len(metric_keys) + 2))
        for r in sorted(rows, key=lambda r: r["ckpt"]):
            cells = [f"{r['metrics'].get(k):.4f}" if r["metrics"].get(k) is not None else "—" for k in metric_keys]
            subset = "*" if r.get("subset") else ""
            lines.append(
                f"| {r['ckpt']}{subset} | "
                + " | ".join(cells)
                + f" | {r['n_prompts']}×{r['samples_per_prompt']} | {r['n_errors']} |"
            )
        lines.append("")
    if any(r.get("subset") for r in summaries):
        lines.append("`*` = ran on a `--num-prompts` subset, not the full protocol.")
    return "\n".join(lines)


def main(out_dir: Path) -> None:
    summaries = collect(out_dir)
    if not summaries:
        raise SystemExit(f"no summary.json found under {out_dir}")
    print(render(summaries))
