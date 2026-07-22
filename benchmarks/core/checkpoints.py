"""Resolve --ckpt / --lora into a base model + optional PEFT adapter dir."""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ResolvedCkpt:
    base: str  # HF repo id or local path, loadable by from_pretrained
    adapter: Optional[str]  # PEFT adapter dir (adapter_config.json), or None
    tag: str  # results directory name


def _slug(name: str) -> str:
    # Local paths keep their last two components: UniRL checkpoints are conventionally
    # named checkpoint-<step>, so basename alone collides across training runs — and a
    # colliding tag would silently resume/rescore another run's outputs.
    parts = Path(name).parts if Path(name).exists() else name.rstrip("/").split("/")[-1:]
    return re.sub(r"[^A-Za-z0-9._-]+", "-", "-".join(parts[-2:]).lstrip("-/")) or "ckpt"


def make_tag(ckpt: str, lora: Optional[str]) -> str:
    return f"{_slug(ckpt)}+{_slug(lora)}" if lora else _slug(ckpt)


def resolve(ckpt: str, lora: Optional[str], work_dir: Path) -> ResolvedCkpt:
    """``ckpt`` is the base model (HF id or local path). ``lora`` is either a PEFT
    adapter dir or a UniRL checkpoint (``checkpoint-<step>`` dir / ``checkpoint.pt``);
    the latter is exported once to ``work_dir/<tag>/adapter`` via
    ``unirl.tools.export_adapter`` (works on both adapter- and full-mode saves).
    A ``.source`` marker pins the export to its origin checkpoint, so a tag collision
    fails loudly instead of silently evaluating another run's adapter.
    """
    tag = make_tag(ckpt, lora)
    if not lora:
        return ResolvedCkpt(base=ckpt, adapter=None, tag=tag)
    if (Path(lora) / "adapter_config.json").is_file():
        return ResolvedCkpt(base=ckpt, adapter=lora, tag=tag)
    exported = work_dir / tag / "adapter"
    marker = exported / ".source"
    source = str(Path(lora).resolve())
    if (exported / "adapter_config.json").is_file():
        recorded = marker.read_text().strip() if marker.is_file() else "<unknown>"
        if recorded != source:
            raise SystemExit(
                f"tag {tag!r} already holds an adapter exported from {recorded}, not {source} — "
                f"pass a distinct --tag (or remove {exported})"
            )
    else:
        print(f"[ckpt] exporting UniRL checkpoint {lora} -> {exported}")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "unirl.tools.export_adapter",
                "--checkpoint",
                lora,
                "--base",
                ckpt,
                "--output",
                str(exported),
            ],
            check=True,
        )
        marker.write_text(source + "\n")
    return ResolvedCkpt(base=ckpt, adapter=str(exported), tag=tag)
