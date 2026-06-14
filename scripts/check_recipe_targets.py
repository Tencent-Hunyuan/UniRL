#!/usr/bin/env python3
"""Static guard: every ``_target_`` in a recipe must resolve to a real symbol.

Renames (e.g. ``unirl.algorithms.ar_grpo.ARGRPO`` -> ``unirl.algorithms.grpo.GRPO``)
break Hydra ``instantiate`` only at *runtime*, and this repo's CI is lint-only, so a
stale dotted path can merge silently. This check parses every recipe ``_target_:``
pointing into the ``unirl`` package and confirms the module file and the attribute
exist — purely via ``ast``, importing nothing (no torch/vllm/sglang needed).

Run by the ``check-recipe-targets`` pre-commit hook (so it rides the existing
``pre-commit run --all-files`` lint CI). Exits non-zero, listing each unresolved
target, when any path is dead.
"""

from __future__ import annotations

import ast
import re
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# YAML trees that hold recipes / stage configs with ``_target_`` entries.
SCAN_DIRS = ["examples", "CPPO", "DRPO", "FlowDPPO", "unirl"]
# Vendored / sub-project trees kept byte-pristine (mirror .pre-commit-config exclude).
SKIP_PARTS = {".git", "vendor"}

_TARGET_RE = re.compile(r"""^\s*_target_:\s*['"]?(unirl\.[A-Za-z0-9_.]+)['"]?\s*$""")


@lru_cache(maxsize=None)
def _module_top_level_names(module_file: Path) -> frozenset[str] | None:
    """Top-level names bound in ``module_file`` (class/func/assign/import), or None."""
    try:
        tree = ast.parse(module_file.read_text(encoding="utf-8"), filename=str(module_file))
    except (OSError, SyntaxError):
        return None
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
    return frozenset(names)


def _resolve(dotted: str) -> bool:
    """True if ``dotted`` (e.g. unirl.algorithms.grpo.GRPO) names a real module attr.

    Walks the standard Python split: try module = all-but-last part, attr = last;
    if that module file is missing, fold trailing parts back into the attribute chain
    until a module file exists, then check the first attribute after it is top-level.
    """
    parts = dotted.split(".")
    for split in range(len(parts) - 1, 0, -1):
        mod_parts, attr_parts = parts[:split], parts[split:]
        base = ROOT.joinpath(*mod_parts)
        module_file = base.with_suffix(".py")
        if not module_file.is_file():
            module_file = base / "__init__.py"
        if not module_file.is_file():
            continue  # not a module here — fold one more part into the attr chain
        names = _module_top_level_names(module_file)
        return names is not None and attr_parts[0] in names
    return False


def main() -> int:
    failures: list[str] = []
    checked = 0
    for d in SCAN_DIRS:
        for path in sorted((ROOT / d).rglob("*.y*ml")):
            if SKIP_PARTS & set(path.relative_to(ROOT).parts):
                continue
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                m = _TARGET_RE.match(line)
                if not m:
                    continue
                checked += 1
                if not _resolve(m.group(1)):
                    rel = path.relative_to(ROOT)
                    failures.append(f"{rel}:{lineno}: unresolved _target_ '{m.group(1)}'")

    if failures:
        print("Unresolved recipe _target_ paths (rename leftover or typo):", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1
    print(f"check-recipe-targets: {checked} unirl _target_ paths resolve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
