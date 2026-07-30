#!/usr/bin/env python3
"""Static guard: every ``_target_`` in a recipe must resolve to a real symbol.

Renames (e.g. ``unirl.algorithms.ar_grpo.ARGRPO`` -> ``unirl.algorithms.grpo.GRPO``)
break Hydra ``instantiate`` only at *runtime*, and this repo's CI is lint-only, so a
stale dotted path can merge silently. This check parses every recipe ``_target_:``
pointing into the ``unirl`` package and confirms the module file and the attribute
exist — purely via ``ast``, importing nothing (no torch/vllm/sglang needed).

Only ~0.2s of the runtime is parsing; the rest is filesystem latency, which dominates
when the checkout lives on a network filesystem. So the tree is walked once to index
every module (rather than probing candidate paths per target, which costs ~10k stat
calls) and file contents are read through a thread pool. On CephFS that is the
difference between ~5min and ~15s per run.

Run by the ``check-recipe-targets`` pre-commit hook (so it rides the existing
``pre-commit run --all-files`` lint CI). Exits non-zero, listing each unresolved
target, when any path is dead.
"""

from __future__ import annotations

import ast
import fnmatch
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# YAML trees that hold recipes / stage configs with ``_target_`` entries.
SCAN_DIRS = ["examples", "CPPO", "DRPO", "FlowDPPO", "unirl"]
# Vendored / sub-project trees kept byte-pristine (mirror .pre-commit-config exclude).
SKIP_PARTS = {".git", "vendor"}
# The only package ``_TARGET_RE`` accepts, so the only one worth indexing.
PACKAGE = "unirl"

_TARGET_RE = re.compile(r"""^\s*_target_:\s*['"]?(unirl\.[A-Za-z0-9_.]+)['"]?\s*$""")

# Deep enough to hide network-filesystem round trips behind each other.
_READ_THREADS = 32


def _scan() -> tuple[dict[str, Path], list[Path]]:
    """One walk over ``SCAN_DIRS``: dotted module path -> file, plus every recipe file."""
    modules: dict[str, Path] = {}
    packages: dict[str, Path] = {}
    recipes: list[Path] = []
    for d in SCAN_DIRS:
        for dirpath, _dirnames, filenames in os.walk(ROOT / d):
            parts = Path(dirpath).relative_to(ROOT).parts
            skipped = bool(SKIP_PARTS & set(parts))
            for name in filenames:
                if parts[0] == PACKAGE and name.endswith(".py"):
                    if name == "__init__.py":
                        packages[".".join(parts)] = Path(dirpath, name)
                    else:
                        modules[".".join((*parts, name[:-3]))] = Path(dirpath, name)
                elif not skipped and fnmatch.fnmatch(name, "*.y*ml"):
                    recipes.append(Path(dirpath, name))
    # A module file shadows a package of the same name, as in Python's own lookup.
    return {**packages, **modules}, sorted(recipes)


def _read_all(paths: list[Path]) -> list[str]:
    with ThreadPoolExecutor(max_workers=_READ_THREADS) as pool:
        return list(pool.map(lambda p: p.read_text(encoding="utf-8"), paths))


def _top_level_names(source: str, path: Path) -> frozenset[str] | None:
    """Top-level names bound in ``source`` (class/func/assign/import), or None."""
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
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


def _split_module(dotted: str, modules: dict[str, Path]) -> tuple[Path, str] | None:
    """Longest module prefix of ``dotted`` that exists, with the attribute after it.

    Walks the standard Python split: try module = all-but-last part, attr = last;
    if that module does not exist, fold trailing parts back into the attribute chain
    until one does. None when no prefix names a module at all.
    """
    parts = dotted.split(".")
    for split in range(len(parts) - 1, 0, -1):
        module_file = modules.get(".".join(parts[:split]))
        if module_file is not None:
            return module_file, parts[split]
    return None


def main() -> int:
    modules, recipes = _scan()

    targets: list[tuple[Path, int, str]] = []
    for path, text in zip(recipes, _read_all(recipes)):
        for lineno, line in enumerate(text.splitlines(), 1):
            m = _TARGET_RE.match(line)
            if m:
                targets.append((path, lineno, m.group(1)))

    module_split = {dotted: _split_module(dotted, modules) for _, _, dotted in targets}
    # Only the modules a recipe actually names are worth reading and parsing.
    needed = sorted({hit[0] for hit in module_split.values() if hit is not None})
    top_level = {path: _top_level_names(source, path) for path, source in zip(needed, _read_all(needed))}

    failures: list[str] = []
    for path, lineno, dotted in targets:
        hit = module_split[dotted]
        names = top_level[hit[0]] if hit is not None else None
        if names is None or hit[1] not in names:
            failures.append(f"{path.relative_to(ROOT)}:{lineno}: unresolved _target_ '{dotted}'")

    if failures:
        print("Unresolved recipe _target_ paths (rename leftover or typo):", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1
    print(f"check-recipe-targets: {len(targets)} unirl _target_ paths resolve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
