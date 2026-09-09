#!/usr/bin/env python3
"""Static guard: every ``_target_`` in a recipe must resolve to a real symbol."""

from __future__ import annotations

import ast
import fnmatch
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCAN_DIRS = ["examples", "experimental", "CPPO", "DRPO", "FlowDPPO", "unirl"]
SKIP_PARTS = {".git", "vendor"}
PACKAGES = ("unirl", "experimental")

_TARGET_RE = re.compile(r"""^\s*_target_:\s*['"]?((?:%s)\.[A-Za-z0-9_.]+)['"]?\s*$""" % "|".join(PACKAGES))

_READ_THREADS = 32


def _scan() -> tuple[dict[str, Path], list[Path]]:
    """One walk over ``SCAN_DIRS``: dotted module path -> file, plus every recipe file."""
    modules: dict[str, Path] = {}
    packages: dict[str, Path] = {}
    recipes: list[Path] = []
    for d in SCAN_DIRS:
        for dirpath, _dirnames, filenames in os.walk(ROOT / d):
            parts = Path(dirpath).relative_to(ROOT).parts
            importable = parts[0] in PACKAGES and all(p.isidentifier() for p in parts)
            skipped = bool(SKIP_PARTS & set(parts))
            for name in filenames:
                path = Path(dirpath, name)
                if importable and name.endswith(".py"):
                    stem = name[:-3]
                    if stem == "__init__":
                        packages[".".join(parts)] = path
                        packages[".".join((*parts, stem))] = path
                    elif stem.isidentifier():
                        modules[".".join((*parts, stem))] = path
                elif not skipped and fnmatch.fnmatch(name, "*.y*ml"):
                    recipes.append(path)
    return {**packages, **modules}, sorted(recipes)


def _read_all(paths: list[Path]) -> list[str]:
    with ThreadPoolExecutor(max_workers=_READ_THREADS) as pool:
        return list(pool.map(lambda p: p.read_text(encoding="utf-8"), paths))


def _top_level_names(source: str, path: Path) -> frozenset[str] | None:
    """Top-level names bound in ``source`` (class/func/assign/import), or None."""
    try:
        tree = ast.parse(source, filename=path)
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
    """Longest module prefix of ``dotted`` that exists, with the attribute after it."""
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
    needed = sorted({hit[0] for hit in module_split.values() if hit is not None})
    top_level = {path: _top_level_names(source, path) for path, source in zip(needed, _read_all(needed))}

    failures: list[str] = []
    tier: list[str] = []
    for path, lineno, dotted in targets:
        hit = module_split[dotted]
        names = top_level[hit[0]] if hit is not None else None
        if names is None or hit[1] not in names:
            failures.append(f"{path.relative_to(ROOT)}:{lineno}: unresolved _target_ '{dotted}'")
        if dotted.startswith("experimental."):
            rel = path.relative_to(ROOT)
            if rel.parts[0] != "experimental":
                tier.append(
                    f"{rel}:{lineno}: core recipe targets '{dotted}' — the dependency arrow points the other way"
                )
            elif len(rel.parts) == 2:
                tier.append(
                    f"{rel}:{lineno}: recipe at the experimental/ root targets '{dotted}' — "
                    "a recipe belongs inside one package (experimental/<name>/...)"
                )
            elif dotted.split(".")[1] != rel.parts[1]:
                tier.append(
                    f"{rel}:{lineno}: recipe under 'experimental/{rel.parts[1]}/' targets sibling '{dotted}' — "
                    "shared code graduates into core"
                )

    if failures:
        print("Unresolved recipe _target_ paths (rename leftover or typo):", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
    if tier:
        print("Experimental-tier direction violations:", file=sys.stderr)
        for f in tier:
            print(f"  {f}", file=sys.stderr)
    if failures or tier:
        return 1
    print(f"check-recipe-targets: {len(targets)} recipe _target_ paths resolve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
