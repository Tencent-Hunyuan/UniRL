#!/usr/bin/env python3
"""Static guard: every docstring is one line, within the ruff line-length."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {".git", "vendor", "__pycache__"}
SCAN_DIRS = ("benchmarks", "datasets", "examples", "experimental", "lint", "CPPO", "DRPO", "FlowDPPO", "unirl")
MAX_LINE = 120  # matches [tool.ruff] line-length

_DOC_OWNERS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)

_MOVE_IT = (
    "move upstream-divergence records, numerical constraints, and implementer contracts to the nearest README.md; "
    "keep runtime-only facts (shapes, dtypes, ranges) — in the one line as ``[B, C, H, W]``, or as a trailing "
    "# annotation"
)


def _py_files():
    for name in SCAN_DIRS:
        base = ROOT / name
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*.py")):
            if not SKIP_PARTS.intersection(path.relative_to(ROOT).parts):
                yield path
    for path in sorted(ROOT.glob("*.py")):
        yield path


def _docstrings(tree: ast.AST):
    """Yield the string ``Constant`` node each module/class/function opens with."""
    for node in ast.walk(tree):
        if not isinstance(node, _DOC_OWNERS) or not node.body:
            continue
        first = node.body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
            yield first.value


def check_docstrings(errors: list[str]) -> int:
    total = 0
    for path in _py_files():
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except (OSError, SyntaxError):
            continue
        lines = source.splitlines()
        rel = path.relative_to(ROOT)
        for node in _docstrings(tree):
            total += 1
            if not node.value.strip():
                errors.append(f"{rel}:{node.lineno}: empty docstring — write one line or delete it")
                continue
            span = node.end_lineno - node.lineno + 1
            if span > 1:
                errors.append(f"{rel}:{node.lineno}: docstring spans {span} lines — one line only; {_MOVE_IT}")
                continue
            width = len(lines[node.lineno - 1])
            if width > MAX_LINE:
                errors.append(
                    f"{rel}:{node.lineno}: docstring line is {width} chars (max {MAX_LINE}) — "
                    f"shorten the summary; detail belongs in the nearest README.md"
                )
    return total


def main() -> int:
    errors: list[str] = []
    total = check_docstrings(errors)
    if errors:
        print("check-docstring-lines: FAILED")
        for err in errors:
            print(f"  {err}")
        print(f"  {len(errors)} violation(s). There is deliberately no allowlist.")
        return 1
    print(f"check-docstring-lines: {total} docstrings, all one line.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
