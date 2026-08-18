#!/usr/bin/env python3
"""Static guard: the core dependency directions that are already clean stay clean."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNIRL = ROOT / "unirl"
SKIP_PARTS = {"vendor", "__pycache__"}

UPWARD = (
    "unirl.algorithms",
    "unirl.data",
    "unirl.models",
    "unirl.reward",
    "unirl.rollout",
    "unirl.train",
    "unirl.trainer",
)

# owner -> (forbidden prefixes, allowed sub-prefixes carved back out of forbidden).
RULES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "config": (UPWARD, ()),
    "data": (UPWARD, ("unirl.data",)),
    "sde": (UPWARD, ()),
    "types": (UPWARD, ()),
    "utils": (UPWARD, ()),
    "model contracts": (UPWARD, ("unirl.models.types",)),
    "models": (
        (
            "unirl.algorithms",
            "unirl.data",
            "unirl.reward",
            "unirl.rollout",
            "unirl.trainer",
        ),
        (),
    ),
    "algorithms": (
        (
            "unirl.data",
            "unirl.models",
            "unirl.reward",
            "unirl.rollout",
            "unirl.trainer",
        ),
        ("unirl.models.types",),
    ),
    "train": (
        ("unirl.algorithms", "unirl.models", "unirl.reward", "unirl.rollout", "unirl.trainer"),
        ("unirl.algorithms.base", "unirl.models.types"),
    ),
    "distributed": (
        (
            "unirl.algorithms",
            "unirl.data",
            "unirl.models",
            "unirl.reward",
            "unirl.rollout",
            "unirl.trainer",
        ),
        (),
    ),
    "reward": (
        (
            "unirl.algorithms",
            "unirl.data",
            "unirl.models",
            "unirl.rollout",
            "unirl.train",
            "unirl.trainer",
        ),
        (),
    ),
    "rollout": (
        ("unirl.algorithms", "unirl.data", "unirl.reward", "unirl.train", "unirl.trainer"),
        (),
    ),
}


def _owner(path: Path) -> str | None:
    parts = path.relative_to(UNIRL).parts
    if parts[:2] == ("models", "types"):
        return "model contracts"
    return parts[0] if parts[0] in RULES else None


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute)
        and isinstance(test.value, ast.Name)
        and test.value.id == "typing"
        and test.attr == "TYPE_CHECKING"
    )


def _resolve_import(path: Path, node: ast.ImportFrom) -> str | None:
    if node.level == 0:
        return node.module
    package = path.relative_to(ROOT).parts[:-1]
    if node.level > len(package):
        return None
    anchor = package[: len(package) - (node.level - 1)]
    suffix = tuple(node.module.split(".")) if node.module else ()
    return ".".join((*anchor, *suffix))


def _imports(path: Path):
    """Yield ``(lineno, dotted)`` per runtime-imported name, skipping TYPE_CHECKING blocks."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return
    stack: list[ast.AST] = [tree]
    while stack:
        node = stack.pop()
        if isinstance(node, ast.If) and _is_type_checking(node.test):
            stack.extend(node.orelse)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, "unirl" if alias.asname is None and alias.name.startswith("unirl.") else alias.name
        elif isinstance(node, ast.ImportFrom):
            module = _resolve_import(path, node)
            if not module:
                continue
            # A symbol import only lengthens the dotted path, which prefix matching tolerates.
            for alias in node.names:
                yield node.lineno, module if alias.name == "*" else f"{module}.{alias.name}"
        else:
            stack.extend(ast.iter_child_nodes(node))


def _matches(module: str, prefix: str) -> bool:
    return module == prefix or module.startswith(prefix + ".")


def main() -> int:
    errors: list[str] = []
    for path in sorted(UNIRL.rglob("*.py")):
        if SKIP_PARTS.intersection(path.relative_to(ROOT).parts):
            continue
        owner = _owner(path)
        if owner is None:
            continue
        forbidden, allowed = RULES[owner]
        for lineno, module in _imports(path):
            if module == "unirl":
                errors.append(
                    f"{path.relative_to(ROOT)}:{lineno}: {owner} imports the bare 'unirl' namespace, "
                    "which reaches every loaded sibling; use `from ... import ...` or alias a dotted import with `as`"
                )
                continue
            if any(_matches(module, prefix) for prefix in allowed):
                continue
            if any(_matches(module, prefix) for prefix in forbidden):
                errors.append(
                    f"{path.relative_to(ROOT)}:{lineno}: {owner} imports {module!r}; "
                    "depend on a lower-level contract/mechanism or move the integration to its owner"
                )

    if errors:
        print("check-core-dependencies: FAILED")
        for error in errors:
            print(f"  {error}")
        return 1
    print("check-core-dependencies: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
