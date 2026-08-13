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

# Each owner lists forbidden namespace prefixes. A rule may allow narrower
# leaves, such as algorithms/train depending on model Protocols but not a
# concrete model family.
RULES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "data": (UPWARD, ("unirl.data",)),
    "types": (UPWARD, ("unirl.types",)),
    "sde": (UPWARD, ("unirl.sde",)),
    "model contracts": (UPWARD, ("unirl.models.types",)),
    "models": (
        (
            "unirl.algorithms",
            "unirl.data",
            "unirl.reward",
            "unirl.rollout",
            "unirl.trainer",
        ),
        ("unirl.models",),
    ),
    "algorithms": (
        (
            "unirl.data",
            "unirl.models",
            "unirl.reward",
            "unirl.rollout",
            "unirl.trainer",
        ),
        ("unirl.algorithms", "unirl.models.types"),
    ),
    "train": (
        ("unirl.models", "unirl.reward", "unirl.rollout", "unirl.trainer"),
        ("unirl.models.types", "unirl.train"),
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
        ("unirl.distributed",),
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
        ("unirl.reward",),
    ),
    "rollout": (
        ("unirl.algorithms", "unirl.data", "unirl.reward", "unirl.train", "unirl.trainer"),
        ("unirl.rollout",),
    ),
}


def _owner(path: Path) -> str | None:
    parts = path.relative_to(UNIRL).parts
    if len(parts) >= 2 and parts[:2] == ("models", "types"):
        return "model contracts"
    if not parts:
        return None
    return parts[0] if parts[0] in RULES else None


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
    """Yield ``(lineno, dotted)`` per imported name; ``from x import y`` yields ``x.y``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            module = _resolve_import(path, node)
            if not module:
                continue
            # `from unirl import trainer` binds `unirl.trainer`; a symbol import only
            # lengthens the path, which prefix matching already tolerates.
            for alias in node.names:
                yield node.lineno, module if alias.name == "*" else f"{module}.{alias.name}"


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
