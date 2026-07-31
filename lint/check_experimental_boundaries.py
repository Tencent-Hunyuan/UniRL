#!/usr/bin/env python3
"""Static guard for the experimental-tier boundaries.

Three rules, all enforceable without importing anything (pure ``ast`` +
text, same spirit as ``check_recipe_targets.py``):

1. **Core never imports experimental.** ``unirl/`` must not reference the
   ``experimental`` namespace — the dependency arrow points one way.
2. **Experimental packages never import each other.** ``experimental/<a>``
   may import ``unirl`` and its own package only; sharing code across
   packages means it is ready to graduate into core, not ready to be
   borrowed sideways.
3. **Recipe requirements are additive-only.** Reward and actor share one
   Python process, so an ``experimental/**/requirements.txt`` cannot
   version-"isolate" anything: any requirement whose (normalized) name is
   already declared in ``pyproject.toml`` core dependencies would mutate
   the locked core stack on install and is rejected.

Run by the ``check-experimental-boundaries`` pre-commit hook. Exits
non-zero listing each violation.
"""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {".git", "vendor", "__pycache__"}

_REQ_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _py_files(base: Path):
    for path in sorted(base.rglob("*.py")):
        if not SKIP_PARTS.intersection(path.parts):
            yield path


def _imported_roots(path: Path):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module


def check_core_does_not_import_experimental(errors: list[str]) -> None:
    for path in _py_files(ROOT / "unirl"):
        for module in _imported_roots(path):
            if module == "experimental" or module.startswith("experimental."):
                errors.append(f"{path.relative_to(ROOT)}: core imports {module!r} — the arrow points the other way")


def check_no_cross_package_imports(errors: list[str]) -> None:
    exp = ROOT / "experimental"
    if not exp.is_dir():
        return
    for path in _py_files(exp):
        rel = path.relative_to(exp)
        own = rel.parts[0] if len(rel.parts) > 1 else None
        for module in _imported_roots(path):
            if not module.startswith("experimental."):
                continue
            target = module.split(".")[1] if "." in module else None
            if target and own and target != own:
                errors.append(
                    f"{path.relative_to(ROOT)}: imports {module!r} from a sibling package — "
                    "shared code graduates into core, it is not borrowed sideways"
                )


def check_requirements_additive_only(errors: list[str]) -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    core = {_normalize(_REQ_NAME_RE.match(dep).group(1)) for dep in pyproject["project"]["dependencies"]}
    exp = ROOT / "experimental"
    if not exp.is_dir():
        return
    for req_file in sorted(exp.rglob("requirements.txt")):
        if SKIP_PARTS.intersection(req_file.parts):
            continue
        for line in req_file.read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            match = _REQ_NAME_RE.match(line)
            if match and _normalize(match.group(1)) in core:
                errors.append(
                    f"{req_file.relative_to(ROOT)}: {line!r} re-declares a core dependency — "
                    "requirements are additive-only (same-process colocation cannot version-isolate)"
                )


def main() -> int:
    errors: list[str] = []
    check_core_does_not_import_experimental(errors)
    check_no_cross_package_imports(errors)
    check_requirements_additive_only(errors)
    if errors:
        print("check-experimental-boundaries: FAILED")
        for err in errors:
            print(f"  {err}")
        return 1
    print("check-experimental-boundaries: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
