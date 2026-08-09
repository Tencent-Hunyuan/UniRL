#!/usr/bin/env python3
"""Static guard for the experimental-tier boundaries.

Three rules, all enforceable without importing anything (pure ``ast`` +
text, same spirit as ``check_recipe_targets.py``):

1. **Core never imports experimental.** ``unirl/`` must not reference the
   ``experimental`` namespace — the dependency arrow points one way.
2. **Experimental packages never import each other.** ``experimental/<a>``
   may import ``unirl`` and its own package only — in any spelling:
   absolute, ``from experimental import <b>``, bare ``import experimental``,
   or a relative import that climbs past the package root. Top-level
   ``experimental/*.py`` files are not a shared space and may not import
   the tier at all.
3. **Recipe requirements are additive-only.** Reward and actor share one
   Python process, so an ``experimental/**/requirements*.txt`` cannot
   version-"isolate" anything: any requirement whose (normalized) name the
   locked core stack already governs — ``[project.dependencies]``, every
   ``[project.optional-dependencies]`` extra, or a ``[tool.uv]`` override
   pin — would mutate that stack on install and is rejected, as are pip
   option/include lines and URL requirements the name parser cannot vet.

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
        if not SKIP_PARTS.intersection(path.relative_to(ROOT).parts):
            yield path


def _imports(path: Path):
    """Yield ``(module, from_names)`` per import statement in ``path``.

    ``import a.b`` → ``("a.b", ())``; ``from a import b, c`` → ``("a", ("b", "c"))``.
    Relative imports are resolved against the file's package (its directory
    chain under ROOT) before yielding; a level that climbs out of the tree
    yields nothing.
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError):
        return
    pkg = path.relative_to(ROOT).parts[:-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, ()
        elif isinstance(node, ast.ImportFrom):
            names = tuple(alias.name for alias in node.names)
            if node.level == 0:
                if node.module:
                    yield node.module, names
            elif node.level <= len(pkg):
                anchor = pkg[: len(pkg) - (node.level - 1)]
                parts = (*anchor, *(node.module.split(".") if node.module else ()))
                if parts:
                    yield ".".join(parts), names


def check_core_does_not_import_experimental(errors: list[str]) -> None:
    for path in _py_files(ROOT / "unirl"):
        for module, _ in _imports(path):
            if module == "experimental" or module.startswith("experimental."):
                errors.append(f"{path.relative_to(ROOT)}: core imports {module!r} — the arrow points the other way")


def check_no_cross_package_imports(errors: list[str]) -> None:
    exp = ROOT / "experimental"
    if not exp.is_dir():
        return
    for path in _py_files(exp):
        rel = path.relative_to(exp)
        own = rel.parts[0] if len(rel.parts) > 1 else None

        def flag(module: str) -> None:
            if own is None:
                errors.append(
                    f"{path.relative_to(ROOT)}: imports {module!r} from top-level experimental/ — "
                    "the tier root is not a shared space; code lives inside one package"
                )
            else:
                errors.append(
                    f"{path.relative_to(ROOT)}: imports {module!r} from a sibling package — "
                    "shared code graduates into core, it is not borrowed sideways"
                )

        for module, names in _imports(path):
            if module == "experimental":
                if not names:
                    errors.append(
                        f"{path.relative_to(ROOT)}: bare 'import experimental' reaches every sibling — "
                        "import your own package or core explicitly"
                    )
                for name in names:
                    if name != own:
                        flag(f"experimental.{name}")
            elif module.startswith("experimental."):
                if module.split(".")[1] != own:
                    flag(module)


def _core_dependency_names(pyproject: dict) -> set[str]:
    """Every name whose version the locked core stack already governs."""
    deps = list(pyproject["project"]["dependencies"])
    for extra in pyproject["project"].get("optional-dependencies", {}).values():
        deps.extend(extra)
    deps.extend(pyproject.get("tool", {}).get("uv", {}).get("override-dependencies", []))
    return {_normalize(_REQ_NAME_RE.match(dep).group(1)) for dep in deps}


def check_requirements_additive_only(errors: list[str]) -> None:
    core = _core_dependency_names(tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8")))
    exp = ROOT / "experimental"
    if not exp.is_dir():
        return
    for req_file in sorted(exp.rglob("requirements*.txt")):
        if SKIP_PARTS.intersection(req_file.relative_to(ROOT).parts):
            continue
        rel = req_file.relative_to(ROOT)
        for lineno, raw in enumerate(req_file.read_text(encoding="utf-8").splitlines(), 1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if line.startswith("-"):
                errors.append(
                    f"{rel}:{lineno}: {line!r} — pip option/include lines are not allowed; "
                    "declare additive name-based pins only"
                )
                continue
            match = _REQ_NAME_RE.match(line)
            rest = line[match.end() :] if match else ""
            if not match or (rest and rest[0] not in " \t@[<>=!~;,"):
                errors.append(f"{rel}:{lineno}: {line!r} — unparseable requirement; use PEP 508 name-based pins")
            elif _normalize(match.group(1)) in core:
                errors.append(
                    f"{rel}:{lineno}: {line!r} re-declares a core dependency — "
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
