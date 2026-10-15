#!/usr/bin/env python3
"""Statically guard recipe contracts, engine metadata, and entrypoint wiring."""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ENGINE_ROOT = ROOT / "unirl" / "rollout" / "engine"
ENTRYPOINT_OVERRIDES = {
    "examples/unified_model/hi3_it2i.yaml": "train_diffusion",
    "examples/unified_model/hi3_trainside_t2i.yaml": "train_diffusion",
}


def _load_contracts() -> types.ModuleType:
    """Load contracts without importing the torch-dependent package surface."""
    sys.modules.setdefault("unirl", types.ModuleType("unirl"))
    for package in ("config", "types"):
        parent = types.ModuleType(f"unirl.{package}")
        parent.__path__ = [str(ROOT / "unirl" / package)]
        sys.modules[f"unirl.{package}"] = parent
    for package, name in (("types", "loss_agg"), ("config", "require"), ("config", "contracts")):
        spec = importlib.util.spec_from_file_location(
            f"unirl.{package}.{name}", ROOT / "unirl" / package / f"{name}.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return sys.modules["unirl.config.contracts"]


contracts = _load_contracts()


def _merge(base: dict, override: dict) -> dict:
    """Recursively merge one plain YAML mapping with Hydra-style mapping precedence."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_recipe(path: Path, stack: tuple[Path, ...] = ()) -> dict:
    """Load the local string-only defaults form used by shipped composed recipes."""
    if path in stack:
        cycle = " -> ".join(str(item.relative_to(ROOT)) for item in (*stack, path))
        raise ValueError(f"{path.relative_to(ROOT)}: cyclic defaults: {cycle}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"{path.relative_to(ROOT)}: cannot load recipe: {exc}") from exc
    try:
        recipe = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"{path.relative_to(ROOT)}: invalid YAML: {exc}") from exc
    if not isinstance(recipe, dict):
        raise ValueError(f"{path.relative_to(ROOT)}: recipe must be a mapping")
    defaults = recipe.pop("defaults", None)
    if defaults is None:
        return recipe
    if not isinstance(defaults, list):
        raise ValueError(f"{path.relative_to(ROOT)}: defaults must be a list")
    merged: dict = {}
    placed_self = False
    for item in defaults:
        if item == "_self_":
            merged = _merge(merged, recipe)
            placed_self = True
            continue
        if not isinstance(item, str):
            raise ValueError(f"{path.relative_to(ROOT)}: static contract composition supports string defaults only")
        parent = path.parent / f"{item}.yaml"
        parent_recipe = _load_recipe(parent, (*stack, path))
        merged = _merge(merged, parent_recipe)
    return merged if placed_self else _merge(merged, recipe)


def _entrypoint_for(path: Path) -> str | None:
    """Infer the repository entrypoint that owns one shipped recipe."""
    rel = path.relative_to(ROOT)
    if override := ENTRYPOINT_OVERRIDES.get(rel.as_posix()):
        return override
    if len(rel.parts) < 3:
        return None
    domain = rel.parts[1]
    if domain == "ar":
        return "train_async_ar" if path.stem.endswith("_async") else "train_ar"
    if domain == "diffusion":
        return "train_async_diffusion" if path.stem.endswith("_async") else "train_diffusion"
    return {
        "deep_research": "train_agentic",
        "pe": "train_pe",
        "sft": "train_sft",
        "unified_model": "train_unified_model",
    }.get(domain)


def check_recipes() -> tuple[list[str], int]:
    """Run the runtime gate over every shipped training recipe."""
    failures: list[str] = []
    checked = 0
    unseen_overrides = set(ENTRYPOINT_OVERRIDES)
    for path in sorted((ROOT / "examples").rglob("*.y*ml")):
        relative = path.relative_to(ROOT).as_posix()
        unseen_overrides.discard(relative)
        try:
            recipe = _load_recipe(path)
        except ValueError as exc:
            message = str(exc)
            failures.append(message if message.startswith(f"{relative}:") else f"{relative}: {message}")
            continue
        entrypoint = _entrypoint_for(path)
        if entrypoint is None:
            failures.append(f"{relative}: cannot infer the owning train_*.py entrypoint")
            continue
        checked += 1
        try:
            contracts.validate_recipe(recipe, entrypoint=entrypoint)
        except ValueError as exc:
            failures.append(f"{relative}: {exc}")
    failures.extend(f"ENTRYPOINT_OVERRIDES declares missing path {path}" for path in sorted(unseen_overrides))
    return failures, checked


def _engine_class(family_dir: Path) -> ast.ClassDef | None:
    """Return the concrete RolloutEngine class declared by one family."""
    engine_py = family_dir / "engine.py"
    if not engine_py.is_file():
        return None
    tree = ast.parse(engine_py.read_text(encoding="utf-8"), filename=str(engine_py))
    return next(
        (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name.endswith("RolloutEngine")),
        None,
    )


def check_engine_families() -> list[str]:
    """Compare family declarations with concrete engine constructors and methods."""
    failures: list[str] = []
    declared = dict(contracts.ENGINE_FAMILIES)
    for family_dir in sorted(path for path in ENGINE_ROOT.iterdir() if path.is_dir()):
        cls = _engine_class(family_dir)
        if cls is None:
            continue
        family = family_dir.name
        if family not in declared:
            failures.append(
                f"{family_dir.relative_to(ROOT)}/engine.py defines {cls.name}, but ENGINE_FAMILIES omits it"
            )
            continue
        entry = declared.pop(family)
        init = next(
            (node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__"),
            None,
        )
        params = set() if init is None else {arg.arg for arg in [*init.args.args, *init.args.kwonlyargs]}
        actual_direct = "pipeline" in params
        if actual_direct != entry.direct_sampling:
            failures.append(
                f"ENGINE_FAMILIES[{family!r}].direct_sampling={entry.direct_sampling}, "
                f"but {cls.name}.__init__ pipeline parameter says {actual_direct}"
            )
        defined = {node.name for node in cls.body if isinstance(node, ast.FunctionDef)}
        actual_capabilities = frozenset(method for method in contracts.ENGINE_CAPABILITY_METHODS if method in defined)
        if actual_capabilities != entry.capabilities:
            failures.append(
                f"ENGINE_FAMILIES[{family!r}].capabilities={sorted(entry.capabilities)}, "
                f"but {cls.name} implements {sorted(actual_capabilities)}"
            )
    for family in sorted(declared):
        failures.append(f"ENGINE_FAMILIES declares stale family {family!r}")
    return failures


SYNC_HANDLER_FILES = {
    "IPCWeightSync": "unirl/distributed/weight_sync/full/ipc.py",
    "TensorWeightSync": "unirl/distributed/weight_sync/full/tensor.py",
    "NCCLWeightSync": "unirl/distributed/weight_sync/full/nccl.py",
    "LocalLoraWeightSync": "unirl/distributed/weight_sync/lora/local.py",
    "RemoteLoraWeightSync": "unirl/distributed/weight_sync/lora/remote.py",
    "CheckpointWeightSync": "unirl/distributed/weight_sync/full/checkpoint.py",
    "CkptEngineIPCWeightSync": "unirl/distributed/weight_sync/full/ckpt_engine_ipc.py",
}


def check_sync_handlers() -> list[str]:
    """Verify handler topology metadata against constructor ownership boundaries."""
    failures: list[str] = []
    if set(SYNC_HANDLER_FILES) != set(contracts.SYNC_HANDLERS):
        return ["SYNC_HANDLER_FILES and SYNC_HANDLERS must declare identical handler classes"]
    for name, relative in SYNC_HANDLER_FILES.items():
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"{relative} is missing")
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        cls = next((node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name), None)
        if cls is None:
            failures.append(f"{relative} does not define {name}")
            continue
        init = next(
            (node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__"),
            None,
        )
        params = set() if init is None else {arg.arg for arg in [*init.args.args, *init.args.kwonlyargs]}
        actual = contracts.SYNC_LOCAL if "rollout" in params else contracts.SYNC_REMOTE
        declared = contracts.SYNC_HANDLERS[name].topology
        if actual != declared:
            failures.append(f"SYNC_HANDLERS[{name!r}].topology={declared!r}, constructor implies {actual!r}")
    return failures


def check_algorithm_loss_agg_modes() -> list[str]:
    """Require ALGORITHM_LOSS_AGG_MODES to list exactly the algorithms whose constructor takes loss_agg_mode."""
    failures: list[str] = []
    declared = dict(contracts.ALGORITHM_LOSS_AGG_MODES)
    for path in sorted((ROOT / "unirl" / "algorithms").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for cls in (node for node in tree.body if isinstance(node, ast.ClassDef)):
            init = next(
                (node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__"),
                None,
            )
            if init is None:
                continue
            args = init.args
            named = [*args.args, *args.kwonlyargs]
            if "loss_agg_mode" not in {arg.arg for arg in named}:
                continue
            target = f"unirl.algorithms.{path.stem}.{cls.name}"
            modes = declared.pop(target, None)
            if modes is None:
                failures.append(f"{target}.__init__ takes loss_agg_mode, but ALGORITHM_LOSS_AGG_MODES omits it")
                continue
            positional_defaults = dict(zip([arg.arg for arg in args.args][-len(args.defaults) :], args.defaults))
            keyword_defaults = {arg.arg: default for arg, default in zip(args.kwonlyargs, args.kw_defaults)}
            default = {**positional_defaults, **keyword_defaults}.get("loss_agg_mode")
            if isinstance(default, ast.Constant) and default.value not in modes:
                failures.append(
                    f"{target}.__init__ defaults loss_agg_mode={default.value!r}, "
                    f"outside ALGORITHM_LOSS_AGG_MODES {sorted(modes)}"
                )
    failures.extend(f"ALGORITHM_LOSS_AGG_MODES declares stale algorithm {target!r}" for target in sorted(declared))
    return failures


def check_entrypoint_gates() -> list[str]:
    """Require validate_recipe to be the first statement in every train main."""
    failures: list[str] = []
    seen: set[str] = set()
    for path in sorted((ROOT / "unirl").glob("train_*.py")):
        entrypoint = path.stem
        seen.add(entrypoint)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        main = next(
            (node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"),
            None,
        )
        first = None if main is None or not main.body else main.body[0]
        call = first.value if isinstance(first, ast.Expr) and isinstance(first.value, ast.Call) else None
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name) or call.func.id != "validate_recipe":
            failures.append(f"{path.relative_to(ROOT)}: main() must call validate_recipe first")
            continue
        keyword = next((kw.value for kw in call.keywords if kw.arg == "entrypoint"), None)
        if not isinstance(keyword, ast.Constant) or keyword.value != entrypoint:
            failures.append(f"{path.relative_to(ROOT)}: validate_recipe entrypoint must be {entrypoint!r}")
    missing = set(contracts.KNOWN_ENTRYPOINTS) - seen
    extra = seen - set(contracts.KNOWN_ENTRYPOINTS)
    failures.extend(f"KNOWN_ENTRYPOINTS declares missing unirl/{name}.py" for name in sorted(missing))
    failures.extend(f"unirl/{name}.py is absent from KNOWN_ENTRYPOINTS" for name in sorted(extra))
    return failures


def main() -> int:
    recipe_failures, checked = check_recipes()
    table_failures = check_engine_families()
    handler_failures = check_sync_handlers()
    algorithm_failures = check_algorithm_loss_agg_modes()
    entrypoint_failures = check_entrypoint_gates()
    sections = (
        ("Recipes violate contracts", recipe_failures),
        ("Engine metadata drifted", table_failures),
        ("Sync-handler metadata drifted", handler_failures),
        ("Algorithm loss_agg_mode metadata drifted", algorithm_failures),
        ("Training entrypoints bypass the gate", entrypoint_failures),
    )
    for title, failures in sections:
        if failures:
            print(f"{title}:", file=sys.stderr)
            for failure in failures:
                print(f"  {failure}", file=sys.stderr)
    if any(failures for _, failures in sections):
        return 1
    print(
        f"check-recipe-contracts: {checked} recipes; "
        f"{len(contracts.ENGINE_FAMILIES)} engine families; all entrypoints gated."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
