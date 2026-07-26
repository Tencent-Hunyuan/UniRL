#!/usr/bin/env python3
"""Static guard for the cross-component recipe contracts.

Sibling of ``check_recipe_targets.py``: that one proves every ``_target_`` still
resolves, this one proves every recipe still satisfies the rules that span
sections (``unirl/config/contracts.py``). Two failure modes it closes, both of
which this repo's lint-only CI would otherwise let merge silently:

1. **A recipe that contradicts itself** — a ``sync:`` block on a trainside
   engine, ``enable_fsdp_offload: true`` under direct sampling, a dedicated
   engine with no sync handler. Runs the real contracts over every shipped
   recipe, so a bad one fails here rather than after a GPU allocation.
2. **A contracts table that has drifted from the code** — ``ENGINE_FAMILIES``
   declares, per engine family, whether it samples inside the train actor and
   which weight-sync receive paths it implements. Both are read back off the
   engine classes with ``ast`` and compared, so adding or changing an engine
   without updating the table fails here instead of silently mis-gating recipes.

Exits non-zero listing every violation.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

# YAML trees holding recipes (mirrors check_recipe_targets.py).
SCAN_DIRS = ["examples", "CPPO", "DRPO", "FlowDPPO"]
ENGINE_ROOT = ROOT / "unirl" / "rollout" / "engine"


def _load_contracts() -> types.ModuleType:
    """Load ``unirl.config.contracts`` without importing the ``unirl`` package.

    ``import unirl.config.contracts`` would execute ``unirl/config/__init__.py``,
    which pulls in torch; this hook runs in the lint venv, which has none. The
    contracts module and its one dependency (``require``) are stdlib-only by
    design, so registering a stub parent package and loading the two files
    directly is enough.
    """
    sys.modules.setdefault("unirl", types.ModuleType("unirl"))
    parent = types.ModuleType("unirl.config")
    parent.__path__ = [str(ROOT / "unirl" / "config")]
    sys.modules["unirl.config"] = parent
    for name in ("require", "contracts"):
        spec = importlib.util.spec_from_file_location(f"unirl.config.{name}", ROOT / "unirl" / "config" / f"{name}.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return sys.modules["unirl.config.contracts"]


contracts = _load_contracts()


# ---------------------------------------------------------------------------
# 1. Recipes satisfy the contracts
# ---------------------------------------------------------------------------


def check_recipes() -> tuple[list[str], int]:
    """Run every contract over every recipe. Returns (failures, recipes checked).

    Reads the YAML as plain data, leaving ``${...}`` interpolations unresolved.
    That is exact for these contracts: the keys they read (``_target_`` paths,
    ``layout``, ``enable_fsdp_offload``) are literals in every recipe, and a
    recipe that starts interpolating them would surface here as a violation
    rather than pass silently.
    """
    failures: list[str] = []
    checked = 0
    for scan_dir in SCAN_DIRS:
        for path in sorted((ROOT / scan_dir).rglob("*.y*ml")):
            try:
                recipe = yaml.safe_load(path.read_text(encoding="utf-8"))
            except yaml.YAMLError:
                continue  # check-yaml owns malformed YAML
            if not isinstance(recipe, dict):
                continue
            if not contracts.RecipeFacts.from_cfg(recipe).engines:
                continue  # no rollout engine: no cross-component contract to check
            checked += 1
            for contract in contracts.CONTRACTS:
                try:
                    contract(recipe)
                except ValueError as exc:
                    failures.append(f"{path.relative_to(ROOT)}: {exc}")
    return failures, checked


# ---------------------------------------------------------------------------
# 2. ENGINE_FAMILIES still matches the engine classes
# ---------------------------------------------------------------------------


def _engine_class(family_dir: Path) -> ast.ClassDef | None:
    """The concrete ``*RolloutEngine`` class defined in ``<family>/engine.py``."""
    engine_py = family_dir / "engine.py"
    if not engine_py.is_file():
        return None
    tree = ast.parse(engine_py.read_text(encoding="utf-8"), filename=str(engine_py))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name.endswith("RolloutEngine"):
            return node
    return None


def check_engine_families() -> list[str]:
    """Compare the declared table against what the engine classes actually do.

    ``direct_sampling`` is read back as "does ``__init__`` take a ``pipeline``
    parameter" — the same duck-typed test the trainers use to decide whether an
    engine samples in-process. ``weight_sync`` is read back as which receive
    methods the concrete class overrides; inherited ``BaseRolloutEngine`` stubs
    only raise ``NotImplementedError``, so a family that does not override one
    genuinely cannot receive weights that way.
    """
    failures: list[str] = []
    declared = dict(contracts.ENGINE_FAMILIES)
    for family_dir in sorted(p for p in ENGINE_ROOT.iterdir() if p.is_dir()):
        cls = _engine_class(family_dir)
        if cls is None:
            continue  # no engine class here (e.g. a helpers-only package)
        family = family_dir.name
        if family not in declared:
            failures.append(
                f"unirl/rollout/engine/{family}/engine.py defines {cls.name} but "
                f"ENGINE_FAMILIES has no '{family}' entry, so recipes using it skip every "
                "engine-dependent contract. Add it to unirl/config/contracts.py."
            )
            continue
        entry = declared.pop(family)

        init = next(
            (n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"),
            None,
        )
        params = set() if init is None else {a.arg for a in [*init.args.args, *init.args.kwonlyargs]}
        actual_direct = "pipeline" in params
        if actual_direct != entry.direct_sampling:
            failures.append(
                f"ENGINE_FAMILIES['{family}'].direct_sampling={entry.direct_sampling} but "
                f"{cls.name}.__init__ {'takes' if actual_direct else 'does not take'} a "
                f"'pipeline' parameter, so it {'is' if actual_direct else 'is not'} a "
                "direct-sampling engine."
            )

        defined = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
        actual_sync = frozenset(m for m in contracts.SYNC_RECEIVE_METHODS if m in defined)
        if actual_sync != entry.weight_sync:
            missing = sorted(actual_sync - entry.weight_sync)
            extra = sorted(entry.weight_sync - actual_sync)
            failures.append(
                f"ENGINE_FAMILIES['{family}'].weight_sync is out of date: {cls.name} "
                f"implements {sorted(actual_sync)}."
                + (f" Undeclared: {missing}." if missing else "")
                + (f" Declared but not implemented: {extra}." if extra else "")
            )

    for family in sorted(declared):
        failures.append(
            f"ENGINE_FAMILIES declares '{family}', but unirl/rollout/engine/{family}/engine.py "
            "has no engine class — stale entry after a move or removal."
        )
    return failures


# ---------------------------------------------------------------------------
# 3. The contracts actually reject the combinations they claim to
# ---------------------------------------------------------------------------

TRAINSIDE = "unirl.rollout.engine.trainside.engine.TrainsideRolloutEngine"
SGLANG_DIFFUSION = "unirl.rollout.engine.sglang_diffusion.engine.SGLangDiffusionRolloutEngine"
VLLM_OMNI = "unirl.rollout.engine.vllm_omni.engine.VLLMOmniRolloutEngine"
FASTVIDEO = "unirl.rollout.engine.fastvideo.engine.FastVideoRolloutEngine"
COMPOSED = "unirl.rollout.engine.composed.engine.ComposedRolloutEngine"

TENSOR_SYNC = "unirl.distributed.weight_sync.full.tensor.TensorWeightSync"
IPC_SYNC = "unirl.distributed.weight_sync.full.ipc.IPCWeightSync"
LORA_SYNC = "unirl.distributed.weight_sync.lora.LocalLoraWeightSync"
CHECKPOINT_SYNC = "unirl.distributed.weight_sync.full.checkpoint.CheckpointWeightSync"

#: Recipes that must be rejected, with the reason each is invalid. The first two
#: are the combinations ``unirl/config/README.md`` has always called invalid and
#: which nothing rejected before these contracts were wired up.
MUST_REJECT: dict[str, dict] = {
    "sync block on a trainside engine": {
        "rollout": {"_target_": TRAINSIDE},
        "sync": {"_target_": TENSOR_SYNC},
    },
    "direct sampling with offload": {
        "rollout": {"_target_": TRAINSIDE},
        "enable_fsdp_offload": True,
    },
    "direct sampling on a separate slab": {
        "rollout": {"_target_": TRAINSIDE},
        "layout": "separate",
    },
    "dedicated engine with no sync handler": {
        "rollout": {"_target_": SGLANG_DIFFUSION},
    },
    "dedicated engine with a handler-less sync section": {
        "rollout": {"_target_": SGLANG_DIFFUSION},
        "sync": {"track_prefix": "diffusion"},
    },
    "CUDA-IPC sync on an engine without an IPC receive path": {
        "rollout": {"_target_": SGLANG_DIFFUSION},
        "sync": {"_target_": IPC_SYNC},
    },
    "tensor-payload sync on a checkpoint-only engine": {
        "rollout": {"_target_": FASTVIDEO},
        "sync": {"_target_": TENSOR_SYNC},
    },
    "checkpoint sync on an engine without a checkpoint receive path": {
        "rollout": {"_target_": SGLANG_DIFFUSION},
        "sync": {"_target_": CHECKPOINT_SYNC},
    },
    "misspelled layout": {
        "rollout": {"_target_": SGLANG_DIFFUSION},
        "sync": {"_target_": TENSOR_SYNC},
        "layout": "seperate",
    },
    "engine sections on opposite sides of the sampling split": {
        "ar_rollout": {"_target_": TRAINSIDE},
        "dit_rollout": {"_target_": VLLM_OMNI},
        "sync": {"_target_": TENSOR_SYNC},
    },
}

#: Recipes that must pass, so a contract cannot be "fixed" by rejecting more.
MUST_ACCEPT: dict[str, dict] = {
    "colocated direct sampling": {"rollout": {"_target_": TRAINSIDE}},
    "direct sampling explicitly declining offload": {
        "rollout": {"_target_": TRAINSIDE},
        "enable_fsdp_offload": False,
    },
    "dedicated engine on a separate slab": {
        "rollout": {"_target_": SGLANG_DIFFUSION},
        "sync": {"_target_": TENSOR_SYNC},
        "layout": "separate",
    },
    "checkpoint sync on the checkpoint-only engine": {
        "rollout": {"_target_": FASTVIDEO},
        "sync": {"_target_": CHECKPOINT_SYNC},
    },
    "two dedicated engines under one handler": {
        "ar_rollout": {"_target_": VLLM_OMNI},
        "dit_rollout": {"_target_": VLLM_OMNI},
        "sync": {"_target_": TENSOR_SYNC},
    },
    "per-track sync map on the PE coordinator": {
        "rollout": {"_target_": COMPOSED},
        "sync": {"ar": {"_target_": TENSOR_SYNC}, "diffusion": {"_target_": LORA_SYNC}},
    },
    "supervised recipe with no rollout at all": {"bundle": {"_target_": "unirl.models.sd3.bundle.SD3Bundle"}},
    "out-of-tree engine, which no contract can judge": {"rollout": {"_target_": "acme.engines.MyEngine"}},
}


def check_gate_bites() -> list[str]:
    """Prove the contracts still reject what they claim to, and nothing more.

    Without this a green run is ambiguous: contracts that silently no-op — the
    exact bug this module was added to fix — would also pass ``check_recipes``.
    """
    failures = []
    for reason, recipe in MUST_REJECT.items():
        try:
            contracts.validate_recipe(recipe, entrypoint="check")
        except ValueError:
            continue
        failures.append(f"not rejected: {reason} ({recipe})")
    for reason, recipe in MUST_ACCEPT.items():
        try:
            contracts.validate_recipe(recipe, entrypoint="check")
        except ValueError as exc:
            failures.append(f"wrongly rejected: {reason} -- {exc}")
    return failures


def main() -> int:
    gate_failures = check_gate_bites()
    recipe_failures, checked = check_recipes()
    table_failures = check_engine_families()
    if gate_failures:
        print("Cross-component contracts are not gating what they claim:", file=sys.stderr)
        for f in gate_failures:
            print(f"  {f}", file=sys.stderr)
    if table_failures:
        print("Engine family table drifted from the engine classes:", file=sys.stderr)
        for f in table_failures:
            print(f"  {f}", file=sys.stderr)
    if recipe_failures:
        print("Recipes violating a cross-component contract:", file=sys.stderr)
        for f in recipe_failures:
            print(f"  {f}", file=sys.stderr)
    if gate_failures or recipe_failures or table_failures:
        return 1
    print(
        f"check-recipe-contracts: {len(MUST_REJECT)} invalid combinations rejected, "
        f"{len(MUST_ACCEPT)} valid ones accepted; {checked} recipes satisfy the contracts; "
        f"{len(contracts.ENGINE_FAMILIES)} engine families match their classes."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
