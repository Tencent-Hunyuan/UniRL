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


TRAINSIDE = "unirl.rollout.engine.trainside.engine.TrainsideRolloutEngine"
SGLANG = "unirl.rollout.engine.sglang.engine.SGLangRolloutEngine"
SGLANG_DIFFUSION = "unirl.rollout.engine.sglang_diffusion.engine.SGLangDiffusionRolloutEngine"
VLLM_OMNI = "unirl.rollout.engine.vllm_omni.engine.VLLMOmniRolloutEngine"
FASTVIDEO = "unirl.rollout.engine.fastvideo.engine.FastVideoRolloutEngine"
COMPOSED = "unirl.rollout.engine.composed.engine.ComposedRolloutEngine"
AGENTIC = "unirl.rollout.engine.agentic.engine.AgenticRolloutEngine"
SGLANG_CONFIG = "unirl.rollout.engine.sglang.config.SGLangEngineConfig"
SGLANG_DIFFUSION_CONFIG = "unirl.rollout.engine.sglang_diffusion.config.SGLangDiffusionEngineConfig"
VLLM_OMNI_CONFIG = "unirl.rollout.engine.vllm_omni.config.VLLMOmniEngineConfig"
FASTVIDEO_CONFIG = "unirl.rollout.engine.fastvideo.config.FastVideoEngineConfig"

TENSOR_SYNC = "unirl.distributed.weight_sync.full.tensor.TensorWeightSync"
IPC_SYNC = "unirl.distributed.weight_sync.full.ipc.IPCWeightSync"
NCCL_SYNC = "unirl.distributed.weight_sync.full.nccl.NCCLWeightSync"
LOCAL_LORA_SYNC = "unirl.distributed.weight_sync.lora.LocalLoraWeightSync"
REMOTE_LORA_SYNC = "unirl.distributed.weight_sync.lora.RemoteLoraWeightSync"
CHECKPOINT_SYNC = "unirl.distributed.weight_sync.full.checkpoint.CheckpointWeightSync"
CKPT_ENGINE_IPC_SYNC = "unirl.distributed.weight_sync.full.ckpt_engine_ipc.CkptEngineIPCWeightSync"
AR_SAMPLING = "unirl.types.sampling.ARSamplingParams"
DIFFUSION_SAMPLING = "unirl.types.sampling.DiffusionSamplingParams"

CASE_SAMPLING_TARGETS = {
    "train_ar": AR_SAMPLING,
    "train_async_ar": AR_SAMPLING,
    "train_diffusion": DIFFUSION_SAMPLING,
    "train_async_diffusion": DIFFUSION_SAMPLING,
    "train_agentic": AR_SAMPLING,
}


def _composed_rollout(
    *,
    ar: str = SGLANG_CONFIG,
    diffusion: str = SGLANG_DIFFUSION_CONFIG,
) -> dict:
    return {
        "_target_": COMPOSED,
        "config": {
            "ar": {"_target_": ar},
            "diffusion": {"_target_": diffusion},
        },
    }


MUST_REJECT: dict[str, tuple[str, dict]] = {
    "sync on direct sampling": (
        "train_diffusion",
        {"rollout": {"_target_": TRAINSIDE}, "sync": {"_target_": TENSOR_SYNC}},
    ),
    "direct sampling on separate layout": (
        "train_diffusion",
        {"rollout": {"_target_": TRAINSIDE}, "layout": "separate"},
    ),
    "direct sampling under async AR": ("train_async_ar", {"rollout": {"_target_": TRAINSIDE}}),
    "AR SGLang under diffusion": (
        "train_diffusion",
        {"rollout": {"_target_": SGLANG}, "sync": {"_target_": TENSOR_SYNC}},
    ),
    "diffusion SGLang under AR": (
        "train_ar",
        {"rollout": {"_target_": SGLANG_DIFFUSION}, "sync": {"_target_": TENSOR_SYNC}},
    ),
    "FastVideo under AR": (
        "train_ar",
        {"rollout": {"_target_": FASTVIDEO}, "sync": {"_target_": CHECKPOINT_SYNC}},
    ),
    "composed PE engine under diffusion": (
        "train_diffusion",
        {"rollout": _composed_rollout(), "sync": {"_target_": TENSOR_SYNC}},
    ),
    "agentic engine under AR": (
        "train_ar",
        {
            "rollout": {"_target_": AGENTIC, "config": {"inner": {"_target_": SGLANG_CONFIG}}},
            "sync": {"_target_": TENSOR_SYNC},
        },
    ),
    "AR sampling under diffusion with multi-domain engine": (
        "train_diffusion",
        {
            "rollout": {"_target_": VLLM_OMNI},
            "sync": {"_target_": TENSOR_SYNC},
            "sampling": {"_target_": AR_SAMPLING},
        },
    ),
    "diffusion sampling under AR with multi-domain engine": (
        "train_ar",
        {
            "rollout": {"_target_": VLLM_OMNI},
            "sync": {"_target_": TENSOR_SYNC},
            "sampling": {"_target_": DIFFUSION_SAMPLING},
        },
    ),
    "dedicated engine without sync": ("train_diffusion", {"rollout": {"_target_": SGLANG_DIFFUSION}}),
    "IPC on SGLang": (
        "train_diffusion",
        {"rollout": {"_target_": SGLANG_DIFFUSION}, "sync": {"_target_": IPC_SYNC}},
    ),
    "tensor sync on checkpoint-only engine": (
        "train_diffusion",
        {"rollout": {"_target_": FASTVIDEO}, "sync": {"_target_": TENSOR_SYNC}},
    ),
    "checkpoint sync on SGLang": (
        "train_diffusion",
        {"rollout": {"_target_": SGLANG_DIFFUSION}, "sync": {"_target_": CHECKPOINT_SYNC}},
    ),
    "checkpoint-engine IPC on vLLM-Omni": (
        "train_ar",
        {"rollout": {"_target_": VLLM_OMNI}, "sync": {"_target_": CKPT_ENGINE_IPC_SYNC}},
    ),
    "checkpoint-engine IPC on native SGLang": (
        "train_ar",
        {
            "rollout": {
                "_target_": SGLANG,
                "config": {"_target_": SGLANG_CONFIG, "backend": "native"},
            },
            "sync": {"_target_": CKPT_ENGINE_IPC_SYNC},
        },
    ),
    "checkpoint-engine IPC with speculative decoding": (
        "train_ar",
        {
            "rollout": {
                "_target_": SGLANG,
                "config": {
                    "_target_": SGLANG_CONFIG,
                    "engine_kwargs": {"speculative_algorithm": "EAGLE"},
                },
            },
            "sync": {"_target_": CKPT_ENGINE_IPC_SYNC},
        },
    ),
    "checkpoint-engine IPC with server DP": (
        "train_ar",
        {
            "rollout": {
                "_target_": SGLANG,
                "config": {"_target_": SGLANG_CONFIG, "engine_kwargs": {"dp_size": 2}},
            },
            "sync": {"_target_": CKPT_ENGINE_IPC_SYNC},
        },
    ),
    "checkpoint-engine IPC with string server DP": (
        "train_ar",
        {
            "rollout": {
                "_target_": SGLANG,
                "config": {"_target_": SGLANG_CONFIG, "engine_kwargs": {"dp_size": "1"}},
            },
            "sync": {"_target_": CKPT_ENGINE_IPC_SYNC},
        },
    ),
    "checkpoint-engine IPC with boolean server DP": (
        "train_ar",
        {
            "rollout": {
                "_target_": SGLANG,
                "config": {"_target_": SGLANG_CONFIG, "engine_kwargs": {"dp_size": True}},
            },
            "sync": {"_target_": CKPT_ENGINE_IPC_SYNC},
        },
    ),
    "checkpoint-engine IPC with floating server DP": (
        "train_ar",
        {
            "rollout": {
                "_target_": SGLANG,
                "config": {"_target_": SGLANG_CONFIG, "engine_kwargs": {"dp_size": 1.0}},
            },
            "sync": {"_target_": CKPT_ENGINE_IPC_SYNC},
        },
    ),
    "checkpoint-engine IPC with list server DP": (
        "train_ar",
        {
            "rollout": {
                "_target_": SGLANG,
                "config": {"_target_": SGLANG_CONFIG, "engine_kwargs": {"dp_size": [1]}},
            },
            "sync": {"_target_": CKPT_ENGINE_IPC_SYNC},
        },
    ),
    "falsey layout": (
        "train_diffusion",
        {"rollout": {"_target_": SGLANG_DIFFUSION}, "sync": {"_target_": TENSOR_SYNC}, "layout": False},
    ),
    "null layout": (
        "train_diffusion",
        {"rollout": {"_target_": SGLANG_DIFFUSION}, "sync": {"_target_": TENSOR_SYNC}, "layout": None},
    ),
    "separate layout with local handler": (
        "train_diffusion",
        {"rollout": {"_target_": SGLANG_DIFFUSION}, "sync": {"_target_": TENSOR_SYNC}, "layout": "separate"},
    ),
    "colocate layout with remote handler": (
        "train_diffusion",
        {"rollout": {"_target_": SGLANG_DIFFUSION}, "sync": {"_target_": NCCL_SYNC}},
    ),
    "copy LoRA on SGLang": (
        "train_diffusion",
        {
            "rollout": {"_target_": SGLANG_DIFFUSION},
            "sync": {"_target_": REMOTE_LORA_SYNC, "copy": True},
            "layout": "separate",
        },
    ),
    "copy option on tensor handler": (
        "train_diffusion",
        {
            "rollout": {"_target_": VLLM_OMNI},
            "sync": {"_target_": TENSOR_SYNC, "copy": False},
        },
    ),
    "LoRA verification on SGLang": (
        "train_diffusion",
        {
            "rollout": {"_target_": SGLANG_DIFFUSION},
            "sync": {"_target_": LOCAL_LORA_SYNC, "verify": True},
        },
    ),
    "composed checkpoint-engine IPC": (
        "train_pe",
        {
            "rollout": _composed_rollout(),
            "sync": {
                "ar": {"_target_": CKPT_ENGINE_IPC_SYNC, "track_prefix": "ar"},
                "diffusion": {"_target_": LOCAL_LORA_SYNC, "track_prefix": "diffusion"},
            },
        },
    ),
    "composed LoRA verification": (
        "train_pe",
        {
            "rollout": _composed_rollout(ar=VLLM_OMNI_CONFIG, diffusion=VLLM_OMNI_CONFIG),
            "sync": {
                "ar": {
                    "_target_": LOCAL_LORA_SYNC,
                    "track_prefix": "ar",
                    "verify": True,
                },
                "diffusion": {
                    "_target_": LOCAL_LORA_SYNC,
                    "track_prefix": "diffusion",
                    "verify": True,
                },
            },
        },
    ),
    "composed checkpoint path sync": (
        "train_pe",
        {
            "rollout": _composed_rollout(diffusion=FASTVIDEO_CONFIG),
            "sync": {
                "ar": {"_target_": TENSOR_SYNC, "track_prefix": "ar"},
                "diffusion": {
                    "_target_": CHECKPOINT_SYNC,
                    "track_prefix": "diffusion",
                },
            },
        },
    ),
    "mixed unified sampling modes": (
        "train_unified_model",
        {
            "ar_rollout": {"_target_": TRAINSIDE},
            "dit_rollout": {"_target_": VLLM_OMNI},
            "sync": {"_target_": REMOTE_LORA_SYNC},
        },
    ),
    "unified single dedicated engine": (
        "train_unified_model",
        {"rollout": {"_target_": VLLM_OMNI}, "sync": {"_target_": TENSOR_SYNC}},
    ),
    "unified incomplete split engines": (
        "train_unified_model",
        {"ar_rollout": {"_target_": VLLM_OMNI}, "sync": {"_target_": REMOTE_LORA_SYNC}},
    ),
    "unified zero-copy LoRA": (
        "train_unified_model",
        {
            "ar_rollout": {"_target_": VLLM_OMNI},
            "dit_rollout": {"_target_": VLLM_OMNI},
            "sync": {"_target_": REMOTE_LORA_SYNC},
        },
    ),
    "PE missing AR sync": (
        "train_pe",
        {
            "rollout": _composed_rollout(),
            "sync": {"diffusion": {"_target_": LOCAL_LORA_SYNC, "track_prefix": "diffusion"}},
        },
    ),
    "PE mismatched track prefix": (
        "train_pe",
        {
            "rollout": _composed_rollout(),
            "sync": {
                "ar": {"_target_": TENSOR_SYNC, "track_prefix": "diffusion"},
                "diffusion": {"_target_": LOCAL_LORA_SYNC, "track_prefix": "diffusion"},
            },
        },
    ),
    "PE IPC on SGLang child": (
        "train_pe",
        {
            "rollout": _composed_rollout(),
            "sync": {
                "ar": {"_target_": IPC_SYNC, "track_prefix": "ar"},
                "diffusion": {"_target_": LOCAL_LORA_SYNC, "track_prefix": "diffusion"},
            },
        },
    ),
    "async AR with local handler": (
        "train_async_ar",
        {"rollout": {"_target_": SGLANG}, "sync": {"_target_": TENSOR_SYNC}},
    ),
    "anchored AR with local handler": (
        "train_ar",
        {
            "rollout": {"_target_": VLLM_OMNI},
            "sync": {"_target_": TENSOR_SYNC},
            "rollout_anchor_device": 1,
        },
    ),
    "anchored AR on rank zero": (
        "train_ar",
        {
            "rollout": {"_target_": VLLM_OMNI},
            "sync": {"_target_": REMOTE_LORA_SYNC, "copy": True},
            "rollout_anchor_device": 0,
        },
    ),
    "anchored direct AR engine": (
        "train_ar",
        {"rollout": {"_target_": TRAINSIDE}, "rollout_anchor_device": 1},
    ),
    "negative rollout anchor": (
        "train_ar",
        {
            "rollout": {"_target_": VLLM_OMNI},
            "sync": {"_target_": REMOTE_LORA_SYNC, "copy": True},
            "rollout_anchor_device": -1,
        },
    ),
    "floating rollout anchor": (
        "train_ar",
        {
            "rollout": {"_target_": VLLM_OMNI},
            "sync": {"_target_": REMOTE_LORA_SYNC, "copy": True},
            "rollout_anchor_device": 1.9,
        },
    ),
    "boolean rollout anchor": (
        "train_ar",
        {
            "rollout": {"_target_": VLLM_OMNI},
            "sync": {"_target_": REMOTE_LORA_SYNC, "copy": True},
            "rollout_anchor_device": True,
        },
    ),
    "AR claims ignored layout": (
        "train_ar",
        {"rollout": {"_target_": SGLANG}, "sync": {"_target_": TENSOR_SYNC}, "layout": "separate"},
    ),
    "async diffusion layout field": (
        "train_async_diffusion",
        {"rollout": {"_target_": SGLANG_DIFFUSION}, "sync": {"_target_": NCCL_SYNC}, "layout": "separate"},
    ),
    "agentic with non-Tensor handler": (
        "train_agentic",
        {
            "rollout": {"_target_": AGENTIC, "config": {"inner": {"_target_": SGLANG_CONFIG}}},
            "sync": {"_target_": LOCAL_LORA_SYNC},
        },
    ),
    "agentic namespaced sampling": (
        "train_agentic",
        {
            "sampling": {"ar": {"_target_": AR_SAMPLING}},
            "rollout": {"_target_": AGENTIC, "config": {"inner": {"_target_": SGLANG_CONFIG}}},
            "sync": {"_target_": TENSOR_SYNC},
        },
    ),
    "swapped PE sampling tracks": (
        "train_pe",
        {
            "sampling": {
                "ar": {"_target_": DIFFUSION_SAMPLING},
                "diffusion": {"_target_": AR_SAMPLING},
            },
            "rollout": _composed_rollout(),
            "sync": {
                "ar": {"_target_": TENSOR_SYNC, "track_prefix": "ar"},
                "diffusion": {"_target_": LOCAL_LORA_SYNC, "track_prefix": "diffusion"},
            },
        },
    ),
    "swapped unified sampling tracks": (
        "train_unified_model",
        {
            "sampling": {
                "ar": {"_target_": DIFFUSION_SAMPLING},
                "diffusion": {"_target_": AR_SAMPLING},
            },
            "ar_rollout": {"_target_": VLLM_OMNI},
            "dit_rollout": {"_target_": VLLM_OMNI},
            "sync": {"_target_": REMOTE_LORA_SYNC, "copy": True},
        },
    ),
    "unified diffusion-only sampling": (
        "train_unified_model",
        {
            "sampling": {"diffusion": {"_target_": DIFFUSION_SAMPLING}},
            "rollout": {"_target_": TRAINSIDE},
        },
    ),
    "SFT with dead sync": (
        "train_sft",
        {
            "bundle": {"_target_": "unirl.models.sd3.bundle.SD3Bundle"},
            "sync": {"_target_": TENSOR_SYNC},
        },
    ),
    "diffusion with dead rollout anchor": (
        "train_diffusion",
        {
            "rollout": {"_target_": TRAINSIDE},
            "rollout_anchor_device": 1,
        },
    ),
    "AR with dynamic dead layout": (
        "train_ar",
        {
            "rollout": {"_target_": SGLANG},
            "sync": {"_target_": TENSOR_SYNC},
            "layout": "${oc.env:LAYOUT,colocate}",
        },
    ),
}

MUST_ACCEPT: dict[str, tuple[str, dict]] = {
    "colocated direct sampling": ("train_diffusion", {"rollout": {"_target_": TRAINSIDE}}),
    "separate diffusion with NCCL": (
        "train_diffusion",
        {"rollout": {"_target_": SGLANG_DIFFUSION}, "sync": {"_target_": NCCL_SYNC}, "layout": "separate"},
    ),
    "checkpoint sync on FastVideo": (
        "train_diffusion",
        {"rollout": {"_target_": FASTVIDEO}, "sync": {"_target_": CHECKPOINT_SYNC}},
    ),
    "checkpoint-engine IPC on SGLang": (
        "train_ar",
        {"rollout": {"_target_": SGLANG}, "sync": {"_target_": CKPT_ENGINE_IPC_SYNC}},
    ),
    "two vLLM unified engines": (
        "train_unified_model",
        {
            "ar_rollout": {"_target_": VLLM_OMNI},
            "dit_rollout": {"_target_": VLLM_OMNI},
            "sync": {"_target_": REMOTE_LORA_SYNC, "copy": True},
        },
    ),
    "complete PE map": (
        "train_pe",
        {
            "rollout": _composed_rollout(),
            "sync": {
                "ar": {"_target_": TENSOR_SYNC, "track_prefix": "ar"},
                "diffusion": {"_target_": LOCAL_LORA_SYNC, "track_prefix": "diffusion"},
            },
        },
    ),
    "frozen PE map": (
        "train_pe",
        {
            "rollout": _composed_rollout(),
            "freeze_llm": True,
            "sync": {"diffusion": {"_target_": LOCAL_LORA_SYNC, "track_prefix": "diffusion"}},
        },
    ),
    "anchored AR remote LoRA": (
        "train_ar",
        {
            "rollout": {"_target_": VLLM_OMNI},
            "sync": {"_target_": REMOTE_LORA_SYNC, "copy": True},
            "rollout_anchor_device": 1,
        },
    ),
    "dynamic anchored AR remote LoRA": (
        "train_ar",
        {
            "rollout": {"_target_": VLLM_OMNI},
            "sync": {"_target_": REMOTE_LORA_SYNC},
            "rollout_anchor_device": "${oc.env:ANCHOR,1}",
        },
    ),
    "async AR NCCL": (
        "train_async_ar",
        {"rollout": {"_target_": SGLANG}, "sync": {"_target_": NCCL_SYNC}},
    ),
    "async diffusion NCCL": (
        "train_async_diffusion",
        {"rollout": {"_target_": SGLANG_DIFFUSION}, "sync": {"_target_": NCCL_SYNC}},
    ),
    "namespaced AR sampling": (
        "train_ar",
        {
            "sampling": {"ar": {"_target_": AR_SAMPLING}},
            "rollout": {"_target_": SGLANG},
            "sync": {"_target_": TENSOR_SYNC},
        },
    ),
    "namespaced diffusion sampling": (
        "train_diffusion",
        {
            "sampling": {"diffusion": {"_target_": DIFFUSION_SAMPLING}},
            "rollout": {"_target_": SGLANG_DIFFUSION},
            "sync": {"_target_": TENSOR_SYNC},
        },
    ),
    "dynamic diffusion layout": (
        "train_diffusion",
        {
            "layout": "${oc.env:LAYOUT,colocate}",
            "rollout": {"_target_": SGLANG_DIFFUSION},
            "sync": {"_target_": TENSOR_SYNC},
        },
    ),
    "agentic Tensor sync": (
        "train_agentic",
        {
            "rollout": {"_target_": AGENTIC, "config": {"inner": {"_target_": SGLANG_CONFIG}}},
            "sync": {"_target_": TENSOR_SYNC},
        },
    ),
    "supervised recipe": ("train_sft", {"bundle": {"_target_": "unirl.models.sd3.bundle.SD3Bundle"}}),
    "out-of-tree direct engine": ("train_diffusion", {"rollout": {"_target_": "acme.engines.MyEngine"}}),
    "out-of-tree dedicated engine": (
        "train_diffusion",
        {
            "rollout": {"_target_": "acme.engines.MyDedicatedEngine"},
            "sync": {"_target_": "acme.sync.CustomWeightSync"},
        },
    ),
    "out-of-tree top-level sampling": (
        "train_diffusion",
        {
            "sampling": {"_target_": "acme.sampling.CustomSampler"},
            "rollout": {"_target_": TRAINSIDE},
        },
    ),
    "out-of-tree namespaced sampling": (
        "train_ar",
        {
            "sampling": {"ar": {"_target_": "acme.sampling.CustomSampler"}},
            "rollout": {"_target_": SGLANG},
            "sync": {"_target_": TENSOR_SYNC},
        },
    ),
}

EXPECTED_REJECT_MESSAGES = {
    "unified single dedicated engine": "single-engine mode does not wire weight sync",
}


def check_gate_bites() -> list[str]:
    """Prove every documented invalid and valid combination reaches the expected side."""
    failures = []
    for reason, (entrypoint, recipe) in MUST_REJECT.items():
        recipe = _with_case_sampling(entrypoint, recipe)
        try:
            contracts.validate_recipe(recipe, entrypoint=entrypoint)
        except ValueError as exc:
            expected = EXPECTED_REJECT_MESSAGES.get(reason)
            if expected is not None and expected not in str(exc):
                failures.append(f"wrong rejection: {reason} -- {exc}")
            continue
        failures.append(f"not rejected: {reason} ({recipe})")
    for reason, (entrypoint, recipe) in MUST_ACCEPT.items():
        recipe = _with_case_sampling(entrypoint, recipe)
        try:
            contracts.validate_recipe(recipe, entrypoint=entrypoint)
        except ValueError as exc:
            failures.append(f"wrongly rejected: {reason} -- {exc}")
    return failures


def _with_case_sampling(entrypoint: str, recipe: dict) -> dict:
    completed = dict(recipe)
    target = CASE_SAMPLING_TARGETS.get(entrypoint)
    if target is not None:
        completed.setdefault("sampling", {"_target_": target})
    elif entrypoint in ("train_pe", "train_unified_model"):
        completed.setdefault(
            "sampling",
            {
                "ar": {"_target_": AR_SAMPLING},
                "diffusion": {"_target_": DIFFUSION_SAMPLING},
            },
        )
    return completed


def main() -> int:
    gate_failures = check_gate_bites()
    recipe_failures, checked = check_recipes()
    table_failures = check_engine_families()
    handler_failures = check_sync_handlers()
    entrypoint_failures = check_entrypoint_gates()
    sections = (
        ("Contract mutation cases failed", gate_failures),
        ("Recipes violate contracts", recipe_failures),
        ("Engine metadata drifted", table_failures),
        ("Sync-handler metadata drifted", handler_failures),
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
        f"check-recipe-contracts: {len(MUST_REJECT)} invalid rejected, {len(MUST_ACCEPT)} valid accepted; "
        f"{checked} recipes; {len(contracts.ENGINE_FAMILIES)} engine families; all entrypoints gated."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
