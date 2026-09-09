"""Selective-import shim for the VeOmni distributed layer."""

from __future__ import annotations

import functools
import importlib
import importlib.machinery
import importlib.util
import logging
import os
import sys
import types
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def _stub_package(name: str, path: str) -> types.ModuleType:
    """Insert a package-shaped stub into ``sys.modules`` (idempotent)."""
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    mod = types.ModuleType(name)
    mod.__path__ = [path]
    spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    spec.submodule_search_locations = [path]
    mod.__spec__ = spec
    sys.modules[name] = mod
    return mod


def _install_path_stubs() -> None:
    """Stub ``veomni`` and ``veomni.models`` so their inits never execute."""
    if "veomni" not in sys.modules:
        spec = importlib.util.find_spec("veomni")  # locates; does not execute
        if spec is None or spec.origin is None:
            raise ModuleNotFoundError(
                'veomni is not installed — install unirl with the [veomni] extra (uv pip install -e ".[...,veomni]")'
            )
        pkg_dir = os.path.dirname(spec.origin)
        _stub_package("veomni", pkg_dir)
    else:
        pkg_dir = list(sys.modules["veomni"].__path__)[0]
    _stub_package("veomni.models", os.path.join(pkg_dir, "models"))


def _attach_models_names() -> None:
    """Populate the ``veomni.models`` stub with the loaders ``torch_parallelize`` name-imports; must run first."""
    models_mod = sys.modules["veomni.models"]
    if hasattr(models_mod, "load_model_weights"):
        return
    module_utils = importlib.import_module("veomni.models.module_utils")
    models_mod.load_model_weights = module_utils.load_model_weights
    models_mod.rank0_load_and_broadcast_weights = module_utils.rank0_load_and_broadcast_weights


@functools.cache
def ensure_installed() -> None:
    """Install the ``sys.modules`` path stubs (cached, idempotent)."""
    _install_path_stubs()
    _attach_models_names()
    logger.info("veomni distributed layer installed via selective-import shim")


@functools.cache
def ensure_qwen3_moe_installed() -> None:
    """Register only VeOmni's Qwen3-MoE modeling implementation."""
    ensure_installed()
    pkg_dir = list(sys.modules["veomni"].__path__)[0]
    _stub_package(
        "veomni.models.transformers",
        os.path.join(pkg_dir, "models", "transformers"),
    )
    importlib.import_module("veomni.models.transformers.qwen3_moe")
    importlib.import_module("veomni.ops")
    ensure_attention_patch_installed()
    logger.info("veomni Qwen3-MoE modeling registered via selective import")


@functools.cache
def ensure_qwen3_5_moe_installed() -> None:
    """Register only VeOmni's Qwen3.5-MoE modeling implementation."""
    ensure_installed()
    pkg_dir = list(sys.modules["veomni"].__path__)[0]
    _stub_package(
        "veomni.models.transformers",
        os.path.join(pkg_dir, "models", "transformers"),
    )
    importlib.import_module("veomni.models.transformers.qwen3_5_moe")
    importlib.import_module("veomni.ops")
    ensure_attention_patch_installed()
    logger.info("veomni Qwen3.5-MoE modeling registered via selective import")


@functools.cache
def ensure_attention_patch_installed() -> None:
    """Register VeOmni's Ulysses SP attention into HF ``ALL_ATTENTION_FUNCTIONS``."""
    ensure_installed()
    pkg_dir = list(sys.modules["veomni"].__path__)[0]
    _stub_package("veomni.ops", os.path.join(pkg_dir, "ops"))
    _stub_package("veomni.ops.kernels", os.path.join(pkg_dir, "ops", "kernels"))
    from veomni.ops.kernels.attention import apply_veomni_attention_patch

    apply_veomni_attention_patch()
    logger.info("veomni Ulysses SP attention registered via selective-import shim")


def rank_world_local() -> Tuple[int, int, int]:
    """Resolve ``(rank, world_size, local_rank)`` from the actor env."""
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world, local


def ensure_dist_initialized(local_rank: Optional[int] = None) -> None:
    """Idempotently bring up the default process group."""
    import torch
    import torch.distributed as dist

    if torch.cuda.is_available() and local_rank is not None:
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group()
        logger.info(
            "ensure_dist_initialized: default process group up (rank=%s world=%s)",
            dist.get_rank(),
            dist.get_world_size(),
        )
