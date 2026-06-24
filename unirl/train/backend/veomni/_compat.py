"""Selective-import shim for the VeOmni distributed layer.

This module is the ONLY place in unirl that imports ``veomni``. VeOmni's
package roots have import-time behavior we must not run in unirl processes:

* ``veomni/__init__.py`` executes ``_apply_patches()`` (global kernel/ops
  monkey-patching) and eagerly imports the ops/kernel registry.
* ``veomni/models/__init__.py`` eagerly registers the model zoo, whose
  generated modeling files import transformers-5.9-only internals.

The torch-native layer we actually consume (``veomni.distributed.*``,
``veomni.arguments``) has no such coupling — ``veomni/distributed/__init__.py``
and ``veomni/utils/__init__.py`` are empty.  So instead of importing through
the package roots, we pre-seed ``sys.modules`` with *path stubs* for
``veomni`` and ``veomni.models``: package-shaped module objects whose
``__path__`` points at the installed tree but whose ``__init__`` code never
runs.  Submodule imports (``veomni.distributed.parallel_state`` etc.) then
resolve and execute normally.

One wrinkle: ``veomni/distributed/torch_parallelize.py`` does a *name*
import — ``from ..models import load_model_weights,
rank0_load_and_broadcast_weights`` — so the ``veomni.models`` stub must
already carry those attributes (sourced from ``veomni.models.module_utils``,
which is itself clean: safetensors + stable ``transformers.utils`` APIs)
before ``torch_parallelize`` is imported.  :func:`ensure_installed` runs
both stub steps in that order; call it before importing any veomni symbol.

Zero veomni functions are replaced; this is selective importing, not
behavior patching (the veomni dependency is exact-pinned in ``pyproject.toml``).
"""

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
    """Insert a package-shaped stub into ``sys.modules`` (idempotent).

    If ``name`` is already imported — real or stubbed — it is left alone:
    a real import means its init side effects already ran and re-stubbing
    would only desynchronize attribute state.
    """
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
    # NOTE: never importlib.util.find_spec("veomni.models") — resolving a
    # submodule spec imports the parent package for real.
    _stub_package("veomni.models", os.path.join(pkg_dir, "models"))


def _attach_models_names() -> None:
    """Populate the ``veomni.models`` stub with the loader functions that
    ``torch_parallelize`` name-imports from it (must run before that import).
    """
    models_mod = sys.modules["veomni.models"]
    if hasattr(models_mod, "load_model_weights"):
        return
    module_utils = importlib.import_module("veomni.models.module_utils")
    models_mod.load_model_weights = module_utils.load_model_weights
    models_mod.rank0_load_and_broadcast_weights = module_utils.rank0_load_and_broadcast_weights


@functools.cache
def ensure_installed() -> None:
    """Install the ``sys.modules`` path stubs (cached, idempotent).

    Runs *both* stub steps so that ``import veomni.distributed.*`` resolves
    without executing veomni's package-root init. ``_attach_models_names``
    must populate the ``veomni.models`` stub before anything imports
    ``veomni.distributed.torch_parallelize`` (which name-imports those
    attributes), so always run the pair together. Call this once before
    importing any veomni symbol; keep those imports lazy (function-local),
    since this module's importers load long before the stubs are installed.
    """
    _install_path_stubs()
    _attach_models_names()
    logger.info("veomni distributed layer installed via selective-import shim")


@functools.cache
def ensure_attention_patch_installed() -> None:
    """Register VeOmni's Ulysses SP attention into HF ``ALL_ATTENTION_FUNCTIONS``.

    A model whose ``config._attn_implementation`` is
    ``"veomni_flash_attention_2_with_sp"`` then routes attention through
    VeOmni's Ulysses all-to-all wrapper (a no-op when ``ulysses_size == 1``).

    Selective import, same discipline as :func:`ensure_installed`: we stub
    ``veomni.ops`` and ``veomni.ops.kernels`` so their eager package inits never
    run — ``veomni/ops/__init__.py`` imports liger / moe / cross-entropy kernels
    (heavy, and absent under the ``--no-deps`` veomni install), none of which the
    attention patch needs. ``veomni.ops.kernels.attention`` itself has clean deps
    (transformers + ``veomni.distributed``). Cached / idempotent.
    """
    ensure_installed()
    pkg_dir = list(sys.modules["veomni"].__path__)[0]
    _stub_package("veomni.ops", os.path.join(pkg_dir, "ops"))
    _stub_package("veomni.ops.kernels", os.path.join(pkg_dir, "ops", "kernels"))
    from veomni.ops.kernels.attention import apply_veomni_attention_patch

    apply_veomni_attention_patch()
    logger.info("veomni Ulysses SP attention registered via selective-import shim")


def rank_world_local() -> Tuple[int, int, int]:
    """Resolve ``(rank, world_size, local_rank)`` from the actor env.

    Backends are constructed by ``Worker.add_remote`` *before*
    ``Remote.setup()`` runs, so ctor kwargs carry no live rank info — but
    DevicePool bakes ``RANK``/``WORLD_SIZE``/``MASTER_ADDR``/``MASTER_PORT``
    into each worker's runtime env at spawn, and masks ``CUDA_VISIBLE_DEVICES``
    to exactly one GPU (so the local device index is 0 unless ``LOCAL_RANK``
    says otherwise).
    """
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    return rank, world, local


def ensure_dist_initialized(local_rank: Optional[int] = None) -> None:
    """Idempotently bring up the default process group.

    UniRL itself never calls ``dist.init_process_group`` on the train side —
    today torch's ``fully_shard`` lazily auto-inits it (no-arg ``env://``
    rendezvous over the DevicePool-baked env).  VeOmni's
    ``init_parallel_state`` builds device meshes *before* any ``fully_shard``
    call, so the lazy path never fires for this backend — replicate the
    exact same no-arg init explicitly.
    """
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
