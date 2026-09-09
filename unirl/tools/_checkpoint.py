"""Offline reader for the two checkpoint flavors ``BaseFSDP2Backend.save`` writes."""

from __future__ import annotations

import os
import pickle
from typing import Dict

import torch


def load_training_checkpoint(path: str) -> Dict[str, object]:
    """Load a UniRL checkpoint (``torch`` or ``dcp``) into the legacy dict shape."""
    if os.path.isdir(path):
        dcp_metadata = os.path.join(path, ".metadata")
        app_metadata = os.path.join(path, "metadata.pt")
        file_path = os.path.join(path, "checkpoint.pt")
        if os.path.exists(dcp_metadata):
            if not os.path.exists(app_metadata):
                raise RuntimeError(f"incomplete DCP checkpoint: missing {app_metadata}")
            return _load_dcp(path)
        if os.path.exists(app_metadata) or any(
            name.endswith(".distcp") or name == ".metadata.tmp" for name in os.listdir(path)
        ):
            raise RuntimeError(f"incomplete DCP checkpoint at {path!r}: async shard flush has not published .metadata")
        if os.path.exists(file_path):
            checkpoint = _torch_load(file_path)
            checkpoint["_checkpoint_format"] = "torch"
            return checkpoint
        checkpoint = _torch_load(file_path)
    else:
        checkpoint = _torch_load(path)
    checkpoint["_checkpoint_format"] = "torch"
    return checkpoint


def _torch_load(file_path: str, *, allow_unsafe_fallback: bool = True) -> Dict[str, object]:
    try:
        return torch.load(file_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(file_path, map_location="cpu")
    except pickle.UnpicklingError:
        if not allow_unsafe_fallback:
            raise
        return torch.load(file_path, map_location="cpu", weights_only=False)
    except RuntimeError as exc:
        if "Weights only load failed" not in str(exc) or not allow_unsafe_fallback:
            raise
        return torch.load(file_path, map_location="cpu", weights_only=False)


def _load_dcp(path: str) -> Dict[str, object]:
    try:
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
        from torch.distributed.checkpoint.state_dict_loader import _load_state_dict
    except ImportError as exc:
        raise RuntimeError(f"DCP export requires PyTorch >= 2.4; installed version is {torch.__version__}") from exc

    sharded: Dict[str, object] = {}
    _load_state_dict(
        sharded,
        storage_reader=dcp.FileSystemReader(path),
        planner=_EmptyStateDictLoadPlanner(keys={"model"}),
        no_dist=True,
    )
    model_state = sharded.get("model")
    if not isinstance(model_state, dict):
        raise RuntimeError(f"DCP checkpoint at {path!r} has no model state")
    checkpoint: Dict[str, object] = {"policy_state_dict": model_state}
    meta_path = os.path.join(path, "metadata.pt")
    checkpoint.update(_torch_load(meta_path, allow_unsafe_fallback=False))
    checkpoint["_checkpoint_format"] = "dcp"
    return checkpoint
