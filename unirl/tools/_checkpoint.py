"""Offline reader for the two checkpoint flavors ``BaseFSDP2Backend.save`` writes."""

from __future__ import annotations

import os
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
            import torch.distributed.checkpoint as dcp
            from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
            from torch.distributed.checkpoint.state_dict_loader import _load_state_dict

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
            checkpoint.update(torch.load(app_metadata, map_location="cpu", weights_only=True))
            checkpoint["_checkpoint_format"] = "dcp"
            return checkpoint
        if os.path.exists(app_metadata) or any(
            name.endswith(".distcp") or name == ".metadata.tmp" for name in os.listdir(path)
        ):
            raise RuntimeError(f"incomplete DCP checkpoint at {path!r}: async shard flush has not published .metadata")
        path = file_path
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    checkpoint["_checkpoint_format"] = "torch"
    return checkpoint
