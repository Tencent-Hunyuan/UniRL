from __future__ import annotations

import os
import tempfile
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from unirl.train.backend.base_backend import BaseFSDP2Backend


def test_move_unsharded_model_state_leaves_dtensor_shards_on_cpu() -> None:
    if not dist.is_available():
        pytest.skip("torch.distributed is unavailable")
    if dist.is_initialized():
        pytest.skip("test needs ownership of a one-rank process group")

    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Shard

    rendezvous = tempfile.NamedTemporaryFile(delete=False)
    rendezvous.close()
    try:
        dist.init_process_group(
            "gloo",
            init_method=f"file://{rendezvous.name}",
            rank=0,
            world_size=1,
        )
        mesh = init_device_mesh("cpu", (1,))
        shard = DTensor.from_local(
            torch.ones(2),
            mesh,
            [Shard(0)],
            run_check=False,
        )
        model = torch.nn.Module()
        model.register_parameter("regular", torch.nn.Parameter(torch.ones(2)))
        model.register_parameter("sharded", torch.nn.Parameter(shard))

        backend = BaseFSDP2Backend.__new__(BaseFSDP2Backend)
        backend.model = model
        backend._move_unsharded_model_state("meta")

        assert model.regular.is_meta
        assert isinstance(model.sharded, DTensor)
        assert model.sharded.to_local().device.type == "cpu"
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        if os.path.exists(rendezvous.name):
            os.unlink(rendezvous.name)


def test_move_bundle_module_state_includes_frozen_auxiliary_modules() -> None:
    transformer = torch.nn.Linear(2, 2)
    parent = torch.nn.Module()
    parent.add_module("language_model", transformer)
    parent.register_parameter("generation_head", torch.nn.Parameter(torch.ones(2), requires_grad=False))
    vae = torch.nn.Linear(2, 2)

    backend = BaseFSDP2Backend.__new__(BaseFSDP2Backend)
    backend.model = transformer
    backend._bundle = SimpleNamespace(model=parent, transformer=transformer, vae=vae)
    backend._move_bundle_module_state("meta")

    assert transformer.weight.is_meta
    assert parent.generation_head.is_meta
    assert vae.weight.is_meta
