"""Small EP checkpoint round-trip using synthetic expert tensors.

This exercises the production single-file model + Adam-state helpers without a
large MoE checkpoint or fused kernels. It defaults to CPU/Gloo; set
``DEVICE=cuda`` for NCCL:

    torchrun --nproc_per_node=4 scripts/ep_verify/unirl_ep_checkpoint_roundtrip.py \
        2 /tmp/unirl_ep_checkpoint_roundtrip.pt
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist
from torch import nn

from unirl.train.backend.veomni.ep.checkpoint import (
    gather_ep_model_state_dict,
    gather_ep_optimizer_state_dict,
    load_ep_model_state_dict,
    load_ep_optimizer_state_dict,
)


class _TinyExpertModel(nn.Module):
    def __init__(self, expert_param: nn.Parameter) -> None:
        super().__init__()
        self.experts = expert_param
        self._extra_parallel_param_groups = {
            "ep": [self.experts],
            "non_extra_parallel": [],
        }


def main() -> None:
    ep_size = int(sys.argv[1])
    checkpoint_path = sys.argv[2]
    local_rank = int(os.environ["LOCAL_RANK"])
    device_type = os.environ.get("DEVICE", "cpu").strip().lower()
    if device_type == "cuda":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    elif device_type == "cpu":
        device = torch.device("cpu")
        backend = "gloo"
    else:
        raise ValueError(f"DEVICE must be 'cpu' or 'cuda', got {device_type!r}")
    dist.init_process_group(backend)

    from unirl.train.backend.veomni import _compat

    _compat.ensure_installed()
    from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state

    world = dist.get_world_size()
    if world % ep_size:
        raise ValueError(f"world_size={world} must be divisible by ep_size={ep_size}")
    init_parallel_state(
        dp_size=world,
        ulysses_size=1,
        dp_mode="fsdp2",
        device_type=device_type,
        extra_parallel_sizes=(ep_size,),
        extra_parallel_names=("ep",),
    )
    ps = get_parallel_state()
    ep_rank = int(ps.extra_parallel_rank("ep"))

    from torch.distributed.tensor import Replicate, Shard, distribute_tensor

    # Match VeOmni exactly: the outer EP split is already applied to this
    # rank's [E/ep,H] block; the DTensor records only the inner ep_fsdp shard.
    local_expert_block = torch.full((2, 4), float(ep_rank + 1), device=device)
    full_ep_mesh = ps.extra_parallel_fsdp_device_mesh["ep"]
    ep_fsdp_mesh = full_ep_mesh["ep_fsdp"]
    assert ep_fsdp_mesh.mesh_dim_names == ("ep_fsdp",)
    placements = [Replicate()] * (ep_fsdp_mesh.ndim - 1) + [Shard(1)]
    expert_dtensor = distribute_tensor(local_expert_block, ep_fsdp_mesh, placements)
    model = _TinyExpertModel(nn.Parameter(expert_dtensor))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, foreach=False)
    model.experts.grad = torch.full_like(model.experts, 0.1 * (ep_rank + 1))
    optimizer.step()

    expected_param = model.experts.to_local().detach().clone()
    expected_optim = {
        key: (value.to_local() if hasattr(value, "to_local") else value).detach().clone()
        for key, value in optimizer.state[model.experts].items()
        if isinstance(value, torch.Tensor)
    }

    model_state = gather_ep_model_state_dict(model)
    optimizer_state = gather_ep_optimizer_state_dict(model, optimizer)
    if dist.get_rank() == 0:
        local_global_shape = tuple(model.experts.shape)
        expected_shape = (local_global_shape[0] * ep_size, *local_global_shape[1:])
        assert tuple(model_state["experts"].shape) == expected_shape
        assert tuple(optimizer_state["state"]["experts"]["exp_avg"].shape) == expected_shape
        blocks = model_state["experts"].reshape(ep_size, local_global_shape[0], *local_global_shape[1:])
        assert any(not torch.equal(blocks[0], blocks[index]) for index in range(1, ep_size))
        torch.save({"model": model_state, "optimizer": optimizer_state}, checkpoint_path)
    dist.barrier()

    with torch.no_grad():
        model.experts.zero_()
        for value in optimizer.state[model.experts].values():
            if isinstance(value, torch.Tensor):
                value.zero_()

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    load_ep_model_state_dict(model, checkpoint["model"], strict=True)
    load_ep_optimizer_state_dict(model, optimizer, checkpoint["optimizer"])

    torch.testing.assert_close(model.experts.to_local(), expected_param, rtol=0, atol=0)
    for key, expected in expected_optim.items():
        actual = optimizer.state[model.experts][key]
        actual = actual.to_local() if hasattr(actual, "to_local") else actual
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    dist.barrier()
    if dist.get_rank() == 0:
        print(f"EP checkpoint round-trip PASS (world={world}, ep={ep_size})", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
