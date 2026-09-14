"""Multi-GPU FSDP2 copy-engine correctness and timing smoke.

Run with torchrun. The process group must span whole HSDP shard groups.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard


class Block(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.up = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.down = nn.Linear(hidden_size * 2, hidden_size, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.down(torch.nn.functional.gelu(self.up(inputs)))


class Model(nn.Module):
    def __init__(self, hidden_size: int, num_blocks: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(Block(hidden_size) for _ in range(num_blocks))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            inputs = block(inputs)
        return inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--copy-engine", action="store_true")
    parser.add_argument("--shard-size", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--num-blocks", type=int, default=6)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--trace-path", type=Path)
    return parser.parse_args()


def train_step(model: nn.Module, optimizer: torch.optim.Optimizer, inputs: torch.Tensor) -> float:
    optimizer.zero_grad(set_to_none=True)
    output = model(inputs)
    loss = output.float().square().mean()
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def main() -> None:
    args = parse_args()
    if args.copy_engine:
        os.environ["NCCL_CTA_POLICY"] = "2"

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size % args.shard_size:
        raise ValueError(f"world_size={world_size} is not divisible by shard_size={args.shard_size}")

    torch.manual_seed(1234)
    model = Model(args.hidden_size, args.num_blocks).to(device=device, dtype=torch.bfloat16)
    mesh = init_device_mesh(
        "cuda",
        (world_size // args.shard_size, args.shard_size),
        mesh_dim_names=("dp_replicate", "dp_shard"),
    )
    for block in model.blocks:
        fully_shard(block, mesh=mesh)
        if args.copy_engine:
            block.set_symm_mem_for_comm("NCCL")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    generator = torch.Generator(device=device).manual_seed(5678)
    inputs = torch.randn(2, 32, args.hidden_size, device=device, dtype=torch.bfloat16, generator=generator)

    for _ in range(args.warmup_steps):
        train_step(model, optimizer, inputs)
    dist.barrier()
    torch.cuda.reset_peak_memory_stats(device)

    durations_ms: list[float] = []
    losses: list[float] = []
    for _ in range(args.steps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        losses.append(train_step(model, optimizer, inputs))
        end.record()
        end.synchronize()
        durations_ms.append(float(start.elapsed_time(end)))

    if args.trace_path is not None:
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(activities=activities) as profile:
            train_step(model, optimizer, inputs)
        if rank == 0:
            args.trace_path.parent.mkdir(parents=True, exist_ok=True)
            profile.export_chrome_trace(str(args.trace_path))

    local_checksum = sum(float(param.detach().float().sum()) for param in model.parameters())
    checksums: list[float | None] = [None] * world_size
    dist.all_gather_object(checksums, local_checksum)
    replica_spread = 0.0
    for shard_rank in range(args.shard_size):
        replicas = [
            float(checksums[replica_rank * args.shard_size + shard_rank])
            for replica_rank in range(world_size // args.shard_size)
        ]
        replica_spread = max(replica_spread, max(replicas) - min(replicas))

    if not all(torch.isfinite(torch.tensor(losses))):
        raise RuntimeError(f"non-finite loss: {losses}")
    if replica_spread != 0.0:
        raise RuntimeError(f"HSDP replicas diverged: max checksum spread={replica_spread}")

    if rank == 0:
        print(
            "FSDP_COPY_ENGINE_RESULT "
            + json.dumps(
                {
                    "copy_engine": args.copy_engine,
                    "world_size": world_size,
                    "shard_size": args.shard_size,
                    "median_step_ms": statistics.median(durations_ms),
                    "min_step_ms": min(durations_ms),
                    "max_step_ms": max(durations_ms),
                    "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                    "final_loss": losses[-1],
                    "replica_checksum_spread": replica_spread,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
