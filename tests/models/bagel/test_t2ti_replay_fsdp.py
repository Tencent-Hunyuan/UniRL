from __future__ import annotations

import datetime
import multiprocessing as mp
import queue
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn


class _ReplayBlock(nn.Module):
    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.linear = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.norm(self.linear(hidden)) + hidden)


class _ReplayDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_ReplayBlock(), _ReplayBlock(), _ReplayBlock()])
        self.register_buffer("initial_hidden", torch.randn(1, 4))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


def _replay_loss(model: _ReplayDecoder, *, rank: int, pad_to_equal_depth: bool) -> torch.Tensor:
    real_depth = 5 if rank == 0 else 3
    hidden = model.initial_hidden
    for index in range(real_depth):
        hidden = model(hidden + (index + 1) * 0.01)
    real_terminal = hidden

    pad_zero = torch.zeros((), dtype=hidden.dtype)
    if pad_to_equal_depth and real_depth < 5:
        # Mirrors BAGEL's bounded-memory padding: a fresh one-token hidden state,
        # a zero-valued graph edge from the real cache, then sequential no-cache
        # decoder traversals. The semantic branch is created after this suffix.
        dummy_hidden = model.initial_hidden + real_terminal.reshape(-1)[0].float() * 0.0
        for _ in range(5 - real_depth):
            dummy_hidden = model(dummy_hidden)
        pad_zero = dummy_hidden.reshape(-1)[0].float() * 0.0

    semantic = model(real_terminal + 0.125)
    return semantic.float().square().mean() + pad_zero


def _run_fsdp_replay_worker(rank: int, store_path: str, result_queue: Any) -> None:
    import torch.distributed as dist

    try:
        from torch.distributed._composable import checkpoint
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard

        dist.init_process_group(
            "gloo",
            init_method=f"file://{store_path}",
            rank=rank,
            world_size=2,
            timeout=datetime.timedelta(seconds=15),
        )
        mesh = init_device_mesh("cpu", (2,))

        torch.manual_seed(1234)
        reference = _ReplayDecoder()
        model = _ReplayDecoder()
        model.load_state_dict(reference.state_dict())

        # Padding must be mathematically inert. Compare FSDP's reduced gradient
        # with the mean of the two unpadded rank-local reference gradients.
        _replay_loss(reference, rank=rank, pad_to_equal_depth=False).backward()
        expected_gradients = []
        for parameter in reference.layers.parameters():
            gradient = parameter.grad.detach().clone()
            dist.all_reduce(gradient)
            expected_gradients.append((gradient / 2).reshape(-1))

        for layer in model.layers:
            checkpoint(layer)
        for layer in model.layers:
            fully_shard(layer, mesh=mesh, reshard_after_forward=True)

        loss = _replay_loss(model, rank=rank, pad_to_equal_depth=True)
        loss.backward()

        actual_gradients = []
        for parameter in model.layers.parameters():
            assert parameter.grad is not None
            actual_gradients.append(parameter.grad.full_tensor().detach().reshape(-1))

        actual = torch.cat(actual_gradients)
        expected = torch.cat(expected_gradients)
        dist.barrier()
        result_queue.put(
            (
                rank,
                "ok",
                float((actual - expected).abs().max().item()),
                float(actual.norm().item()),
                bool(torch.isfinite(actual).all().item()),
            )
        )
    except Exception as error:
        result_queue.put((rank, "error", type(error).__name__, str(error), traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_dependent_dummy_suffix_preserves_fsdp2_collective_order_and_gradients() -> None:
    dist = torch.distributed
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("requires torch.distributed with Gloo")
    try:
        from torch.distributed._composable import checkpoint as _checkpoint  # noqa: F401
        from torch.distributed.fsdp import fully_shard as _fully_shard  # noqa: F401
    except ImportError:
        pytest.skip("requires composable checkpointing and FSDP2")

    context = mp.get_context("spawn")
    result_queue = context.Queue()
    with tempfile.TemporaryDirectory() as temp_dir:
        store_path = str(Path(temp_dir) / "store")
        processes = [
            context.Process(target=_run_fsdp_replay_worker, args=(rank, store_path, result_queue))
            for rank in range(2)
        ]
        for process in processes:
            process.start()

        deadline = time.monotonic() + 25
        for process in processes:
            process.join(max(0.0, deadline - time.monotonic()))

        hanging = [process for process in processes if process.is_alive()]
        for process in hanging:
            process.kill()
        for process in hanging:
            process.join(5)
        if hanging:
            pytest.fail("two-rank FSDP2 replay did not finish before the 25-second deadline")

        results = []
        for _ in processes:
            try:
                results.append(result_queue.get(timeout=2))
            except queue.Empty:
                break

    assert len(results) == 2, f"missing worker result; exit codes={[p.exitcode for p in processes]}"
    errors = [result for result in results if result[1] != "ok"]
    assert not errors, "\n".join(str(error) for error in errors)

    results.sort()
    for _rank, _status, max_error, gradient_norm, finite in results:
        assert finite
        assert gradient_norm > 0.0
        assert max_error <= 1e-6
    assert results[0][3] == pytest.approx(results[1][3], rel=1e-6, abs=1e-7)
