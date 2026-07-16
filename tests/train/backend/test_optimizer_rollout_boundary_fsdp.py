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

from unirl.train.backend.base_backend import BaseFSDP2Backend


class _TinyFSDPModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input = nn.Linear(8, 16)
        self.output = nn.Linear(16, 4)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.output(torch.nn.functional.silu(self.input(value)))


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    to_local = getattr(tensor, "to_local", None)
    return to_local() if callable(to_local) else tensor


def _state_slots(
    optimizer: torch.optim.Optimizer,
) -> list[tuple[dict[str, Any], str, torch.Tensor]]:
    return [
        (state, key, value)
        for state in optimizer.state.values()
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
    ]


def _optimizer_state_local_nbytes(
    optimizer: torch.optim.Optimizer,
    *,
    keys: set[str],
) -> int:
    return sum(
        _local_tensor(value).numel() * _local_tensor(value).element_size()
        for _state, key, value in _state_slots(optimizer)
        if key in keys
    )


def _loss(model: nn.Module, *, rank: int, phase: int, device: torch.device) -> torch.Tensor:
    value = torch.arange(32, device=device, dtype=torch.float32).reshape(4, 8)
    value = value / 31.0 + rank * 0.07 + phase * 0.03
    target = torch.linspace(-0.4, 0.6, 16, device=device).reshape(4, 4)
    return (model(value) - target).square().mean()


def _run_optimizer_boundary_fsdp_worker(
    rank: int,
    store_path: str,
    result_queue: Any,
) -> None:
    import torch.distributed as dist

    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard

        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
        dist.init_process_group(
            "nccl",
            init_method=f"file://{store_path}",
            rank=rank,
            world_size=2,
            timeout=datetime.timedelta(seconds=60),
        )
        mesh = init_device_mesh("cuda", (2,))

        torch.manual_seed(4107)
        model = _TinyFSDPModel().to(device)
        fully_shard(model, mesh=mesh, reshard_after_forward=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, foreach=False)

        # Materialize AdamW state, then leave a completed FSDP2 gradient for the
        # rollout boundary to release.
        _loss(model, rank=rank, phase=0, device=device).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        _loss(model, rank=rank, phase=1, device=device).backward()

        parameters = tuple(model.parameters())
        parameter_ids = tuple(id(parameter) for parameter in parameters)
        parameter_shards = tuple(_local_tensor(parameter.detach()).clone() for parameter in parameters)
        parameter_devices = tuple(_local_tensor(parameter).device for parameter in parameters)
        assert parameter_devices and all(value == device for value in parameter_devices)
        assert all(parameter.grad is not None for parameter in parameters)
        assert all(_local_tensor(parameter.grad).device == device for parameter in parameters)

        slots = _state_slots(optimizer)
        step_slots = [(state, key, value) for state, key, value in slots if key == "step"]
        moment_slots = [(state, key, value) for state, key, value in slots if key in {"exp_avg", "exp_avg_sq"}]
        assert len(step_slots) == len(parameters)
        assert len(moment_slots) == 2 * len(parameters)

        # This is torch 2.11 AdamW's non-capturable state layout: step is a
        # CPU-native scalar while both moment buffers follow the parameter.
        assert all(value.shape == () and _local_tensor(value).device.type == "cpu" for _, _, value in step_slots)
        assert all(_local_tensor(value).device == device for _, _, value in moment_slots)
        step_ids = tuple(id(value) for _, _, value in step_slots)
        step_values = tuple(value.item() for _, _, value in step_slots)
        moment_values = {
            (id(state), key): _local_tensor(value).detach().cpu().clone() for state, key, value in moment_slots
        }
        expected_moment_bytes = _optimizer_state_local_nbytes(
            optimizer,
            keys={"exp_avg", "exp_avg_sq"},
        )

        backend = BaseFSDP2Backend.__new__(BaseFSDP2Backend)
        backend._device = device
        backend.model = model
        backend.optimizer = optimizer

        first_park = backend.park_optimizer_state_for_rollout()
        assert all(parameter.grad is None for parameter in parameters)
        assert first_park["grad_bytes_cleared"] > 0
        assert first_park["optimizer_state_bytes_parked"] == expected_moment_bytes
        assert all(_local_tensor(state[key]).device.type == "cpu" for state, key, _value in moment_slots)
        assert tuple(id(state["step"]) for state, _, _ in step_slots) == step_ids
        assert all(state["step"].device.type == "cpu" for state, _, _ in step_slots)

        # Simulate an interrupted lifecycle which moved one recorded moment back
        # early. A repeated park must re-park it without duplicating or losing
        # the pending restoration record.
        partial_state, partial_key, _partial_value = moment_slots[0]
        partial_state[partial_key] = partial_state[partial_key].to(device)
        repeated_park = backend.park_optimizer_state_for_rollout()
        assert repeated_park["grad_bytes_cleared"] == 0
        assert _local_tensor(partial_state[partial_key]).device.type == "cpu"
        assert all(_local_tensor(state[key]).device.type == "cpu" for state, key, _value in moment_slots)

        # Likewise, make restoration partially complete before invoking the
        # public operation. It must restore only what remains and converge to
        # the exact original device layout.
        partial_state[partial_key] = partial_state[partial_key].to(device)
        first_restore = backend.restore_optimizer_state_after_rollout()
        assert 0 < first_restore["optimizer_state_bytes_restored"] <= expected_moment_bytes
        assert all(_local_tensor(state[key]).device == device for state, key, _value in moment_slots)
        assert all(state["step"].device.type == "cpu" for state, _, _ in step_slots)
        repeated_restore = backend.restore_optimizer_state_after_rollout()
        assert repeated_restore["optimizer_state_bytes_restored"] == 0

        # The boundary cannot mutate or replace FSDP parameter objects/shards.
        assert tuple(id(parameter) for parameter in model.parameters()) == parameter_ids
        assert all(
            _local_tensor(parameter).device == original_device
            and torch.equal(_local_tensor(parameter.detach()), original_shard)
            for parameter, original_device, original_shard in zip(
                model.parameters(), parameter_devices, parameter_shards
            )
        )
        assert tuple(id(state["step"]) for state, _, _ in step_slots) == step_ids
        assert tuple(state["step"].item() for state, _, _ in step_slots) == step_values
        assert all(
            torch.equal(_local_tensor(state[key]).detach().cpu(), moment_values[(id(state), key)])
            for state, key, _value in moment_slots
        )

        # A real FSDP2 forward/backward and AdamW step is the final device-layout
        # oracle. In particular, it fails if CPU-native steps were moved to CUDA
        # or any moment was left parked.
        before_step = tuple(_local_tensor(parameter.detach()).clone() for parameter in model.parameters())
        _loss(model, rank=rank, phase=2, device=device).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        assert all(parameter.grad is None for parameter in model.parameters())
        assert all(state["step"].device.type == "cpu" for state, _, _ in step_slots)
        assert all(state["step"].item() == old_step + 1 for (state, _, _), old_step in zip(step_slots, step_values))
        assert any(
            not torch.equal(_local_tensor(parameter.detach()), old_shard)
            for parameter, old_shard in zip(model.parameters(), before_step)
        )

        dist.barrier()
        result_queue.put((rank, "ok", torch.__version__, expected_moment_bytes))
    except Exception as error:
        result_queue.put((rank, "error", type(error).__name__, str(error), traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_optimizer_only_rollout_parking_preserves_fsdp2_cuda_state_layout() -> None:
    dist = torch.distributed
    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() < 2
        or not dist.is_available()
        or not dist.is_nccl_available()
    ):
        pytest.skip("requires two CUDA devices and torch.distributed with NCCL")
    try:
        from torch.distributed.fsdp import fully_shard as _fully_shard  # noqa: F401
    except ImportError:
        pytest.skip("requires FSDP2")

    context = mp.get_context("spawn")
    result_queue = context.Queue()
    with tempfile.TemporaryDirectory() as temp_dir:
        store_path = str(Path(temp_dir) / "store")
        processes = [
            context.Process(
                target=_run_optimizer_boundary_fsdp_worker,
                args=(rank, store_path, result_queue),
            )
            for rank in range(2)
        ]
        for process in processes:
            process.start()
        deadline = time.monotonic() + 90
        for process in processes:
            process.join(max(0.0, deadline - time.monotonic()))
        hanging = [process for process in processes if process.is_alive()]
        for process in hanging:
            process.kill()
        for process in hanging:
            process.join(5)
        if hanging:
            pytest.fail("two-rank optimizer-boundary FSDP2 test exceeded the 90-second deadline")

        results = []
        for _ in processes:
            try:
                results.append(result_queue.get(timeout=2))
            except queue.Empty:
                break

    assert len(results) == 2, f"missing worker result; exit codes={[process.exitcode for process in processes]}"
    errors = [result for result in results if result[1] != "ok"]
    assert not errors, "\n".join(str(error) for error in errors)
    assert all(result[2].startswith("2.11.0") for result in results)
    assert all(result[3] > 0 for result in results)
