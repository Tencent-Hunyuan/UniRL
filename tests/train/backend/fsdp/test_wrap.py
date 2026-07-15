from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed._composable as composable
import torch.distributed.fsdp as torch_fsdp
from torch import nn

from unirl.train.backend.fsdp.wrap import fsdp_wrap


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(2))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.weight * hidden_states


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Block(), _Block()])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


def test_activation_checkpointing_composes_before_fully_shard_without_renaming(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _Model()
    events: list[tuple[str, int]] = []

    def fake_checkpoint(module: nn.Module, **_kwargs: object) -> nn.Module:
        events.append(("checkpoint", id(module)))
        return module

    def fake_fully_shard(module: nn.Module, **_kwargs: object) -> nn.Module:
        events.append(("fully_shard", id(module)))
        return module

    monkeypatch.setattr(composable, "checkpoint", fake_checkpoint)
    monkeypatch.setattr(torch_fsdp, "fully_shard", fake_fully_shard)

    state_keys = tuple(model.state_dict())
    parameter_names = tuple(name for name, _ in model.named_parameters())
    fsdp_wrap(
        model,
        block_class_names=("_Block",),
        activation_checkpointing=True,
        root_wrap=False,
    )

    block_ids = [id(layer) for layer in model.layers]
    assert events == [("checkpoint", block_id) for block_id in block_ids] + [
        ("fully_shard", block_id) for block_id in block_ids
    ]
    assert tuple(model.state_dict()) == state_keys
    assert tuple(name for name, _ in model.named_parameters()) == parameter_names
    assert all("forward" not in layer.__dict__ for layer in model.layers)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.distributed.is_nccl_available(),
    reason="requires one CUDA device with NCCL",
)
def test_activation_checkpoint_recompute_unshards_cpu_offloaded_fsdp_block() -> None:
    if torch.distributed.is_initialized():
        pytest.skip("requires an isolated process group")

    with tempfile.TemporaryDirectory() as tmpdir:
        store = Path(tmpdir) / "store"
        torch.distributed.init_process_group(
            "nccl",
            init_method=f"file://{store}",
            rank=0,
            world_size=1,
        )
        try:
            torch.cuda.set_device(0)
            model = _Model()
            calls = [0, 0]
            for index, layer in enumerate(model.layers):
                original_forward = layer.forward

                def counted_forward(
                    hidden_states: torch.Tensor,
                    *,
                    _index: int = index,
                    _forward=original_forward,
                ) -> torch.Tensor:
                    calls[_index] += 1
                    return _forward(hidden_states)

                layer.forward = counted_forward

            fsdp_wrap(
                model,
                block_class_names=("_Block",),
                cpu_offload=True,
                master_dtype="fp32",
                activation_checkpointing=True,
                reshard_after_forward=True,
                root_wrap=False,
            )

            hidden_states = torch.randn(
                4,
                2,
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            model(hidden_states).float().square().mean().backward()

            assert calls == [2, 2]
            assert tuple(model.state_dict()) == ("layers.0.weight", "layers.1.weight")
            for parameter in model.parameters():
                assert parameter.device.type == "cpu"
                assert parameter.grad is not None
                assert torch.isfinite(parameter.grad.to_local()).all()
        finally:
            torch.distributed.destroy_process_group()
