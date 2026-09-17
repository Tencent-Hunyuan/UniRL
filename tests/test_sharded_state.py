import torch
from torch import nn

from unirl.train.backend.sharded_state import gather_optimizer_state_dict, load_optimizer_state_dict


def test_cold_optimizer_load_restores_checkpoint_lr() -> None:
    source = nn.Linear(2, 1)
    source_optimizer = torch.optim.AdamW(source.parameters(), lr=1e-3)
    state_dict = gather_optimizer_state_dict(source, source_optimizer)

    assert not source_optimizer.state
    assert all(float(entry["step"]) == 0 for entry in state_dict["state"].values())

    target = nn.Linear(2, 1)
    target_optimizer = torch.optim.AdamW(target.parameters(), lr=0.123)
    load_optimizer_state_dict(target, target_optimizer, state_dict, broadcast_from_rank0=False)

    assert target_optimizer.param_groups[0]["lr"] == 1e-3
    assert not target_optimizer.state


def test_rank0_broadcast_keeps_loaded_warm_state(monkeypatch) -> None:
    model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def fake_set_optimizer_state_dict(model, optimizer, optim_state_dict, *, options=None) -> None:
        del model, optim_state_dict, options
        for param in optimizer.param_groups[0]["params"]:
            optimizer.state[param] = {
                "step": torch.tensor(3.0),
                "exp_avg": torch.zeros_like(param),
                "exp_avg_sq": torch.zeros_like(param),
            }

    monkeypatch.setattr(
        torch.distributed.checkpoint.state_dict,
        "set_optimizer_state_dict",
        fake_set_optimizer_state_dict,
    )

    load_optimizer_state_dict(model, optimizer, {}, broadcast_from_rank0=True)

    assert len(optimizer.state) == 2
    assert all(float(entry["step"]) == 3 for entry in optimizer.state.values())
