from types import SimpleNamespace

import torch
from torch import nn

from unirl.train.backend.veomni.ep.models.qwen3_moe import fused_expert_kind
from unirl.utils import peft_merge


class _PackedExperts(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.zeros(2, 4, 3))
        self.down_proj = nn.Parameter(torch.zeros(2, 3, 2))


class _PackedQwenMoe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Module()
        self.layer.mlp = nn.Module()
        self.layer.mlp.experts = _PackedExperts()
        self.config = SimpleNamespace(
            model_type="qwen3_moe",
            moe_intermediate_size=2,
            tie_word_embeddings=False,
        )


def test_train_side_ep_defers_qwen_expert_unpack_until_ep_gather(monkeypatch) -> None:
    model = _PackedQwenMoe()
    monkeypatch.setattr(
        peft_merge,
        "_to_full_tensor",
        lambda tensor, dtype=None: tensor,
    )

    ordinary_names = [name for name, _ in peft_merge.raw_state_dict(model)]
    assert ordinary_names == [
        "layer.mlp.experts.0.gate_proj.weight",
        "layer.mlp.experts.0.up_proj.weight",
        "layer.mlp.experts.1.gate_proj.weight",
        "layer.mlp.experts.1.up_proj.weight",
        "layer.mlp.experts.0.down_proj.weight",
        "layer.mlp.experts.1.down_proj.weight",
    ]

    model._extra_parallel_param_groups = {
        "ep": [
            model.layer.mlp.experts.gate_up_proj,
            model.layer.mlp.experts.down_proj,
        ]
    }
    ep_names = [name for name, _ in peft_merge.raw_state_dict(model)]
    assert ep_names == [
        "layer.mlp.experts.gate_up_proj",
        "layer.mlp.experts.down_proj",
    ]
    assert all(fused_expert_kind(name) is not None for name in ep_names)
