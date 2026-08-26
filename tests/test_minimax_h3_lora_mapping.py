"""MiniMax-H3 trainer-to-rollout LoRA layout tests."""

import pytest
import torch

from unirl.distributed.weight_sync.lora.base import LoraWeightSyncBase
from unirl.rollout.engine.vllm_omni.worker.ipc_receive_mixin import (
    BucketedIPCReceiveMixin,
    remap_minimax_h3_lora,
    validate_minimax_h3_lora_coverage,
)


def test_h3_fc1_lora_swaps_diffusers_up_gate_halves() -> None:
    prefix = "transformer.transformer_blocks.0.ff.net.0.proj"
    lora_a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    up = torch.full((4, 2), 11.0)
    gate = torch.full((4, 2), 29.0)
    tensors = {
        f"{prefix}.lora_A.default.weight": lora_a,
        f"{prefix}.lora_B.default.weight": torch.cat((up, gate), dim=0),
    }
    config = {
        "target_modules": (
            r"^transformer_blocks\.(?:.*\.)?"
            r"(?:attn\.to_q|attn\.to_k|attn\.to_v|attn\.to_out\.0|ff\.net\.0\.proj|ff\.net\.2)$"
        )
    }

    mapped, mapped_config, renamed = remap_minimax_h3_lora(tensors, config)

    gate_prefix = "transformer.blocks.0.mlp.gate_proj"
    up_prefix = "transformer.blocks.0.mlp.up_proj"
    assert renamed == 4
    torch.testing.assert_close(mapped[f"{gate_prefix}.lora_B.default.weight"], gate)
    torch.testing.assert_close(mapped[f"{up_prefix}.lora_B.default.weight"], up)
    torch.testing.assert_close(mapped[f"{gate_prefix}.lora_A.default.weight"], lora_a)
    torch.testing.assert_close(mapped[f"{up_prefix}.lora_A.default.weight"], lora_a)
    assert mapped_config["target_modules"].startswith(r"^blocks\.")
    assert r"(?:mlp\.gate_proj|mlp\.up_proj)" in mapped_config["target_modules"]


def test_h3_lora_remaps_attention_and_fc2_names() -> None:
    tensors = {
        "transformer.transformer_blocks.1.attn.to_out.0.lora_A.default.weight": torch.ones(2, 2),
        "transformer.transformer_blocks.1.ff.net.2.lora_B.default.weight": torch.ones(2, 2),
    }

    mapped, _, renamed = remap_minimax_h3_lora(tensors, {})

    assert renamed == 2
    assert "transformer.blocks.1.attn.out_proj.lora_A.default.weight" in mapped
    assert "transformer.blocks.1.mlp.fc2.lora_B.default.weight" in mapped


def test_h3_fc1_mapping_preserves_swiglu_output() -> None:
    torch.manual_seed(7)
    hidden = torch.randn(5, 3)
    base_up = torch.randn(4, 3)
    base_gate = torch.randn(4, 3)
    lora_a = torch.randn(2, 3)
    lora_up = torch.randn(4, 2)
    lora_gate = torch.randn(4, 2)
    prefix = "transformer.transformer_blocks.0.ff.net.0.proj"
    mapped, _, _ = remap_minimax_h3_lora(
        {
            f"{prefix}.lora_A.default.weight": lora_a,
            f"{prefix}.lora_B.default.weight": torch.cat((lora_up, lora_gate), dim=0),
        },
        {},
    )

    actor_up = hidden @ (base_up + lora_up @ lora_a).T
    actor_gate = hidden @ (base_gate + lora_gate @ lora_a).T
    actor_output = torch.nn.functional.silu(actor_gate) * actor_up

    gate_prefix = "transformer.blocks.0.mlp.gate_proj"
    up_prefix = "transformer.blocks.0.mlp.up_proj"
    rollout_gate = (
        hidden
        @ (
            base_gate + mapped[f"{gate_prefix}.lora_B.default.weight"] @ mapped[f"{gate_prefix}.lora_A.default.weight"]
        ).T
    )
    rollout_up = (
        hidden
        @ (base_up + mapped[f"{up_prefix}.lora_B.default.weight"] @ mapped[f"{up_prefix}.lora_A.default.weight"]).T
    )
    rollout_output = torch.nn.functional.silu(rollout_gate) * rollout_up

    torch.testing.assert_close(rollout_output, actor_output)


def _one_block_diffusers_lora() -> dict[str, torch.Tensor]:
    modules = (
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.to_out.0",
        "ff.net.0.proj",
        "ff.net.2",
    )
    tensors = {}
    for module in modules:
        prefix = f"transformer.transformer_blocks.0.{module}"
        tensors[f"{prefix}.lora_A.default.weight"] = torch.ones(2, 3)
        rows = 8 if module == "ff.net.0.proj" else 4
        tensors[f"{prefix}.lora_B.default.weight"] = torch.ones(rows, 2)
    return tensors


def test_h3_lora_coverage_accepts_exact_logical_slice_set() -> None:
    mapped, _, _ = remap_minimax_h3_lora(_one_block_diffusers_lora(), {})
    validate_minimax_h3_lora_coverage(mapped, block_count=1)


def test_h3_lora_coverage_accepts_vllm_peft_envelope() -> None:
    mapped, _, _ = remap_minimax_h3_lora(_one_block_diffusers_lora(), {})
    enveloped = {f"base_model.model.{name}": tensor for name, tensor in mapped.items()}

    validate_minimax_h3_lora_coverage(enveloped, block_count=1)


def test_h3_lora_coverage_rejects_partial_payload() -> None:
    mapped, _, _ = remap_minimax_h3_lora(_one_block_diffusers_lora(), {})
    mapped.pop("transformer.blocks.0.attn.to_v.lora_A.default.weight")
    mapped.pop("transformer.blocks.0.attn.to_v.lora_B.default.weight")

    with pytest.raises(RuntimeError, match="coverage mismatch"):
        validate_minimax_h3_lora_coverage(mapped, block_count=1)


def test_h3_lora_verify_expectations_follow_rollout_remap() -> None:
    expected_a, expected_b = LoraWeightSyncBase._expected_checksums(
        _one_block_diffusers_lora(),
        {"r": 2, "lora_alpha": 4},
    )
    expected_named = LoraWeightSyncBase._expected_named_checksums(
        _one_block_diffusers_lora(),
        {"r": 2, "lora_alpha": 4},
    )

    assert len(expected_a) == 7
    assert len(expected_b) == 7
    assert len(expected_named) == 7
    assert set(expected_named["transformer.blocks.0.mlp.gate_proj"]) == {"lora_a", "lora_b"}


def test_h3_named_lora_verify_rejects_value_swaps() -> None:
    sync = object.__new__(LoraWeightSyncBase)
    sync._param_prefix = ""
    expected_named = {
        "transformer.blocks.0.attn.to_q": {"lora_a": "a0", "lora_b": "b0"},
        "transformer.blocks.0.attn.to_k": {"lora_a": "a1", "lora_b": "b1"},
    }
    loaded = {
        0: [
            {
                "transformer.blocks.0.attn.to_q": {"lora_a": "a1", "lora_b": "b0"},
                "transformer.blocks.0.attn.to_k": {"lora_a": "a0", "lora_b": "b1"},
            }
        ]
    }

    with pytest.raises(RuntimeError, match="named verify FAILED"):
        sync._assert_loaded(
            ["a0", "a1"],
            ["b0", "b1"],
            loaded,
            topology={0: 1},
            label="test",
            expected_named=expected_named,
        )


def test_h3_active_lora_verify_rejects_buffer_mismatch() -> None:
    sync = object.__new__(LoraWeightSyncBase)
    sync._param_prefix = ""
    expected_named = {
        "transformer.blocks.0.attn.to_q": {"lora_a": "a0", "lora_b": "b0"},
        "transformer.blocks.0.attn.to_k": {"lora_a": "a1", "lora_b": "b1"},
    }
    loaded = {
        0: [
            {
                "transformer.blocks.0.attn.to_q": {
                    "lora_a": "a0",
                    "lora_b": "b0",
                    "active_expected_lora_a": "a0",
                    "active_expected_lora_b": "b0",
                    "active_lora_a": "a1",
                    "active_lora_b": "b0",
                },
                "transformer.blocks.0.attn.to_k": {
                    "lora_a": "a1",
                    "lora_b": "b1",
                    "active_expected_lora_a": "a1",
                    "active_expected_lora_b": "b1",
                    "active_lora_a": "a0",
                    "active_lora_b": "b1",
                },
            }
        ]
    }

    with pytest.raises(RuntimeError, match="active-buffer verify FAILED"):
        sync._assert_loaded(
            ["a0", "a1"],
            ["b0", "b1"],
            loaded,
            topology={0: 1},
            label="test",
            expected_named=expected_named,
            require_active=True,
        )


def test_h3_active_checksum_uses_tp_local_packed_slices() -> None:
    gate = type(
        "Layer",
        (),
        {
            "lora_a": torch.arange(6, dtype=torch.bfloat16).reshape(2, 3),
            "lora_b": torch.arange(8, dtype=torch.bfloat16).reshape(4, 2),
        },
    )()
    up = type(
        "Layer",
        (),
        {
            "lora_a": torch.arange(6, 12, dtype=torch.bfloat16).reshape(2, 3),
            "lora_b": torch.arange(8, 16, dtype=torch.bfloat16).reshape(4, 2),
        },
    )()
    lora_model = type(
        "LoRAModel",
        (),
        {
            "loras": {
                "transformer.blocks.0.mlp.gate_proj": gate,
                "transformer.blocks.0.mlp.up_proj": up,
            },
            "get_lora": lambda self, name: self.loras.get(name),
        },
    )()

    class FakePackedLayer:
        n_slices = 2
        tp_size = 2

        def __init__(self) -> None:
            local_b = [gate.lora_b[:2], up.lora_b[:2]]
            self.lora_a_stacked = tuple(value[None, None].clone() for value in (gate.lora_a, up.lora_a))
            self.lora_b_stacked = tuple(value[None, None].clone() for value in local_b)

        @staticmethod
        def slice_lora_a(values):
            return values

        @staticmethod
        def slice_lora_b(values):
            return [value[:2] for value in values]

    manager = type(
        "Manager",
        (),
        {
            "_registered_adapters": {1: lora_model},
            "_active_adapter_id": 1,
            "_adapter_scales": {1: 1.0},
            "_lora_modules": {"transformer.blocks.0.mlp.fc1": FakePackedLayer()},
            "_get_lora_weights": lambda self, model, name: model.get_lora(name),
            "_get_packed_sublayer_suffixes": (
                lambda self, suffix, count: ["gate_proj", "up_proj"] if (suffix, count) == ("fc1", 2) else None
            ),
        },
    )()
    worker = type("Worker", (), {"lora_manager": manager})()

    checksums = BucketedIPCReceiveMixin._diffrl_loaded_lora_checksums(worker, adapter_id=1)

    for name in ("transformer.blocks.0.mlp.gate_proj", "transformer.blocks.0.mlp.up_proj"):
        fields = checksums[name]
        assert fields["active_lora_a"] == fields["active_expected_lora_a"]
        assert fields["active_lora_b"] == fields["active_expected_lora_b"]
