"""Synthetic bit-exact check for the shared Qwen3-MoE expert layout."""

from __future__ import annotations

import torch

from unirl.train.backend.veomni.ep.models.qwen3_moe import (
    build_local_fused_block,
    iter_hf_expert_tensors,
)


def main() -> None:
    prefix = "model.layers.0.mlp"
    num_experts, local_experts, intermediate, hidden = 4, 2, 3, 2
    source = {}
    for expert in range(num_experts):
        base = expert * 100
        source[f"{prefix}.experts.{expert}.gate_proj.weight"] = (
            torch.arange(intermediate * hidden).reshape(intermediate, hidden) + base
        )
        source[f"{prefix}.experts.{expert}.up_proj.weight"] = (
            torch.arange(intermediate * hidden).reshape(intermediate, hidden) + base + 20
        )
        source[f"{prefix}.experts.{expert}.down_proj.weight"] = (
            torch.arange(hidden * intermediate).reshape(hidden, intermediate) + base + 40
        )

    gate_up_blocks = []
    down_blocks = []
    for ep_rank in range(num_experts // local_experts):
        gate_up_blocks.append(
            build_local_fused_block(
                fused_param_name=f"{prefix}.experts.gate_up_proj",
                expected_shape=(local_experts, 2 * intermediate, hidden),
                ep_rank=ep_rank,
                available_keys=set(source),
                get_tensor=source.__getitem__,
            )
        )
        down_blocks.append(
            build_local_fused_block(
                fused_param_name=f"{prefix}.experts.down_proj",
                expected_shape=(local_experts, hidden, intermediate),
                ep_rank=ep_rank,
                available_keys=set(source),
                get_tensor=source.__getitem__,
            )
        )

    recovered = dict(
        iter_hf_expert_tensors(
            f"{prefix}.experts.gate_up_proj",
            torch.cat(gate_up_blocks),
        )
    )
    recovered.update(
        iter_hf_expert_tensors(
            f"{prefix}.experts.down_proj",
            torch.cat(down_blocks),
        )
    )
    assert source.keys() == recovered.keys()
    for key in source:
        torch.testing.assert_close(recovered[key], source[key], rtol=0, atol=0)

    partial = dict(source)
    partial.pop(f"{prefix}.experts.1.up_proj.weight")
    try:
        build_local_fused_block(
            fused_param_name=f"{prefix}.experts.gate_up_proj",
            expected_shape=(local_experts, 2 * intermediate, hidden),
            ep_rank=0,
            available_keys=set(partial),
            get_tensor=partial.__getitem__,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("partial per-expert checkpoint did not fail closed")

    print(f"Qwen3-MoE layout round-trip PASS ({len(recovered)}/{len(source)} tensors)", flush=True)


if __name__ == "__main__":
    main()
