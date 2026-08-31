"""MiniMax-H3 trainer-to-serving LoRA layout contract."""

from __future__ import annotations

import torch


def remap_minimax_h3_lora(
    lora_tensors: dict[str, object],
    peft_config: dict,
) -> tuple[dict[str, object], dict, int]:
    """Map Diffusers H3 LoRAs into the serving fused block layout."""
    remapped: dict[str, object] = {}
    renamed = 0
    for name, tensor in lora_tensors.items():
        mapped = name.replace("transformer.transformer_blocks.", "transformer.blocks.")
        mapped = mapped.replace(".attn.to_out.0.", ".attn.out_proj.")
        mapped = mapped.replace(".ff.net.2.", ".mlp.fc2.")
        if ".ff.net.0.proj." in mapped:
            gate_name = mapped.replace(".ff.net.0.proj.", ".mlp.gate_proj.")
            up_name = mapped.replace(".ff.net.0.proj.", ".mlp.up_proj.")
            if ".lora_B." in mapped:
                if not torch.is_tensor(tensor) or tensor.shape[0] % 2:
                    raise ValueError(
                        f"MiniMax-H3 fused FFN LoRA-B must split evenly, got "
                        f"{getattr(tensor, 'shape', None)} for {name}"
                    )
                # Diffusers SwiGLU stores B rows as [up, gate], while the serving fused
                # fc1 chunks its output as [gate, up].
                up, gate = tensor.chunk(2, dim=0)
                remapped[gate_name] = gate.contiguous()
                remapped[up_name] = up.contiguous()
            else:
                remapped[gate_name] = tensor.clone() if torch.is_tensor(tensor) else tensor
                remapped[up_name] = tensor.clone() if torch.is_tensor(tensor) else tensor
            renamed += 2
            continue
        renamed += int(mapped != name)
        remapped[mapped] = tensor

    mapped_config = dict(peft_config or {})
    target_modules = mapped_config.get("target_modules")
    if isinstance(target_modules, str):
        # DiffusionLoRAManager matches this pattern against the component-qualified
        # module name ("transformer.blocks.0..."), so the trainside anchor keeps its
        # "transformer." segment rather than being stripped to "^blocks".
        target_modules = target_modules.replace("^transformer_blocks", r"^transformer\.blocks")
        target_modules = target_modules.replace(r"attn\.to_out\.0", r"attn\.out_proj")
        target_modules = target_modules.replace(
            r"ff\.net\.0\.proj",
            r"(?:mlp\.gate_proj|mlp\.up_proj)",
        )
        target_modules = target_modules.replace(r"ff\.net\.2", r"mlp\.fc2")
        mapped_config["target_modules"] = target_modules
    return remapped, mapped_config, renamed


def validate_minimax_h3_lora_coverage(lora_tensors: dict[str, object], *, block_count: int) -> None:
    """Require every expected H3 block/slice to carry both LoRA A and B."""
    suffixes = (
        "attn.to_q",
        "attn.to_k",
        "attn.to_v",
        "attn.out_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.fc2",
    )
    expected = {
        f"transformer.blocks.{block_index}.{suffix}" for block_index in range(int(block_count)) for suffix in suffixes
    }
    actual: dict[str, set[str]] = {}
    for name, tensor in lora_tensors.items():
        field = None
        module = None
        for candidate in ("lora_A", "lora_B"):
            marker = f".{candidate}."
            if marker in name:
                module = name.split(marker, 1)[0]
                module = module.removeprefix("base_model.model.")
                field = candidate
                break
        if module is None:
            continue
        if not torch.is_tensor(tensor):
            raise TypeError(f"MiniMax-H3 LoRA tensor {name!r} is not a tensor")
        actual.setdefault(module, set()).add(field)

    actual_names = set(actual)
    missing = sorted(expected - actual_names)
    extra = sorted(actual_names - expected)
    incomplete = sorted(name for name, fields in actual.items() if fields != {"lora_A", "lora_B"})
    if missing or extra or incomplete:
        raise RuntimeError(
            "MiniMax-H3 LoRA payload coverage mismatch: "
            f"expected_modules={len(expected)} actual_modules={len(actual_names)} "
            f"missing={missing[:8]} extra={extra[:8]} incomplete={incomplete[:8]}"
        )


__all__ = ["remap_minimax_h3_lora", "validate_minimax_h3_lora_coverage"]
