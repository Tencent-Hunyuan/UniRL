#!/usr/bin/env python
"""Export a UniRL training checkpoint to a Hugging Face ``save_pretrained`` folder.

``checkpoint.pt`` (see ``FSDPBackend.save``) is a raw trainer pickle: the
trainable module's state dict with PEFT-injected names
(``*.base_layer.weight`` / ``*.lora_A.<adapter>.weight``) plus optimizer and
scheduler state. This script folds the LoRA delta into the base weights,
restores the upstream parameter names, strict-loads them into the base model
class, and writes a standard HF folder — ready for ``from_pretrained`` or
``hf upload``.

The fold mirrors :func:`unirl.utils.peft_merge.merged_state_dict` (fp32 merge,
same key grammar) but runs offline on the checkpoint dict, so the LoRA scaling
cannot be read off a live ``peft_config`` — pass ``--lora-alpha`` from the
recipe (``backend.lora_cfg.alpha``; scaling = alpha / rank, rank is inferred
from the weights).

Examples:
    # SD3.5 LoRA run (alpha from the recipe), diffusers transformer subfolder
    python -m unirl.utils.export_hf \\
        --checkpoint /ckpts/sd3_trainside/checkpoint-500 \\
        --base stabilityai/stable-diffusion-3.5-medium --subfolder transformer \\
        --lora-alpha 64 --output /ckpts/sd3_trainside/hf-500

    # AR (transformers CausalLM)
    python -m unirl.utils.export_hf \\
        --checkpoint /ckpts/qwen3/checkpoint-300 --library transformers \\
        --base Qwen/Qwen3-4B-Base --lora-alpha 64 --output /ckpts/qwen3/hf-300

Loading the SD3 result back into a pipeline:
    transformer = AutoModel.from_pretrained("/ckpts/sd3_trainside/hf-500", torch_dtype=torch.bfloat16)
    pipe = StableDiffusion3Pipeline.from_pretrained(base, transformer=transformer)
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, Optional

import torch

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def merge_lora_state_dict(
    state_dict: Dict[str, torch.Tensor],
    *,
    adapter: str = "default",
    alpha: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    """Fold ``adapter``'s LoRA delta into the base weights; restore upstream names.

    No-op (copy) for checkpoints without LoRA keys (full-finetune recipes).
    Other adapters' keys (e.g. the NFT shadow ``old``) are dropped.
    """
    if not any(".lora_A." in k for k in state_dict):
        return dict(state_dict)
    if alpha is None:
        raise SystemExit(
            "checkpoint contains LoRA adapters — pass --lora-alpha (backend.lora_cfg.alpha in the training recipe)"
        )

    out: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if ".lora_A." in key or ".lora_B." in key:
            continue  # folded below, or a non-exported adapter
        if ".base_layer." not in key:
            out[key] = value
            continue
        original = key.replace(".base_layer.", ".")
        if key.endswith(".base_layer.weight"):
            stem = key[: -len(".base_layer.weight")]
            lora_a = state_dict.get(f"{stem}.lora_A.{adapter}.weight")
            lora_b = state_dict.get(f"{stem}.lora_B.{adapter}.weight")
            if lora_a is not None and lora_b is not None:
                scaling = float(alpha) / lora_a.shape[0]
                # Merge in fp32: a bf16 base + bf16 delta rounds the update away.
                value = (value.float() + (lora_b.float() @ lora_a.float()) * scaling).to(value.dtype)
        out[original] = value
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="checkpoint-<step> dir, or the checkpoint.pt itself")
    parser.add_argument("--base", required=True, help="HF repo id / local snapshot of the BASE model")
    parser.add_argument("--output", required=True, help="output folder for save_pretrained")
    parser.add_argument("--subfolder", default=None, help='base subfolder, e.g. "transformer" for diffusers pipelines')
    parser.add_argument("--library", choices=("diffusers", "transformers"), default="diffusers")
    parser.add_argument("--adapter", default="default", help='LoRA adapter to fold ("old" = the NFT EMA shadow)')
    parser.add_argument("--lora-alpha", type=float, default=None, help="backend.lora_cfg.alpha from the recipe")
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="bf16")
    args = parser.parse_args()

    path = args.checkpoint
    if os.path.isdir(path):
        path = os.path.join(path, "checkpoint.pt")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = checkpoint["policy_state_dict"]
    print(f"loaded {path}: {len(state_dict)} tensors, step={checkpoint.get('step')}")

    merged = merge_lora_state_dict(state_dict, adapter=args.adapter, alpha=args.lora_alpha)

    dtype = DTYPES[args.dtype]
    from_pretrained_kwargs = {"torch_dtype": dtype}
    if args.subfolder:
        from_pretrained_kwargs["subfolder"] = args.subfolder
    if args.library == "diffusers":
        from diffusers import AutoModel

        model = AutoModel.from_pretrained(args.base, **from_pretrained_kwargs)
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(args.base, **from_pretrained_kwargs)

    # strict: naming drift between checkpoint and base class is a hard error,
    # not a silently half-loaded export.
    model.load_state_dict({k: v.to(dtype) if v.is_floating_point() else v for k, v in merged.items()}, strict=True)
    model.save_pretrained(args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
