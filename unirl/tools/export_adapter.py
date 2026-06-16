"""Export a UniRL LoRA checkpoint to a PEFT adapter folder.

``FSDPBackend.save(..., mode="adapter")`` writes a UniRL resume checkpoint:
LoRA weights plus optimizer/scheduler/trainer state. This tool extracts just
one adapter and writes the standard PEFT serving artifact:

* ``adapter_model.safetensors``
* ``adapter_config.json``

It also works on ``save_mode=full`` checkpoints by filtering the LoRA keys out
of the full policy state dict.
"""

from __future__ import annotations

import argparse
import os
import pickle
from collections.abc import Iterable
from typing import Dict, List, Optional

import torch
from safetensors.torch import save_file

PEFT_PREFIX = "base_model.model."


def export_adapter_state_dict(
    state_dict: Dict[str, torch.Tensor],
    *,
    adapter: str = "default",
    peft_prefix: str = PEFT_PREFIX,
) -> Dict[str, torch.Tensor]:
    """Convert UniRL PEFT-injected LoRA keys to PEFT adapter-file keys."""
    pairs: Dict[str, Dict[str, torch.Tensor]] = {}
    out: Dict[str, torch.Tensor] = {}
    for suffix in ("lora_A", "lora_B"):
        marker = f".{suffix}.{adapter}.weight"
        for key, value in state_dict.items():
            if not key.endswith(marker):
                continue
            stem = key[: -len(marker)]
            if peft_prefix and not stem.startswith(peft_prefix):
                stem = f"{peft_prefix}{stem}"
            pairs.setdefault(stem, {})[suffix] = value
    if not pairs:
        raise SystemExit(f"no LoRA tensors for adapter {adapter!r} in the checkpoint")
    incomplete = [stem for stem, pair in pairs.items() if "lora_A" not in pair or "lora_B" not in pair]
    if incomplete:
        raise SystemExit(f"incomplete LoRA adapter pairs in checkpoint: {incomplete[:3]}")
    for stem, pair in pairs.items():
        out[f"{stem}.lora_A.weight"] = pair["lora_A"].detach().cpu()
        out[f"{stem}.lora_B.weight"] = pair["lora_B"].detach().cpu()
    return out


def write_adapter_config(
    output: str,
    *,
    base: str,
    r: int,
    lora_alpha: int,
    target_modules: List[str],
    lora_dropout: float = 0.0,
    bias: str = "none",
    task_type: str = "FEATURE_EXTRACTION",
) -> None:
    os.makedirs(output, exist_ok=True)
    try:
        from peft import LoraConfig

        config = LoraConfig(
            r=int(r),
            lora_alpha=int(lora_alpha),
            target_modules=list(target_modules),
            lora_dropout=float(lora_dropout),
            bias=str(bias),
            task_type=str(task_type),
        )
        config.base_model_name_or_path = str(base)
        config.save_pretrained(output)
    except Exception:
        # Keep export usable in minimal/offline tool environments where PEFT is
        # unavailable or its config API changes; the written fields are the PEFT
        # LoRA essentials needed by common loaders.
        import json

        with open(os.path.join(output, "adapter_config.json"), "w") as f:
            json.dump(
                {
                    "alpha_pattern": {},
                    "base_model_name_or_path": str(base),
                    "bias": str(bias),
                    "fan_in_fan_out": False,
                    "inference_mode": True,
                    "init_lora_weights": True,
                    "lora_alpha": int(lora_alpha),
                    "lora_dropout": float(lora_dropout),
                    "modules_to_save": None,
                    "peft_type": "LORA",
                    "r": int(r),
                    "rank_pattern": {},
                    "revision": None,
                    "target_modules": list(target_modules),
                    "task_type": str(task_type),
                    "use_rslora": False,
                },
                f,
                indent=2,
                sort_keys=True,
            )


def _load_checkpoint(path: str) -> Dict[str, object]:
    if os.path.isdir(path):
        path = os.path.join(path, "checkpoint.pt")
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except (TypeError, pickle.UnpicklingError):
        return torch.load(path, map_location="cpu")
    except RuntimeError as exc:
        if "Weights only load failed" not in str(exc):
            raise
        return torch.load(path, map_location="cpu")


def _split_modules(values: Optional[Iterable[str]]) -> Optional[List[str]]:
    if values is None:
        return None
    modules: List[str] = []
    for value in values:
        modules.extend(part.strip() for part in str(value).split(",") if part.strip())
    return modules


def _require_int(value: object, *, name: str) -> int:
    if value in (None, ""):
        raise SystemExit(f"missing LoRA {name}; pass --lora-{name.replace('_', '-')}")
    return int(value)


def _require_modules(value: object) -> List[str]:
    if isinstance(value, str):
        modules = _split_modules([value])
    elif isinstance(value, Iterable):
        modules = [str(item) for item in value]
    else:
        modules = None
    if not modules:
        raise SystemExit("missing LoRA target_modules; pass --target-modules")
    return modules


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="checkpoint-<step> dir, or the checkpoint.pt itself")
    parser.add_argument("--base", required=True, help="HF repo id / local path of the base model")
    parser.add_argument("--output", required=True, help="output folder for the PEFT adapter")
    parser.add_argument("--adapter", default="default", help='adapter name to export, e.g. "default" or "old"')
    parser.add_argument(
        "--peft-prefix",
        default=PEFT_PREFIX,
        help="prefix for PEFT adapter keys; use '' to preserve checkpoint stems",
    )
    parser.add_argument("--lora-r", type=int, default=None, help="override LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=None, help="override LoRA alpha")
    parser.add_argument("--target-modules", nargs="*", default=None, help="override target modules")
    parser.add_argument("--lora-dropout", type=float, default=None, help="override LoRA dropout")
    parser.add_argument("--bias", default=None, help="override LoRA bias setting")
    parser.add_argument("--task-type", default=None, help="override PEFT task_type")
    args = parser.parse_args()

    checkpoint = _load_checkpoint(args.checkpoint)
    state_dict = checkpoint["policy_state_dict"]
    recorded = checkpoint.get("lora_config") or {}

    r = args.lora_r if args.lora_r is not None else recorded.get("rank")
    alpha = args.lora_alpha if args.lora_alpha is not None else recorded.get("alpha")
    target_modules = _split_modules(args.target_modules) or _require_modules(recorded.get("target_modules"))
    dropout = args.lora_dropout if args.lora_dropout is not None else recorded.get("dropout", 0.0)
    bias = args.bias if args.bias is not None else recorded.get("bias", "none")
    task_type = args.task_type if args.task_type is not None else recorded.get("task_type", "FEATURE_EXTRACTION")

    adapter_state = export_adapter_state_dict(
        state_dict,
        adapter=args.adapter,
        peft_prefix=str(args.peft_prefix or ""),
    )

    os.makedirs(args.output, exist_ok=True)
    save_file(adapter_state, os.path.join(args.output, "adapter_model.safetensors"), metadata={"format": "pt"})
    write_adapter_config(
        args.output,
        base=args.base,
        r=_require_int(r, name="r"),
        lora_alpha=_require_int(alpha, name="alpha"),
        target_modules=target_modules,
        lora_dropout=float(dropout),
        bias=str(bias),
        task_type=str(task_type),
    )
    print(f"wrote {len(adapter_state)} LoRA tensors to {args.output}")


if __name__ == "__main__":
    main()
