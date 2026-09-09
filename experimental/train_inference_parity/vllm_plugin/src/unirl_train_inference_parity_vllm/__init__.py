"""vLLM general-plugin entry point for UniRL train/inference parity."""

from __future__ import annotations

import json
import os

from .config import load_config
from .registry import install_selected, installer


@installer("common")
def _install_common() -> None:
    from .common import install_common

    install_common()


@installer("qwen3_moe_30b_a3b")
def _install_qwen3_moe() -> None:
    from .models.qwen3_moe_30b_a3b import install_qwen3_moe_patch

    install_qwen3_moe_patch()


def register() -> tuple[str, ...]:
    config = load_config()
    if config.model != "qwen3_moe_30b_a3b":
        raise ValueError(f"unsupported parity model {config.model!r}")
    installed = install_selected(config.patches, strict=config.strict)
    manifest = {
        "entrypoint": "unirl_train_inference_parity",
        "model": config.model,
        "patches": installed,
        "pid": os.getpid(),
        "profile": config.profile,
    }
    print(
        "[unirl.parity.vllm] manifest=" + json.dumps(manifest, sort_keys=True, separators=(",", ":")),
        flush=True,
    )
    return installed


__all__ = ["register"]
