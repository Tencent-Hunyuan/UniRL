"""vLLM general-plugin entry point for UniRL train/inference parity."""

from __future__ import annotations

import os

_RUNTIME_MANIFEST: dict[str, object] | None = None


def register() -> tuple[str, ...]:
    global _RUNTIME_MANIFEST
    if os.environ.get("UNIRL_PARITY_ENABLE") != "1":
        _RUNTIME_MANIFEST = None
        return ()

    import json

    from .compat import validate_runtime_versions
    from .config import load_config
    from .registry import Installer, install_selected

    config = load_config()
    if config.model != "qwen3_moe_30b_a3b":
        raise ValueError(f"unsupported parity model {config.model!r}")
    versions = validate_runtime_versions()
    from .common import install_common, preflight_common
    from .models.qwen3_moe_30b_a3b import (
        install_qwen3_moe_patch,
        preflight_qwen3_moe_patch,
    )

    installers = {
        "common": Installer(preflight=preflight_common, install=install_common),
        "qwen3_moe_30b_a3b": Installer(
            preflight=preflight_qwen3_moe_patch,
            install=install_qwen3_moe_patch,
        ),
    }
    installed = install_selected(
        config.patches,
        strict=config.strict,
        installers=installers,
    )
    manifest = {
        "entrypoint": "unirl_train_inference_parity",
        "model": config.model,
        "patches": [result.to_manifest() for result in installed],
        "pid": os.getpid(),
        "profile": config.profile,
        "runtime": [result.to_manifest() for result in versions],
        "source_file": __file__,
        "strict": config.strict,
    }
    _RUNTIME_MANIFEST = manifest
    print(
        "[unirl.parity.vllm] manifest=" + json.dumps(manifest, sort_keys=True, separators=(",", ":")),
        flush=True,
    )
    return tuple(result.name for result in installed)


def runtime_manifest() -> dict[str, object] | None:
    return None if _RUNTIME_MANIFEST is None else dict(_RUNTIME_MANIFEST)


__all__ = ["register", "runtime_manifest"]
