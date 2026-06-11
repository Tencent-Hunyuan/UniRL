#!/usr/bin/env python3
"""Probe UniRL compatibility layers on DGX Spark / linux-aarch64.

The probe is intentionally read-only: it imports installed packages, reports CUDA
visibility, and identifies the first missing layer in the aarch64 stack.
"""

from __future__ import annotations

import importlib
import importlib.metadata as md
import json
import platform
import sys
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def version_of(dist: str) -> str | None:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def import_check(module: str, dist: str | None = None) -> Check:
    dist = dist or module.split(".", 1)[0].replace("_", "-")
    try:
        imported = importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001 - diagnostic tool
        return Check(module, False, f"{type(exc).__name__}: {exc}")
    version = version_of(dist) or getattr(imported, "__version__", "unknown")
    return Check(module, True, f"version={version}")


def torch_check() -> list[Check]:
    checks: list[Check] = [import_check("torch")]
    if not checks[-1].ok:
        return checks

    import torch

    checks.append(Check("torch.cuda.is_available", bool(torch.cuda.is_available()), str(torch.cuda.is_available())))
    checks.append(Check("torch.version.cuda", bool(torch.version.cuda), str(torch.version.cuda)))
    if torch.cuda.is_available():
        try:
            checks.append(Check("torch.cuda.device", True, torch.cuda.get_device_name(0)))
            major, minor = torch.cuda.get_device_capability(0)
            checks.append(Check("torch.cuda.capability", True, f"sm_{major}{minor}"))
            x = torch.ones((2, 2), device="cuda")
            checks.append(Check("torch.cuda.tensor_smoke", bool((x + x).sum().item() == 8.0), "2x2 tensor add"))
        except Exception as exc:  # noqa: BLE001 - diagnostic tool
            checks.append(Check("torch.cuda.tensor_smoke", False, f"{type(exc).__name__}: {exc}"))
    return checks


def main() -> int:
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "glibc": platform.libc_ver(),
        "checks": [],
    }

    checks: list[Check] = []
    checks.append(Check("linux-aarch64", sys.platform == "linux" and platform.machine() == "aarch64", f"{sys.platform}/{platform.machine()}"))

    # Layer 1: package metadata and UniRL import surface.
    checks.extend(
        [
            import_check("unirl"),
            import_check("hydra", "hydra-core"),
            import_check("omegaconf"),
            import_check("ray"),
            import_check("diffusers"),
            import_check("transformers"),
            import_check("peft"),
        ]
    )

    # Layer 2: CUDA-bearing torch stack.
    checks.extend(torch_check())

    # Layer 3: risky rollout/native-extension boundary. These may legitimately
    # be missing on DGX Spark until upstream publishes aarch64 wheels.
    checks.extend(
        [
            import_check("sglang"),
            import_check("vllm"),
            import_check("flash_attn", "flash-attn"),
            import_check("sglang_kernel", "sglang-kernel"),
        ]
    )

    report["checks"] = [asdict(check) for check in checks]
    print(json.dumps(report, indent=2, ensure_ascii=False))

    required = ["linux-aarch64", "unirl", "hydra", "omegaconf", "ray", "diffusers", "transformers", "peft", "torch", "torch.cuda.is_available"]
    by_name = {check.name: check for check in checks}
    failed_required = [name for name in required if not by_name.get(name, Check(name, False, "missing")).ok]
    if failed_required:
        print("\nFAILED_REQUIRED=" + ",".join(failed_required), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
