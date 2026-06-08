#!/usr/bin/env python3
"""Inspect Ray runtime_env venvs for each scorer.

Usage (after service is running):
    python scripts/check_venvs.py

Or standalone (starts a temporary Ray cluster):
    python scripts/check_venvs.py --standalone

Reports each scorer's venv path, Python executable, and installed
scorer-level packages. Useful for verifying per-scorer isolation.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _read_reqs(path: str) -> list[str]:
    """Read a requirements file, skip comments and blanks."""
    lines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                lines.append(line)
    return lines


# Packages to check in each venv
_CHECK_PKGS = [
    "transformers", "vllm", "hpsv2", "image_reward",
    "timm", "peft", "diffusers", "fairscale", "ftfy",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--standalone", action="store_true",
        help="Start a temporary local Ray cluster (default: connect to existing)",
    )
    args = parser.parse_args()

    import ray

    if args.standalone:
        ray.init(ignore_reinit_error=True)
    else:
        try:
            ray.init(address="auto", ignore_reinit_error=True)
        except ConnectionError:
            print("No running Ray cluster found. Use --standalone to start one.")
            sys.exit(1)

    @ray.remote
    def probe(scorer_name: str, check_pkgs: list[str]) -> dict:
        """Run inside the venv: report executable and installed packages."""
        result = {
            "scorer": scorer_name,
            "executable": sys.executable,
            "is_venv": "/runtime_resources/pip/" in sys.executable,
            "packages": {},
        }
        for pkg in check_pkgs:
            try:
                mod = __import__(pkg)
                ver = getattr(mod, "__version__", "unknown")
                result["packages"][pkg] = ver
            except ImportError:
                pass
        return result

    from reward_service.workers.group import _build_runtime_env

    # Discover all envs/*.txt files
    envs_dir = Path("envs")
    if not envs_dir.is_dir():
        print(f"Error: {envs_dir.resolve()} not found. Run from project root.")
        sys.exit(1)

    scorers = []
    for txt in sorted(envs_dir.glob("*.txt")):
        if txt.name == "base.txt":
            continue
        scorer_name = txt.stem
        scorers.append((scorer_name, str(txt)))

    print(f"Found {len(scorers)} scorer envs: {[s[0] for s in scorers]}")
    print()

    refs = []
    for name, envfile in scorers:
        runtime_env = _build_runtime_env(envfile)
        reqs = runtime_env["pip"]["packages"]
        print(f"  Submitting {name}: {reqs}")
        ref = probe.options(runtime_env=runtime_env).remote(name, _CHECK_PKGS)
        refs.append(ref)

    print("\nWaiting for venvs to install (this may take a few minutes on first run)...\n")
    results = ray.get(refs, timeout=600)

    # Report
    print("=" * 70)
    print(f"{'Scorer':<18} {'In Venv':<10} {'Key Packages'}")
    print("=" * 70)
    for r in results:
        pkgs = ", ".join(f"{k}=={v}" for k, v in r["packages"].items())
        print(f"{r['scorer']:<18} {str(r['is_venv']):<10} {pkgs or '(none)'}")
        print(f"  {'':18} exec: {r['executable']}")
    print("=" * 70)

    # Check isolation
    execs = {}
    for r in results:
        exe = r["executable"]
        execs.setdefault(exe, []).append(r["scorer"])
    print(f"\nUnique venvs: {len(execs)}")
    for exe, names in execs.items():
        print(f"  {exe}")
        print(f"    -> {', '.join(names)}")

    if not args.standalone:
        print("\n(Connected to existing Ray cluster; venvs persist across runs)")

    ray.shutdown()


if __name__ == "__main__":
    main()
