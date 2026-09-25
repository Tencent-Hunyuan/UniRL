"""Atomic JSON output and source/runtime provenance for parity verification."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

from .runtime_contract import EXPECTED_VERSIONS, PLUGIN_DISTRIBUTION

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_SUFFIXES = {".py", ".yaml", ".yml", ".toml", ".json"}


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return jsonable(item())
        except (TypeError, ValueError):
            pass
    return repr(value)


def atomic_write_json(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> None:
    """Persist strict JSON with same-directory fsync and atomic replacement."""
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(jsonable(payload), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _run_git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(_REPOSITORY_ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout


def git_state() -> dict[str, Any]:
    """Include tracked edits and individual untracked source files, including new directories."""
    try:
        commit = _run_git("rev-parse", "HEAD").strip()
        records = iter(_run_git("status", "--porcelain=v1", "-z", "--untracked-files=all").split("\0"))
        dirty = []
        for record in records:
            if not record:
                continue
            status, path = record[:2], record[3:]
            if "R" in status or "C" in status:
                next(records)  # Porcelain -z puts the original rename/copy path second.
            if status == "??":
                parts = Path(path).parts
                if parts[0] == "outputs" or any(part.endswith(".egg-info") for part in parts):
                    continue
                if Path(path).suffix not in _SOURCE_SUFFIXES:
                    continue
            dirty.append(record)
        return {
            "commit": commit,
            "dirty": bool(dirty),
            "dirty_scope": "tracked_and_untracked_source",
            "dirty_paths": dirty,
        }
    except (OSError, subprocess.SubprocessError) as error:
        return {
            "commit": None,
            "dirty": None,
            "dirty_scope": "tracked_and_untracked_source",
            "error": f"{type(error).__name__}: {error}",
        }


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def software_state() -> dict[str, Any]:
    state = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: _package_version(name) for name in ("torch", "transformers", "vllm", "ray", PLUGIN_DISTRIBUTION)
        },
        "expected_version_lines": dict(EXPECTED_VERSIONS),
    }
    try:
        import torch
    except ImportError:
        return state
    nccl_version = None
    try:
        nccl_version = torch.cuda.nccl.version()
    except (AttributeError, RuntimeError):
        pass
    driver_version = None
    try:
        driver_version = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        pass
    state.update(
        cuda_runtime=torch.version.cuda,
        cuda_driver=driver_version,
        cudnn=torch.backends.cudnn.version(),
        nccl=nccl_version,
    )
    return state


def hardware_state() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"cuda_available": False, "cuda_device_count": 0, "gpus": []}
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    gpus = []
    for index in range(count):
        properties = torch.cuda.get_device_properties(index)
        gpus.append(
            {
                "index": index,
                "name": properties.name,
                "total_memory_bytes": int(properties.total_memory),
                "compute_capability": [int(properties.major), int(properties.minor)],
                "multi_processor_count": int(properties.multi_processor_count),
            }
        )
    return {
        "hostname": platform.node(),
        "machine": platform.machine(),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": count,
        "gpus": gpus,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def model_fingerprint(raw_path: str) -> dict[str, Any]:
    """Hash every checkpoint file byte, including model weights."""
    path = Path(raw_path).expanduser().resolve()
    if path.is_file():
        return {
            "path": str(path),
            "kind": "file_bytes",
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    if not path.is_dir():
        return {"path": str(path), "kind": "missing", "sha256": None}
    entries = []
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        entry = {
            "path": item.relative_to(path).as_posix(),
            "size_bytes": item.stat().st_size,
            "sha256": _sha256_file(item),
        }
        entries.append(entry)
        digest.update(json.dumps(entry, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    parts = path.parts
    revision_index = parts.index("snapshots") + 1 if "snapshots" in parts else len(parts)
    return {
        "path": str(path),
        "kind": "file_tree_bytes",
        "algorithm": "sha256(relative_path,size,file_sha256)",
        "sha256": digest.hexdigest(),
        "revision": parts[revision_index] if revision_index < len(parts) else None,
        "files": entries,
    }


def data_fingerprint(raw_path: str) -> dict[str, Any]:
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        return {"path": str(path), "kind": "missing", "sha256": None}
    with path.open("rb") as handle:
        rows = sum(1 for line in handle if line.strip())
    return {
        "path": str(path),
        "kind": "file_bytes",
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
        "nonempty_rows": rows,
    }
