"""Validate UniRL publication metadata around vLLM's native IPC engine."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

VLLM_NATIVE_PROTOCOL_VERSION = 1
VLLM_NATIVE_TRANSPORT = "vllm_native_ipc"

_HEADER_FIELDS = frozenset(
    {
        "protocol_version",
        "transport",
        "publication_id",
        "model_version",
        "tp_world_size",
        "expected_manifest",
        "device_uuids",
    }
)
_MANIFEST_FIELDS = frozenset({"fingerprint", "tensor_count", "byte_count"})
_TENSOR_FIELDS = frozenset({"name", "shape", "dtype", "numel", "nbytes"})


def structural_manifest(tensors: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Fingerprint an ordered tensor schema without reading payload bytes."""
    normalized: list[dict[str, Any]] = []
    names: list[str] = []
    byte_count = 0
    for position, raw in enumerate(tensors):
        if not isinstance(raw, Mapping):
            raise TypeError(f"vLLM native IPC tensor metadata #{position} must be a mapping")
        missing = sorted(_TENSOR_FIELDS - set(raw))
        if missing:
            raise ValueError(f"vLLM native IPC tensor metadata #{position} missing fields: {missing}")

        name = raw["name"]
        shape = raw["shape"]
        dtype = raw["dtype"]
        numel = raw["numel"]
        nbytes = raw["nbytes"]
        if not isinstance(name, str) or not name:
            raise ValueError(f"vLLM native IPC tensor metadata #{position} has invalid name")
        if not isinstance(shape, (list, tuple)) or any(
            not isinstance(dim, int) or isinstance(dim, bool) or dim < 0 for dim in shape
        ):
            raise ValueError(f"vLLM native IPC tensor {name!r} has invalid shape {shape!r}")
        expected_numel = 1
        for dim in shape:
            expected_numel *= dim
        if not isinstance(numel, int) or isinstance(numel, bool) or numel != expected_numel:
            raise ValueError(f"vLLM native IPC tensor {name!r} numel does not match shape")
        if not isinstance(dtype, str) or not dtype:
            raise ValueError(f"vLLM native IPC tensor {name!r} has invalid dtype")
        if not isinstance(nbytes, int) or isinstance(nbytes, bool) or nbytes < 0:
            raise ValueError(f"vLLM native IPC tensor {name!r} has invalid nbytes")

        normalized.append(
            {
                "name": name,
                "shape": [int(dim) for dim in shape],
                "dtype": dtype,
                "numel": numel,
                "nbytes": nbytes,
            }
        )
        names.append(name)
        byte_count += nbytes

    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        raise ValueError(f"vLLM native IPC manifest has duplicate names: {duplicates[:8]}")
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()
    return {
        "fingerprint": hashlib.sha256(encoded).hexdigest(),
        "tensor_count": len(normalized),
        "byte_count": byte_count,
    }


def _validate_manifest(manifest: Any) -> dict[str, Any]:
    if not isinstance(manifest, Mapping):
        raise RuntimeError(f"vLLM native IPC manifest must be a mapping, got {type(manifest).__name__}")
    missing = sorted(_MANIFEST_FIELDS - set(manifest))
    if missing:
        raise RuntimeError(f"vLLM native IPC manifest missing fields: {missing}")
    fingerprint = manifest["fingerprint"]
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise RuntimeError("vLLM native IPC manifest fingerprint must be SHA-256")
    try:
        int(fingerprint, 16)
    except ValueError as exc:
        raise RuntimeError("vLLM native IPC manifest fingerprint is not hexadecimal") from exc
    tensor_count = manifest["tensor_count"]
    byte_count = manifest["byte_count"]
    if not isinstance(tensor_count, int) or isinstance(tensor_count, bool) or tensor_count <= 0:
        raise RuntimeError("vLLM native IPC manifest tensor_count must be > 0")
    if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count < 0:
        raise RuntimeError("vLLM native IPC manifest byte_count must be >= 0")
    return {
        "fingerprint": fingerprint,
        "tensor_count": int(tensor_count),
        "byte_count": int(byte_count),
    }


def validate_publication_header(header: Mapping[str, Any], *, tp_world_size: int) -> dict[str, Any]:
    """Validate one fail-stop publication identity."""
    if not isinstance(header, Mapping):
        raise TypeError(f"vLLM native IPC header must be a mapping, got {type(header).__name__}")
    missing = sorted(_HEADER_FIELDS - set(header))
    if missing:
        raise ValueError(f"vLLM native IPC header missing fields: {missing}")
    if header["protocol_version"] != VLLM_NATIVE_PROTOCOL_VERSION or header["transport"] != VLLM_NATIVE_TRANSPORT:
        raise ValueError(
            f"unsupported vLLM native IPC protocol/transport {header['protocol_version']!r}/{header['transport']!r}"
        )
    if not isinstance(header["publication_id"], str) or not header["publication_id"]:
        raise ValueError("vLLM native IPC publication_id must be non-empty")
    model_version = header["model_version"]
    if not isinstance(model_version, int) or isinstance(model_version, bool) or model_version <= 0:
        raise ValueError("vLLM native IPC model_version must be > 0")
    header_world = header["tp_world_size"]
    if not isinstance(header_world, int) or isinstance(header_world, bool) or header_world != int(tp_world_size):
        raise ValueError(f"vLLM native IPC TP world mismatch: header={header_world!r}, actual={tp_world_size}")
    _validate_manifest(header["expected_manifest"])
    uuids = header["device_uuids"]
    if (
        not isinstance(uuids, (list, tuple))
        or len(uuids) != int(tp_world_size)
        or any(not isinstance(value, str) or not value for value in uuids)
        or len(set(uuids)) != int(tp_world_size)
    ):
        raise ValueError(f"vLLM native IPC requires one unique CUDA UUID per TP rank, got {uuids!r}")
    return dict(header)


def validate_worker_receipts(
    header: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    *,
    fanout: int,
) -> list[dict[str, Any]]:
    """Validate post-finish attestations from every vLLM TP worker."""
    validate_publication_header(header, tp_world_size=fanout)
    if len(receipts) != int(fanout):
        raise RuntimeError(f"vLLM native IPC receipt count {len(receipts)} != TP fanout {fanout}")
    ranks: list[int] = []
    normalized: list[dict[str, Any]] = []
    for raw in receipts:
        if not isinstance(raw, Mapping):
            raise RuntimeError(f"vLLM native IPC worker returned invalid receipt {raw!r}")
        rank = int(raw.get("tp_rank", -1))
        if rank not in range(int(fanout)):
            raise RuntimeError(f"vLLM native IPC worker returned invalid TP rank {rank}")
        ranks.append(rank)
        identity = (
            raw.get("publication_id"),
            int(raw.get("model_version", -1)),
            int(raw.get("tp_world_size", -1)),
            raw.get("expected_manifest_fingerprint"),
            raw.get("device_uuid"),
        )
        expected = (
            header["publication_id"],
            int(header["model_version"]),
            int(fanout),
            header["expected_manifest"]["fingerprint"],
            header["device_uuids"][rank],
        )
        if identity != expected or not bool(raw.get("ready")):
            raise RuntimeError(f"vLLM native IPC worker {rank} returned an invalid receipt")
        if int(raw.get("resident_parameter_count", 0)) <= 0:
            raise RuntimeError(f"vLLM native IPC worker {rank} reported no resident parameters")
        normalized.append(dict(raw))
    if sorted(ranks) != list(range(int(fanout))):
        raise RuntimeError(f"vLLM native IPC receipt TP ranks {sorted(ranks)} != {list(range(int(fanout)))}")
    return normalized


def validate_publication_result(
    header: Mapping[str, Any],
    response: Any,
    *,
    fanout: int,
    flush_cache: bool,
) -> dict[str, Any]:
    """Validate finish, version visibility, and prefix-cache invalidation."""
    if not isinstance(response, Mapping) or response.get("status") != "committed":
        raise RuntimeError(f"vLLM native IPC returned invalid result {response!r}")
    receipts = validate_worker_receipts(
        header,
        list(response.get("worker_receipts") or ()),
        fanout=fanout,
    )
    if int(response.get("model_version", -1)) != int(header["model_version"]):
        raise RuntimeError("vLLM native IPC committed the wrong model version")
    prefix_cache_reset = bool(response.get("prefix_cache_reset"))
    if flush_cache and not prefix_cache_reset:
        raise RuntimeError("vLLM native IPC committed without resetting prefix cache")
    return {
        "publication_id": str(header["publication_id"]),
        "model_version": int(header["model_version"]),
        "prefix_cache_reset": prefix_cache_reset,
        "receipts": receipts,
    }


__all__ = [
    "VLLM_NATIVE_PROTOCOL_VERSION",
    "VLLM_NATIVE_TRANSPORT",
    "structural_manifest",
    "validate_publication_header",
    "validate_publication_result",
    "validate_worker_receipts",
]
