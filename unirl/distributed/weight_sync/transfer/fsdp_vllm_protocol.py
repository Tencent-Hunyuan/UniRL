"""Protocol helpers for FSDP full-weight publication to direct vLLM TP."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Iterable, Mapping, Sequence

import torch

PROTOCOL = "unirl.fsdp-vllm-full.v1"
REQUIRED_HEADER_FIELDS = frozenset(
    {
        "protocol",
        "sync_id",
        "index",
        "count",
        "is_last",
        "model_version",
        "tp_world_size",
        "fingerprint",
        "tensors",
    }
)
REQUIRED_TENSOR_FIELDS = frozenset({"name", "shape", "dtype", "numel", "sha256"})


def tensor_sha256(tensor: Any) -> str:
    """Hash a tensor's contiguous logical bytes with full SHA-256."""
    value = tensor.detach().contiguous().cpu()
    return hashlib.sha256(value.view(torch.uint8).flatten().numpy().tobytes()).hexdigest()


def tensor_header(name: str, tensor: Any) -> dict[str, Any]:
    """Build the canonical JSON-safe metadata for one full tensor."""
    return {
        "name": str(name),
        "shape": [int(dim) for dim in tensor.shape],
        "dtype": str(tensor.dtype),
        "numel": int(tensor.numel()),
        "nbytes": int(tensor.numel()) * int(tensor.element_size()),
        "sha256": tensor_sha256(tensor),
    }


def bucket_fingerprint(tensors: Sequence[Mapping[str, Any]]) -> str:
    """Hash canonical tensor metadata, including every tensor byte digest."""
    encoded = json.dumps(list(tensors), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_bucket_header(
    *,
    sync_id: str,
    index: int,
    count: int,
    is_last: bool,
    model_version: int,
    tp_world_size: int,
    tensors: Iterable[tuple[str, Any]],
) -> dict[str, Any]:
    """Build one versioned full-weight bucket header."""
    tensor_metadata = [tensor_header(name, tensor) for name, tensor in tensors]
    return {
        "protocol": PROTOCOL,
        "sync_id": str(sync_id),
        "index": int(index),
        "count": int(count),
        "is_last": bool(is_last),
        "model_version": int(model_version),
        "tp_world_size": int(tp_world_size),
        "fingerprint": bucket_fingerprint(tensor_metadata),
        "tensors": tensor_metadata,
    }


def validate_bucket_header(header: Mapping[str, Any], *, payload_count: int, tp_world_size: int) -> list[dict]:
    """Validate structural and TP fanout invariants and return tensor metadata."""
    if not isinstance(header, Mapping):
        raise TypeError(f"FSDP-vLLM bucket header must be a mapping, got {type(header).__name__}")
    missing_fields = sorted(REQUIRED_HEADER_FIELDS - set(header))
    if missing_fields:
        raise ValueError(f"FSDP-vLLM bucket header missing fields: {missing_fields}")
    if header.get("protocol", PROTOCOL) != PROTOCOL:
        raise ValueError(f"unsupported FSDP-vLLM protocol {header.get('protocol')!r}")

    sync_id = header["sync_id"]
    index = header["index"]
    count = header["count"]
    model_version = header["model_version"]
    header_tp_world_size = header["tp_world_size"]
    is_last = header["is_last"]
    fingerprint = header["fingerprint"]
    if not isinstance(sync_id, str) or not sync_id:
        raise ValueError("FSDP-vLLM bucket sync_id must be a non-empty string")
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError(f"FSDP-vLLM bucket index must be >= 0, got {index!r}")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise ValueError(f"FSDP-vLLM bucket count must be > 0, got {count!r}")
    if index >= count:
        raise ValueError(f"FSDP-vLLM bucket index {index} outside count {count}")
    if not isinstance(model_version, int) or isinstance(model_version, bool) or model_version <= 0:
        raise ValueError(f"FSDP-vLLM bucket model_version must be > 0, got {model_version!r}")
    if not isinstance(header_tp_world_size, int) or isinstance(header_tp_world_size, bool) or header_tp_world_size <= 0:
        raise ValueError(f"FSDP-vLLM bucket tp_world_size must be > 0, got {header_tp_world_size!r}")
    if not isinstance(is_last, bool):
        raise ValueError(f"FSDP-vLLM bucket is_last must be bool, got {is_last!r}")
    if is_last != (index == count - 1):
        raise ValueError(f"FSDP-vLLM bucket last marker mismatch: index={index}, count={count}, is_last={is_last}")
    if int(tp_world_size) != header_tp_world_size:
        raise ValueError(f"FSDP-vLLM caller TP world={tp_world_size} != header TP world={header_tp_world_size}")
    if int(payload_count) != header_tp_world_size:
        raise ValueError(
            f"FSDP-vLLM payload fanout mismatch: payloads={payload_count}, TP world={header_tp_world_size}"
        )

    tensors = header["tensors"]
    if not isinstance(tensors, list) or not tensors:
        raise ValueError("FSDP-vLLM bucket tensors must be a non-empty list")
    names: list[str] = []
    normalized: list[dict] = []
    for position, metadata in enumerate(tensors):
        if not isinstance(metadata, Mapping):
            raise TypeError(f"FSDP-vLLM tensor metadata #{position} must be a mapping")
        missing_tensor_fields = sorted(REQUIRED_TENSOR_FIELDS - set(metadata))
        if missing_tensor_fields:
            raise ValueError(f"FSDP-vLLM tensor metadata #{position} missing fields: {missing_tensor_fields}")
        name = metadata["name"]
        shape = metadata["shape"]
        dtype = metadata["dtype"]
        numel = metadata["numel"]
        digest = metadata["sha256"]
        if not isinstance(name, str) or not name:
            raise ValueError(f"FSDP-vLLM tensor metadata #{position} has invalid name {name!r}")
        if not isinstance(shape, (list, tuple)) or any(
            not isinstance(dim, int) or isinstance(dim, bool) or dim < 0 for dim in shape
        ):
            raise ValueError(f"FSDP-vLLM tensor {name!r} has invalid shape {shape!r}")
        expected_numel = 1
        for dim in shape:
            expected_numel *= dim
        if not isinstance(numel, int) or isinstance(numel, bool) or numel != expected_numel:
            raise ValueError(f"FSDP-vLLM tensor {name!r} numel {numel!r} does not match shape product {expected_numel}")
        if not isinstance(dtype, str) or not dtype:
            raise ValueError(f"FSDP-vLLM tensor {name!r} has invalid dtype {dtype!r}")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"FSDP-vLLM tensor {name!r} has invalid SHA-256 {digest!r}")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise ValueError(f"FSDP-vLLM tensor {name!r} has non-hex SHA-256 {digest!r}") from exc
        names.append(name)
        normalized.append(dict(metadata))

    duplicates = sorted(name for name, amount in Counter(names).items() if amount > 1)
    if duplicates:
        raise ValueError(f"FSDP-vLLM bucket header has duplicate tensor names: {duplicates[:8]}")
    expected_fingerprint = bucket_fingerprint(normalized)
    if fingerprint != expected_fingerprint:
        raise ValueError(
            f"FSDP-vLLM bucket fingerprint {fingerprint!r} does not match tensor metadata {expected_fingerprint!r}"
        )
    return normalized


def normalize_receipts(response: Any) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], bool]:
    """Normalize direct-engine update/commit response shapes."""
    committed = False
    commit_receipts: list[Mapping[str, Any]] = []
    if isinstance(response, Mapping):
        receipts = response.get(
            "receipts",
            response.get("update_receipts", response.get("worker_receipts")),
        )
        commit_receipts = list(response.get("commit_receipts") or [])
        committed = bool(response.get("committed", False) or response.get("status") == "committed")
    else:
        receipts = response
    if not isinstance(receipts, Sequence) or isinstance(receipts, (str, bytes)):
        raise RuntimeError(
            f"FSDP-vLLM update must return one receipt per TP worker; got {type(response).__name__}: {response!r}"
        )
    return list(receipts), commit_receipts, committed


def validate_worker_receipts(
    header: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
    *,
    fanout: int,
) -> list[dict[str, Any]]:
    """Validate staged tensor consumption before any worker may commit."""
    if len(receipts) != int(fanout):
        raise RuntimeError(f"FSDP-vLLM receipt count {len(receipts)} != TP fanout {fanout}")

    expected_names = [str(item["name"]) for item in header["tensors"]]
    expected_set = set(expected_names)
    ranks: list[int] = []
    for receipt in receipts:
        if not isinstance(receipt, Mapping):
            raise RuntimeError(f"FSDP-vLLM worker returned invalid receipt {receipt!r}")
        rank = int(receipt.get("rank", -1))
        ranks.append(rank)
        if receipt.get("sync_id") != header["sync_id"] or int(receipt.get("index", -1)) != int(header["index"]):
            raise RuntimeError(f"FSDP-vLLM worker {rank} receipt identifies the wrong bucket: {dict(receipt)!r}")
        if receipt.get("fingerprint") != header["fingerprint"]:
            raise RuntimeError(f"FSDP-vLLM worker {rank} receipt has the wrong bucket fingerprint")
        consumed = [str(name) for name in receipt.get("consumed", ())]
        missing = sorted(str(name) for name in receipt.get("missing", ()))
        unexpected = sorted(str(name) for name in receipt.get("unexpected", ()))
        duplicate = sorted(str(name) for name in receipt.get("duplicate", ()))
        if consumed != expected_names:
            missing = sorted(expected_set - set(consumed))
            unexpected = sorted(set(consumed) - expected_set)
            duplicate = sorted(name for name, amount in Counter(consumed).items() if amount > 1)
        if missing or unexpected or duplicate:
            raise RuntimeError(
                f"FSDP-vLLM worker {rank} rejected bucket names: "
                f"missing={missing[:8]}, unexpected={unexpected[:8]}, duplicate={duplicate[:8]}"
            )
        loaded_missing = sorted(str(name) for name in receipt.get("loaded_missing", ()))
        loaded_unexpected = sorted(str(name) for name in receipt.get("loaded_unexpected", ()))
        if loaded_missing or loaded_unexpected:
            raise RuntimeError(
                f"FSDP-vLLM worker {rank} loaded-model coverage mismatch: "
                f"missing={loaded_missing[:8]}, unexpected={loaded_unexpected[:8]}"
            )
        if int(receipt.get("model_version", -1)) != int(header["model_version"]):
            raise RuntimeError(
                f"FSDP-vLLM worker {rank} receipt model_version {receipt.get('model_version')!r} "
                f"!= header model_version {header['model_version']}"
            )
    if sorted(ranks) != list(range(int(fanout))):
        raise RuntimeError(f"FSDP-vLLM receipt TP ranks {sorted(ranks)} != {list(range(int(fanout)))}")
    return [dict(receipt) for receipt in receipts]


def validate_receipts(header: Mapping[str, Any], response: Any, *, fanout: int) -> dict[str, Any]:
    """Validate consumed-name and version receipts from every direct-vLLM TP worker."""
    receipts, commit_receipts, response_committed = normalize_receipts(response)
    receipts = validate_worker_receipts(header, receipts, fanout=fanout)

    if bool(header["is_last"]):
        if commit_receipts:
            if len(commit_receipts) != int(fanout):
                raise RuntimeError(f"FSDP-vLLM commit receipt count {len(commit_receipts)} != TP fanout {fanout}")
            commit_ranks = sorted(int(receipt.get("rank", -1)) for receipt in commit_receipts)
            committed = all(
                bool(receipt.get("committed"))
                and receipt.get("sync_id") == header["sync_id"]
                and int(receipt.get("model_version", -1)) == int(header["model_version"])
                and int(receipt.get("committed_model_version", -1)) == int(header["model_version"])
                for receipt in commit_receipts
            )
            if commit_ranks != list(range(int(fanout))) or not committed:
                raise RuntimeError(f"FSDP-vLLM invalid commit receipts: {commit_receipts!r}")
        elif not response_committed and not all(bool(receipt.get("committed")) for receipt in receipts):
            raise RuntimeError("FSDP-vLLM last bucket returned before every TP worker committed its version")
    elif response_committed or any(bool(receipt.get("committed")) for receipt in receipts):
        raise RuntimeError("FSDP-vLLM non-last bucket must not commit a model version")

    return {
        "sync_id": str(header["sync_id"]),
        "index": int(header["index"]),
        "count": int(header["count"]),
        "is_last": bool(header["is_last"]),
        "model_version": int(header["model_version"]),
        "tp_world_size": int(header["tp_world_size"]),
        "fingerprint": str(header["fingerprint"]),
        "tensor_count": len(header["tensors"]),
        "byte_count": sum(int(tensor.get("nbytes", 0)) for tensor in header["tensors"]),
        "receipts": [dict(receipt) for receipt in receipts],
        "commit_receipts": [dict(receipt) for receipt in commit_receipts],
        "committed": bool(header["is_last"]),
        "prefix_cache_reset": bool(isinstance(response, Mapping) and response.get("prefix_cache_reset", False)),
    }


__all__ = [
    "PROTOCOL",
    "bucket_fingerprint",
    "build_bucket_header",
    "normalize_receipts",
    "tensor_header",
    "tensor_sha256",
    "validate_bucket_header",
    "validate_receipts",
    "validate_worker_receipts",
]
