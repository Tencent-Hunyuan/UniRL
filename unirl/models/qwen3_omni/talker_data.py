"""Offline Talker codec-data provenance helpers.

Fingerprints are deliberately cheap enough to compute on every training worker:
configuration files are content-hashed and local weight shards are identified by
their relative name and byte size.  The resulting digest catches model/revision,
codec-layout, and accidental checkpoint-directory mismatches without reading a
multi-hundred-GB checkpoint into memory.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .talker_contract import AUDIO_SAMPLE_RATE, NUM_CODE_GROUPS

FINGERPRINT_SCHEMA = "unirl.talker-fingerprint.v1"
CODEC_DATA_SCHEMA = "unirl.talker-codec-data.v1"

_METADATA_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_identity(path: Path) -> Dict[str, Any]:
    files: Dict[str, Any] = {}
    for name in _METADATA_FILES:
        candidate = path / name
        if candidate.is_file():
            files[name] = {"sha256": _sha256_file(candidate), "size": candidate.stat().st_size}
    shards = sorted(path.glob("*.safetensors"))
    if shards:
        files["weight_shards"] = [{"name": shard.name, "size": shard.stat().st_size} for shard in shards]
    if not files:
        raise FileNotFoundError(
            f"Cannot fingerprint {str(path)!r}: no model/config metadata or safetensors shards found."
        )
    return {"type": "local", "files": files}


def model_fingerprint(
    reference: str,
    *,
    kind: str,
    revision: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a stable model/checkpoint fingerprint dictionary."""
    reference = str(reference).strip()
    if not reference:
        raise ValueError("model_fingerprint requires a non-empty model reference")
    path = Path(reference).expanduser()
    if path.exists():
        identity = _local_identity(path.resolve() if path.is_dir() else path.resolve().parent)
    else:
        identity = {"type": "hub", "repository": reference, "revision": revision or "default"}
    payload = {
        "schema": FINGERPRINT_SCHEMA,
        "kind": str(kind),
        "identity": identity,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return {
        "schema": FINGERPRINT_SCHEMA,
        "kind": str(kind),
        "source": reference,
        "revision": revision,
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def codec_fingerprint(
    reference: str,
    *,
    revision: Optional[str] = None,
    sample_rate: int = AUDIO_SAMPLE_RATE,
    channels: int = 1,
    num_quantizers: int = NUM_CODE_GROUPS,
) -> Dict[str, Any]:
    """Fingerprint the Mimi weights together with the exact encoded layout."""
    model = model_fingerprint(reference, kind="mimi_codec_model", revision=revision)
    payload = {
        "schema": FINGERPRINT_SCHEMA,
        "kind": "mimi_codec",
        "model_sha256": model["sha256"],
        "sample_rate": int(sample_rate),
        "channels": int(channels),
        "num_quantizers": int(num_quantizers),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return {
        **payload,
        "source": reference,
        "revision": revision,
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def fingerprint_sha(value: Any, *, field_name: str) -> str:
    """Extract and validate a fingerprint digest from a dict or raw SHA string."""
    if isinstance(value, Mapping):
        if value.get("schema") != FINGERPRINT_SCHEMA:
            raise ValueError(f"{field_name} has schema {value.get('schema')!r}; expected {FINGERPRINT_SCHEMA!r}.")
        value = value.get("sha256")
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"{field_name} must contain a 64-character SHA-256 digest.")
    return digest


def assert_fingerprint(
    actual: Any,
    expected: Any,
    *,
    field_name: str,
    sample_id: str,
) -> None:
    actual_sha = fingerprint_sha(actual, field_name=field_name)
    expected_sha = fingerprint_sha(expected, field_name=f"expected_{field_name}")
    if actual_sha != expected_sha:
        raise ValueError(
            f"Talker SFT record {sample_id!r} {field_name} mismatch: "
            f"row={actual_sha}, expected={expected_sha}. Re-encode offline with the configured model/codec."
        )


__all__ = [
    "CODEC_DATA_SCHEMA",
    "FINGERPRINT_SCHEMA",
    "assert_fingerprint",
    "codec_fingerprint",
    "fingerprint_sha",
    "model_fingerprint",
]
