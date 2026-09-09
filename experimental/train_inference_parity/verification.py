"""Machine-readable evidence writer for the two-phase TP4 parity run."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .runtime_contract import (
    EXPECTED_TP,
    EXPECTED_VERSIONS,
    PLUGIN_DISTRIBUTION,
    ParityRuntimeContract,
)

SCHEMA_VERSION = "unirl.train_inference_parity.verification.v1"
PHASES = ("initial_checkpoint", "post_update_reload")
CANONICAL_PARITY_METRICS = (
    "token_count",
    "torch_equal_fp32",
    "mismatch_count",
    "max_absdiff_fp32",
    "k3_mean",
    "k3_max",
)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _jsonable(item())
        except (TypeError, ValueError):
            pass
    return repr(value)


def atomic_write_json(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> None:
    """Write one durable JSON value via same-directory fsync + replace."""
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_path = Path(handle.name)
            json.dump(
                _jsonable(payload),
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def _run_git(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(_REPOSITORY_ROOT), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=10,
    )
    return result.stdout.strip()


def _git_state() -> Dict[str, Any]:
    try:
        commit = _run_git("rev-parse", "HEAD")
        # Generated artifacts and unrelated untracked inputs do not make tracked
        # source dirty; every tracked modification still does.
        tracked_status = _run_git("status", "--porcelain", "--untracked-files=no")
        return {
            "commit": commit,
            "dirty": bool(tracked_status),
            "dirty_scope": "tracked_files",
        }
    except (FileNotFoundError, subprocess.SubprocessError) as error:
        return {
            "commit": None,
            "dirty": None,
            "dirty_scope": "tracked_files",
            "error": f"{type(error).__name__}: {error}",
        }


def _package_version(distribution: str) -> Optional[str]:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _software_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: _package_version(name)
            for name in (
                "torch",
                "transformers",
                "vllm",
                "ray",
                PLUGIN_DISTRIBUTION,
            )
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
    driver_query = getattr(torch._C, "_cuda_getDriverVersion", None)
    if callable(driver_query):
        try:
            driver_version = driver_query()
        except RuntimeError:
            pass
    state.update(
        {
            "cuda_runtime": torch.version.cuda,
            "cuda_driver": driver_version,
            "cudnn": torch.backends.cudnn.version(),
            "nccl": nccl_version,
        }
    )
    return state


def _hardware_state() -> Dict[str, Any]:
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
                "compute_capability": [
                    int(properties.major),
                    int(properties.minor),
                ],
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


def _source_revision(path: Path) -> Optional[str]:
    parts = path.resolve().parts
    if "snapshots" in parts:
        index = parts.index("snapshots")
        if index + 1 < len(parts):
            return parts[index + 1]
    return None


def _model_fingerprint(raw_path: str) -> Dict[str, Any]:
    """Fingerprint model identity without rereading every multi-GB weight byte."""
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
        relative = item.relative_to(path).as_posix()
        size = item.stat().st_size
        entry: Dict[str, Any] = {
            "path": relative,
            "size_bytes": size,
            "sha256": _sha256_file(item),
        }
        entries.append(entry)
        digest.update(json.dumps(entry, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return {
        "path": str(path),
        "kind": "file_tree_bytes",
        "algorithm": "sha256(relative_path,size,file_sha256)",
        "sha256": digest.hexdigest(),
        "revision": _source_revision(path),
        "files": entries,
    }


def _data_fingerprint(raw_path: str) -> Dict[str, Any]:
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


def _fingerprint_digest(value: Mapping[str, Any]) -> Optional[str]:
    raw = value.get("sha256", value.get("digest"))
    return str(raw) if raw else None


def _normalize_source_fingerprint(value: Any) -> Dict[str, Any]:
    value = _jsonable(value)
    if isinstance(value, Mapping):
        normalized = dict(value)
        if not _fingerprint_digest(normalized):
            raise ValueError("source fingerprint mapping must contain sha256 or digest")
        return normalized
    if not isinstance(value, list) or not value:
        raise TypeError("source_model_fingerprint() must return a mapping or rank-report list")
    if not all(isinstance(item, Mapping) for item in value):
        raise TypeError("source fingerprint rank reports must all be mappings")
    reports = [dict(item) for item in value]
    digests = {_fingerprint_digest(report) for report in reports}
    if None in digests or len(digests) != 1:
        raise ValueError(f"source fingerprint ranks disagree: {digests!r}")
    normalized = {key: item for key, item in reports[0].items() if key != "rank"}
    normalized["rank_reports"] = reports
    return normalized


def _baseline_from_initial_state(value: Any) -> Dict[str, Any]:
    """Extract one consensus source fingerprint from the initial sync-state RPC."""
    value = _jsonable(value)
    states = value if isinstance(value, list) else [value]
    if not states or not all(isinstance(state, Mapping) for state in states):
        raise TypeError("record_initial_actor_fingerprint() must return mapping states")
    fingerprints = [state.get("source_model_fingerprint") for state in states]
    normalized = [_normalize_source_fingerprint(fingerprint) for fingerprint in fingerprints]
    digests = {_fingerprint_digest(fingerprint) for fingerprint in normalized}
    if None in digests or len(digests) != 1:
        raise ValueError(f"initial actor fingerprint states disagree: {digests!r}")
    return normalized[0]


def _call_required(target: Any, method_name: str) -> Any:
    method = getattr(target, method_name, None)
    if not callable(method):
        raise RuntimeError(f"FSDPVLLMFullWeightSync integration is incomplete: missing callable {method_name}()")
    return method()


def _normalize_receipts(value: Any) -> list[Dict[str, Any]]:
    value = _jsonable(value)
    if isinstance(value, list) and value and all(isinstance(item, list) for item in value):
        canonical = {json.dumps(item, sort_keys=True, separators=(",", ":")) for item in value}
        if len(canonical) != 1:
            raise ValueError("weight-sync ranks disagree on verification receipts")
        value = value[0]
    if isinstance(value, Mapping) and "receipts" in value:
        value = value["receipts"]
    if isinstance(value, Mapping):
        value = [value]
    if not isinstance(value, list):
        raise TypeError(f"verification_receipts() returned {type(value).__name__}, expected list")
    receipts = []
    for item in value:
        if not isinstance(item, Mapping):
            raise TypeError("every weight receipt must be a mapping")
        receipts.append(dict(item))
    return receipts


def _normalize_plugin_manifest(value: Any) -> Dict[str, Any]:
    value = _jsonable(value)
    if isinstance(value, list):
        manifests = [item for item in value if isinstance(item, Mapping)]
        if len(manifests) != 1:
            raise ValueError(f"expected one TP-head plugin manifest, got {len(manifests)}")
        value = manifests[0]
    if not isinstance(value, Mapping):
        raise TypeError(f"vLLM plugin manifest must be a mapping, got {type(value).__name__}")
    manifest = dict(value)
    if (
        manifest.get("profile") != "public_reference"
        or manifest.get("model") != "qwen3_moe_30b_a3b"
        or manifest.get("strict") is not True
    ):
        raise ValueError(f"vLLM plugin manifest contract mismatch: {manifest!r}")
    patches = manifest.get("patches")
    if not isinstance(patches, list) or [patch.get("name") for patch in patches] != [
        "common",
        "qwen3_moe_30b_a3b",
    ]:
        raise ValueError(f"vLLM plugin manifest has the wrong patch set: {patches!r}")
    for patch in patches:
        symbols = patch.get("symbols")
        if not isinstance(symbols, list) or not symbols:
            raise ValueError(f"vLLM plugin patch has no symbol evidence: {patch!r}")
        if not all(isinstance(symbol, Mapping) and symbol.get("verified") is True for symbol in symbols):
            raise ValueError(f"vLLM plugin patch contains unverified symbols: {patch!r}")
    versions = manifest.get("runtime")
    if not isinstance(versions, list):
        raise ValueError("vLLM plugin manifest omitted runtime versions")
    installed_versions = {
        str(item.get("package")): str(item.get("installed")).split("+", 1)[0]
        for item in versions
        if isinstance(item, Mapping)
    }
    if installed_versions != EXPECTED_VERSIONS:
        raise ValueError(f"vLLM worker versions={installed_versions!r}; expected {EXPECTED_VERSIONS!r}")
    return manifest


def _per_rank_token_counts(result: Any) -> list[int]:
    totals: list[int] = []
    for micro in getattr(result, "micros", ()) or ():
        counts = (getattr(micro, "metrics", {}) or {}).get("per_rank_token_count")
        if not isinstance(counts, (list, tuple)):
            continue
        if not totals:
            totals = [0] * len(counts)
        if len(counts) != len(totals):
            raise ValueError("parity micro-batches reported inconsistent rank counts")
        totals = [total + int(count) for total, count in zip(totals, counts, strict=True)]
    return totals


class ParityVerification:
    """Collect evidence and emit PASS only after both required phases."""

    def __init__(self, contract: ParityRuntimeContract) -> None:
        self.contract = contract
        self.artifact_path = Path(contract.artifact_path).expanduser().resolve()
        self._weight_sync: Any = None
        self._baseline_source_fingerprint: Optional[Dict[str, Any]] = None
        self._document: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "INCOMPLETE",
            "git": _git_state(),
            "software": _software_state(),
            "hardware": _hardware_state(),
            "model_fingerprint": _model_fingerprint(contract.model_path),
            "data_fingerprint": _data_fingerprint(contract.data_path),
            "flags": contract.flags(),
            "environment": dict(contract.environment),
            "phases": [],
            "failure_reasons": ["both required phases have not completed"],
        }

    def initialize(self) -> None:
        """Persist launch metadata before Ray or model construction begins."""
        self._write()

    def bind(self, trainer: Any) -> None:
        """Capture the initial actor fingerprint before the first optimizer update."""
        self._weight_sync = getattr(trainer, "weight_sync", None)
        if self._weight_sync is None:
            self._fail("trainer has no dedicated weight-sync object")
        try:
            rollout = getattr(trainer, "rollout", None)
            plugin_manifest_method = getattr(rollout, "parity_runtime_manifest", None)
            if not callable(plugin_manifest_method):
                raise RuntimeError("direct vLLM rollout does not expose parity_runtime_manifest()")
            self._document["vllm_plugin_manifest"] = _normalize_plugin_manifest(plugin_manifest_method())
            initial_state = _call_required(self._weight_sync, "record_initial_actor_fingerprint")
            self._document["initial_actor_fingerprint_state"] = _jsonable(initial_state)
            baseline = _baseline_from_initial_state(initial_state)
            self._baseline_source_fingerprint = baseline
            self._document["initial_actor_fingerprint"] = baseline
            self._write()
        except Exception as error:
            self._fail(f"could not capture initial actor fingerprint: {error}", cause=error)

    def on_rollout_complete(
        self,
        *,
        rollout_id: int,
        result: Any,
        mean_reward: float,
        sync_weights: bool,
    ) -> None:
        """ARTrainer callback: record one canonical parity phase."""
        try:
            if rollout_id < 0 or rollout_id >= len(PHASES):
                raise ValueError(f"unexpected rollout_id={rollout_id}; the contract requires exactly two")
            phase_name = PHASES[rollout_id]
            expected_reload = phase_name == "post_update_reload"
            if bool(sync_weights) != expected_reload:
                raise RuntimeError(f"{phase_name} sync_weights={sync_weights}; expected {expected_reload}")

            metrics = dict(getattr(result, "metrics", {}) or {})
            missing_metrics = [name for name in CANONICAL_PARITY_METRICS if name not in metrics]
            if missing_metrics:
                raise RuntimeError(f"missing canonical parity metrics: {missing_metrics}")
            canonical = {name: _jsonable(metrics[name]) for name in CANONICAL_PARITY_METRICS}
            per_rank_token_count = _per_rank_token_counts(result)
            reported_token_count = int(canonical["token_count"])
            token_count = sum(per_rank_token_count) if per_rank_token_count else reported_token_count
            if per_rank_token_count and reported_token_count != token_count:
                raise RuntimeError(
                    f"aggregated token_count={reported_token_count} does not match per-rank total={token_count}"
                )
            canonical["token_count"] = token_count
            if token_count <= 0:
                raise RuntimeError(f"{phase_name} recorded no replay tokens")

            receipts: list[Dict[str, Any]] = []
            parameter_changed: Optional[bool] = None
            receipt_errors: list[str] = []
            if expected_reload:
                receipts = _normalize_receipts(_call_required(self._weight_sync, "verification_receipts"))
                parameter_changed, receipt_errors = self._validate_receipts(receipts)

            exact_metrics = (
                canonical["torch_equal_fp32"] in (True, 1, 1.0)
                and canonical["mismatch_count"] == 0
                and canonical["max_absdiff_fp32"] == 0.0
                and canonical["k3_mean"] == 0.0
                and canonical["k3_max"] == 0.0
            )
            grad_norm = float(getattr(result, "grad_norm"))
            optimizer_updates = int(getattr(result, "optimizer_updates", 0))
            if phase_name == "initial_checkpoint" and optimizer_updates == 1:
                _call_required(self._weight_sync, "mark_actor_updated")
            phase_errors = list(receipt_errors)
            if not exact_metrics:
                phase_errors.append("canonical parity metrics are not exact")
            if not math.isfinite(grad_norm):
                phase_errors.append(f"grad_norm is not finite: {grad_norm!r}")
            if optimizer_updates != 1:
                phase_errors.append(f"optimizer_updates={optimizer_updates}; expected exactly 1")
            if expected_reload and parameter_changed is not True:
                phase_errors.append("source parameters did not change from the initial actor")

            phase = {
                "phase": phase_name,
                "rollout_id": int(rollout_id),
                "status": "PASS" if not phase_errors else "FAIL",
                "token_count": token_count,
                "per_rank_token_count": per_rank_token_count,
                "canonical_parity_metrics": canonical,
                "grad_norm": grad_norm,
                "optimizer_updates": optimizer_updates,
                "mean_reward": float(mean_reward),
                "parameter_changed": parameter_changed,
                "weight_receipts": receipts,
                "failure_reasons": phase_errors,
            }
            self._document["phases"].append(phase)
            self._refresh_status()
            self._write()
            if phase_errors:
                raise RuntimeError(f"{phase_name} verification failed: {'; '.join(phase_errors)}")
        except Exception as error:
            if not self._document["phases"] or self._document["phases"][-1].get("rollout_id") != rollout_id:
                self._document["status"] = "FAIL"
                self._document["failure_reasons"] = [
                    f"{PHASES[rollout_id] if 0 <= rollout_id < len(PHASES) else rollout_id}: {error}"
                ]
                self._write()
            raise

    def _validate_receipts(self, receipts: list[Dict[str, Any]]) -> tuple[bool, list[str]]:
        errors: list[str] = []
        if not receipts:
            return False, ["post-update reload produced no weight receipt"]
        baseline = _fingerprint_digest(self._baseline_source_fingerprint or {})
        source_digests = set()
        for receipt_index, receipt in enumerate(receipts):
            payload = receipt.get("payload")
            workers = receipt.get("workers")
            source = receipt.get("source_model_fingerprint")
            if not isinstance(payload, Mapping):
                errors.append(f"receipt[{receipt_index}].payload is missing")
                continue
            if int(payload.get("tensor_count", 0)) <= 0:
                errors.append(f"receipt[{receipt_index}] has an empty tensor payload")
            if int(payload.get("byte_count", 0)) <= 0:
                errors.append(f"receipt[{receipt_index}] has zero payload bytes")
            if int(receipt.get("tp_world_size", 0)) != EXPECTED_TP:
                errors.append(
                    f"receipt[{receipt_index}] tp_world_size={receipt.get('tp_world_size')!r}; expected {EXPECTED_TP}"
                )
            if not isinstance(source, Mapping) or not _fingerprint_digest(source):
                errors.append(f"receipt[{receipt_index}].source_model_fingerprint is missing")
            else:
                source_digests.add(_fingerprint_digest(source))
            if not isinstance(workers, list):
                errors.append(f"receipt[{receipt_index}].workers is missing")
                continue
            ranks = sorted(int(worker.get("tp_rank", -1)) for worker in workers if isinstance(worker, Mapping))
            if ranks != list(range(EXPECTED_TP)):
                errors.append(f"receipt[{receipt_index}] TP ranks={ranks}; expected {list(range(EXPECTED_TP))}")
            versions = set()
            for worker in workers:
                if not isinstance(worker, Mapping):
                    errors.append(f"receipt[{receipt_index}] has a non-mapping worker")
                    continue
                for field in (
                    "missing",
                    "unexpected",
                    "duplicate",
                    "loaded_missing",
                    "loaded_unexpected",
                ):
                    if worker.get(field):
                        errors.append(
                            f"receipt[{receipt_index}] tp_rank={worker.get('tp_rank')} {field}={worker.get(field)!r}"
                        )
                consumed = int(worker.get("consumed_tensor_count", 0))
                if consumed != int(payload.get("tensor_count", 0)):
                    errors.append(
                        f"receipt[{receipt_index}] tp_rank={worker.get('tp_rank')} "
                        f"consumed={consumed}, payload tensors={payload.get('tensor_count')}"
                    )
                versions.add(worker.get("committed_model_version"))
            expected_model_version = receipt.get("model_version")
            if versions != {expected_model_version}:
                errors.append(
                    f"receipt[{receipt_index}] worker model versions={versions}; "
                    f"expected publication model_version={expected_model_version!r}"
                )
            if receipt.get("reload_applied") is not True:
                errors.append(f"receipt[{receipt_index}] reload_applied is not true")
            if receipt.get("prefix_cache_reset") is not True:
                errors.append(f"receipt[{receipt_index}] prefix_cache_reset is not true")
        changed = bool(baseline and len(source_digests) == 1 and next(iter(source_digests), None) != baseline)
        if len(source_digests) > 1:
            errors.append(f"receipts disagree on source fingerprint: {source_digests}")
        return changed, errors

    def _refresh_status(self) -> None:
        phases = self._document["phases"]
        complete = len(phases) == len(PHASES) and [phase.get("phase") for phase in phases] == list(PHASES)
        failures = [reason for phase in phases for reason in phase.get("failure_reasons", [])]
        git_state = self._document["git"]
        if git_state.get("dirty") is not False:
            failures.append("git tracked-files state is dirty or unknown")
        if complete and not failures and all(phase.get("status") == "PASS" for phase in phases):
            self._document["status"] = "PASS"
            self._document["failure_reasons"] = []
        else:
            self._document["status"] = "FAIL" if failures else "INCOMPLETE"
            self._document["failure_reasons"] = failures or ["both required phases have not completed"]

    def record_failure(self, stage: str, error: BaseException) -> None:
        """Persist an external launch/runtime failure without masking it."""
        reason = f"{stage}: {type(error).__name__}: {error}"
        existing = list(self._document.get("failure_reasons", ()))
        if reason not in existing:
            existing.append(reason)
        self._document["status"] = "FAIL"
        self._document["failure_reasons"] = existing
        self._write()

    def finalize(self) -> None:
        """Require a complete, clean, two-phase PASS artifact."""
        self._refresh_status()
        self._write()
        if self._document["status"] != "PASS":
            raise RuntimeError(
                f"parity verification did not produce PASS: {self._document.get('failure_reasons', [])!r}"
            )

    def _fail(self, message: str, *, cause: Optional[Exception] = None) -> None:
        self._document["status"] = "FAIL"
        self._document["failure_reasons"] = [message]
        self._write()
        if cause is not None:
            raise RuntimeError(message) from cause
        raise RuntimeError(message)

    def _write(self) -> None:
        atomic_write_json(self.artifact_path, self._document)


__all__ = [
    "CANONICAL_PARITY_METRICS",
    "PHASES",
    "SCHEMA_VERSION",
    "ParityVerification",
    "atomic_write_json",
]
