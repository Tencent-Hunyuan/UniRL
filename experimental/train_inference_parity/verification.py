"""Machine-readable evidence writer for the two-phase TP4 parity run."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .artifacts import (
    atomic_write_json,
    data_fingerprint,
    git_state,
    hardware_state,
    jsonable,
    model_fingerprint,
    software_state,
)
from .runtime_contract import (
    EXPECTED_TP,
    EXPECTED_VERSIONS,
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


def _fingerprint_digest(value: Mapping[str, Any]) -> Optional[str]:
    raw = value.get("sha256", value.get("digest"))
    return str(raw) if raw else None


def _normalize_source_fingerprint(value: Any) -> Dict[str, Any]:
    value = jsonable(value)
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
    value = jsonable(value)
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
        raise RuntimeError(f"parity integration is incomplete: missing callable {method_name}()")
    return method()


def _normalize_receipts(value: Any) -> list[Dict[str, Any]]:
    value = jsonable(value)
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


def _validate_plugin_manifest(value: Any) -> Dict[str, Any]:
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


def _normalize_plugin_manifests(value: Any) -> Dict[str, Any]:
    value = jsonable(value)
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list):
        raise TypeError(f"vLLM runtime capabilities must be a list, got {type(value).__name__}")
    manifests = []
    for tp_rank, capability in enumerate(value):
        if not isinstance(capability, Mapping):
            raise TypeError(f"vLLM TP capability {tp_rank} must be a mapping")
        manifest = capability.get("train_inference_parity")
        if manifest is None:
            raise ValueError(f"vLLM TP capability {tp_rank} omitted the parity plugin manifest")
        manifests.append(
            {
                "tp_rank": int(capability.get("tp_rank", -1)),
                "manifest": _validate_plugin_manifest(manifest),
            }
        )
    ranks = sorted(item["tp_rank"] for item in manifests)
    if ranks != list(range(EXPECTED_TP)):
        raise ValueError(f"vLLM plugin manifest TP ranks={ranks}; expected {list(range(EXPECTED_TP))}")
    return {"tp_world_size": len(manifests), "workers": manifests}


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


def _aggregate_parity_metrics(result: Any) -> tuple[Dict[str, Any], list[int]]:
    micros = [dict(getattr(micro, "metrics", {}) or {}) for micro in getattr(result, "micros", ()) or ()]
    for index, metrics in enumerate(micros):
        missing = [name for name in CANONICAL_PARITY_METRICS if name not in metrics]
        if missing:
            raise RuntimeError(f"micro-batch {index} omitted canonical parity metrics: {missing}")
    if not micros:
        metrics = dict(getattr(result, "metrics", {}) or {})
        missing = [name for name in CANONICAL_PARITY_METRICS if name not in metrics]
        if missing:
            raise RuntimeError(f"missing canonical parity metrics: {missing}")
        return {name: jsonable(metrics[name]) for name in CANONICAL_PARITY_METRICS}, []

    token_count = sum(int(metrics["token_count"]) for metrics in micros)
    k3_sum = sum(float(metrics["k3_mean"]) * int(metrics["token_count"]) for metrics in micros)
    canonical = {
        "token_count": token_count,
        "torch_equal_fp32": all(bool(metrics["torch_equal_fp32"]) for metrics in micros),
        "mismatch_count": sum(int(metrics["mismatch_count"]) for metrics in micros),
        "max_absdiff_fp32": max(float(metrics["max_absdiff_fp32"]) for metrics in micros),
        "k3_mean": k3_sum / token_count if token_count else 0.0,
        "k3_max": max(float(metrics["k3_max"]) for metrics in micros),
    }
    return canonical, _per_rank_token_counts(result)


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
            "git": git_state(),
            "software": software_state(),
            "hardware": hardware_state(),
            "model_fingerprint": model_fingerprint(contract.model_path),
            "data_fingerprint": data_fingerprint(contract.data_path),
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
            capability_method = getattr(rollout, "runtime_capabilities", None)
            if not callable(capability_method):
                raise RuntimeError("direct vLLM rollout does not expose runtime_capabilities()")
            self._document["vllm_plugin_manifest"] = _normalize_plugin_manifests(capability_method())
            initial_state = _call_required(self._weight_sync, "record_initial_actor_fingerprint")
            self._document["initial_actor_fingerprint_state"] = jsonable(initial_state)
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
            if rollout_id != len(self._document["phases"]):
                raise ValueError(f"out-of-order parity phase: rollout_id={rollout_id}")
            phase_name = PHASES[rollout_id]
            expected_reload = phase_name == "post_update_reload"
            if bool(sync_weights) != expected_reload:
                raise RuntimeError(f"{phase_name} sync_weights={sync_weights}; expected {expected_reload}")

            canonical, per_rank_token_count = _aggregate_parity_metrics(result)
            reported_token_count = int(canonical["token_count"])
            token_count = sum(per_rank_token_count) if per_rank_token_count else reported_token_count
            if per_rank_token_count and reported_token_count != token_count:
                raise RuntimeError(
                    f"aggregated token_count={reported_token_count} does not match per-rank total={token_count}"
                )
            canonical["token_count"] = token_count
            expected_tokens = int(self.contract.flags().get("expected_token_count") or 0)
            if token_count <= 0:
                raise RuntimeError(f"{phase_name} recorded no replay tokens")
            if expected_tokens > 0 and token_count != expected_tokens:
                raise RuntimeError(
                    f"{phase_name} token_count={token_count}; expected {expected_tokens} "
                    "(batch_size × samples_per_prompt × max_new_tokens with ignore_eos)"
                )

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
            expected_fingerprint = payload.get("fingerprint")
            if not isinstance(expected_fingerprint, str) or len(expected_fingerprint) != 64:
                errors.append(f"receipt[{receipt_index}] has an invalid payload fingerprint")
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
            device_uuids = set()
            for worker in workers:
                if not isinstance(worker, Mapping):
                    errors.append(f"receipt[{receipt_index}] has a non-mapping worker")
                    continue
                if worker.get("publication_id") != receipt.get("publication_id"):
                    errors.append(f"receipt[{receipt_index}] tp_rank={worker.get('tp_rank')} publication_id mismatch")
                if worker.get("expected_manifest_fingerprint") != expected_fingerprint:
                    errors.append(
                        f"receipt[{receipt_index}] tp_rank={worker.get('tp_rank')} manifest fingerprint mismatch"
                    )
                if worker.get("ready") is not True or int(worker.get("resident_parameter_count", 0)) <= 0:
                    errors.append(f"receipt[{receipt_index}] tp_rank={worker.get('tp_rank')} is not ready")
                versions.add(worker.get("model_version"))
                device_uuid = worker.get("device_uuid")
                if not isinstance(device_uuid, str) or not device_uuid:
                    errors.append(f"receipt[{receipt_index}] tp_rank={worker.get('tp_rank')} has no CUDA UUID")
                else:
                    device_uuids.add(device_uuid)
            expected_model_version = receipt.get("model_version")
            if versions != {expected_model_version}:
                errors.append(
                    f"receipt[{receipt_index}] worker model versions={versions}; "
                    f"expected publication model_version={expected_model_version!r}"
                )
            if len(device_uuids) != EXPECTED_TP:
                errors.append(
                    f"receipt[{receipt_index}] unique worker CUDA UUIDs={len(device_uuids)}; expected {EXPECTED_TP}"
                )
            if receipt.get("reload_applied") is not True:
                errors.append(f"receipt[{receipt_index}] reload_applied is not true")
            if receipt.get("prefix_cache_reset") is not True:
                errors.append(f"receipt[{receipt_index}] prefix_cache_reset is not true")
            if receipt.get("parameter_changed") is not True:
                errors.append(f"receipt[{receipt_index}] parameter_changed is not true")
        changed = bool(
            baseline
            and len(source_digests) == 1
            and next(iter(source_digests), None) != baseline
            and all(receipt.get("parameter_changed") is True for receipt in receipts)
        )
        if len(source_digests) > 1:
            errors.append(f"receipts disagree on source fingerprint: {source_digests}")
        return changed, errors

    def _refresh_status(self) -> None:
        phases = self._document["phases"]
        complete = len(phases) == len(PHASES) and [phase.get("phase") for phase in phases] == list(PHASES)
        failures = list(self._document.get("runtime_failures", ()))
        failures.extend(reason for phase in phases for reason in phase.get("failure_reasons", []))
        git_state = self._document["git"]
        if git_state.get("dirty") is not False:
            failures.append("git source tree is dirty or unknown")
        if complete and not failures and all(phase.get("status") == "PASS" for phase in phases):
            self._document["status"] = "PASS"
            self._document["failure_reasons"] = []
        else:
            self._document["status"] = "FAIL" if failures else "INCOMPLETE"
            self._document["failure_reasons"] = failures or ["both required phases have not completed"]

    def record_failure(self, stage: str, error: BaseException) -> None:
        """Persist an external launch/runtime failure without masking it."""
        reason = f"{stage}: {type(error).__name__}: {error}"
        self._document.setdefault("runtime_failures", []).append(reason)
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
