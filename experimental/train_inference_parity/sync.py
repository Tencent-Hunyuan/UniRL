"""Parity evidence adapter over UniRL's production vLLM native IPC sync."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.distributed.weight_sync.full.ipc import IPCWeightSync


class ParityIPCWeightSync(IPCWeightSync):
    """Record actor-change and native publication evidence without owning transport."""

    def __init__(
        self,
        *args,
        fingerprint_param_samples: int = 8,
        fingerprint_element_samples: int = 16,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._fingerprint_param_samples = int(fingerprint_param_samples)
        self._fingerprint_element_samples = int(fingerprint_element_samples)
        if self._fingerprint_param_samples <= 0 or self._fingerprint_element_samples <= 0:
            raise ValueError("parity fingerprint sample counts must be positive")
        self._actor_baseline: Optional[dict[str, str]] = None
        self._actor_baseline_source: Optional[dict[str, Any]] = None
        self._actor_candidate: Optional[dict[str, str]] = None
        self._actor_candidate_source: Optional[dict[str, Any]] = None
        self._actor_update_pending = False
        self._publication_history: list[dict[str, Any]] = []

    def _capture_local_shard_fingerprints(self, phase: str) -> dict[str, str]:
        fingerprints: dict[str, str] = {}
        error: Optional[BaseException] = None
        try:
            import torch
            from torch.distributed.tensor import DTensor

            candidates = [
                (name, parameter)
                for name, parameter in self._backend.model.named_parameters()
                if parameter.requires_grad and not parameter.is_meta
            ]
            candidates.sort(key=lambda item: (int(item[1].numel()), item[0]))
            for name, parameter in candidates[: self._fingerprint_param_samples]:
                local = parameter.detach()
                if isinstance(local, DTensor):
                    local = local.to_local()
                flat = local.reshape(-1)
                if flat.numel() == 0:
                    continue
                count = min(int(flat.numel()), self._fingerprint_element_samples)
                indices = (
                    [0] if count == 1 else [index * (int(flat.numel()) - 1) // (count - 1) for index in range(count)]
                )
                sample = flat.index_select(
                    0,
                    torch.tensor(indices, dtype=torch.long, device=flat.device),
                ).contiguous()
                digest = hashlib.sha256()
                digest.update(name.encode())
                digest.update(str(tuple(parameter.shape)).encode())
                digest.update(str(parameter.dtype).encode())
                digest.update(sample.cpu().view(torch.uint8).numpy().tobytes())
                fingerprints[name] = digest.hexdigest()
            if not fingerprints:
                raise RuntimeError("no trainable local actor shards were available for fingerprinting")
        except BaseException as exc:
            error = exc
        self._consensus(error, phase)
        return fingerprints

    def _aggregate_source_fingerprint(self, fingerprints: dict[str, str]) -> dict[str, Any]:
        reports = self._all_gather({"rank": self._global_rank, "parameters": dict(fingerprints)})
        reports.sort(key=lambda report: int(report["rank"]))
        canonical = json.dumps(reports, sort_keys=True, separators=(",", ":")).encode()
        return {
            "algorithm": "sha256(sampled-local-fsdp-shards-v1)",
            "sha256": hashlib.sha256(canonical).hexdigest(),
            "world_size": len(reports),
            "parameter_samples_per_rank": self._fingerprint_param_samples,
            "element_samples_per_parameter": self._fingerprint_element_samples,
            "rank_reports": reports,
        }

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def record_initial_actor_fingerprint(self) -> dict[str, Any]:
        fingerprints = self._capture_local_shard_fingerprints("initial-actor-fingerprint")
        self._actor_baseline = fingerprints
        self._actor_baseline_source = self._aggregate_source_fingerprint(fingerprints)
        self._actor_candidate = None
        self._actor_candidate_source = None
        self._actor_update_pending = False
        return {
            "baseline_recorded": True,
            "rank": self._global_rank,
            "sampled_parameters": sorted(fingerprints),
            "source_model_fingerprint": dict(self._actor_baseline_source),
        }

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def mark_actor_updated(self) -> None:
        if self._actor_baseline is None:
            raise RuntimeError("actor fingerprint baseline was not recorded before training")
        self._actor_update_pending = True

    def _capture_pending_actor_update(self) -> None:
        if not self._actor_update_pending:
            return
        if self._actor_baseline is None:
            raise RuntimeError("actor fingerprint baseline is missing")
        candidate = self._capture_local_shard_fingerprints("post-update-actor-fingerprint")
        local_changed = sorted(name for name, digest in candidate.items() if self._actor_baseline.get(name) != digest)
        changed_reports = self._all_gather({"rank": self._global_rank, "changed_parameters": local_changed})
        if not any(report["changed_parameters"] for report in changed_reports):
            raise RuntimeError("no sampled actor local shard changed after the optimizer update")
        self._actor_candidate = candidate
        self._actor_candidate_source = self._aggregate_source_fingerprint(candidate)

    def _sync_with_vllm_native_engine(self) -> None:
        self._capture_pending_actor_update()
        super()._sync_with_vllm_native_engine()

    def _on_native_publication_committed(
        self,
        *,
        header: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        source = self._actor_candidate_source or self._actor_baseline_source
        if source is None:
            raise RuntimeError("parity publication has no actor source fingerprint")
        changed = bool(
            self._actor_baseline_source and source.get("sha256") != self._actor_baseline_source.get("sha256")
        )
        publication = {
            "publication_id": str(header["publication_id"]),
            "model_version": int(header["model_version"]),
            "tp_world_size": int(header["tp_world_size"]),
            "payload": dict(header["expected_manifest"]),
            "workers": [dict(item) for item in result["worker_receipts"]],
            "prefix_cache_reset": bool(result["prefix_cache_reset"]),
            "source_model_fingerprint": dict(source),
            "parameter_changed": changed,
            "reload_applied": True,
        }
        self._publication_history.append(publication)
        if self._actor_update_pending:
            self._actor_baseline = self._actor_candidate
            self._actor_baseline_source = self._actor_candidate_source
            self._actor_candidate = None
            self._actor_candidate_source = None
            self._actor_update_pending = False

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def verification_receipts(self) -> list[dict[str, Any]]:
        return [dict(publication) for publication in self._publication_history]


__all__ = ["ParityIPCWeightSync"]
