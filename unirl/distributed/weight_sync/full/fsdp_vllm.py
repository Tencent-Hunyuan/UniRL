"""FSDP actor to direct-vLLM-TP full-weight synchronization."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.distributed.weight_sync.full.base import FullWeightSync
from unirl.distributed.weight_sync.transfer.fsdp_vllm_protocol import (
    PROTOCOL,
    bucket_fingerprint,
    tensor_header,
    tensor_sha256,
    validate_bucket_header,
    validate_receipts,
)

logger = logging.getLogger(__name__)


class FSDPVLLMFullWeightSync(FullWeightSync):
    """Disk-stage one full FSDP materialization for a direct vLLM TP world."""

    def __init__(
        self,
        *,
        backend: Any,
        rollout: Any,
        staging_dir: str,
        bucket_size_mb: int = 512,
        flush_cache: bool = True,
        lora_merged: bool = False,
        adapter_name: Optional[str] = None,
        name_remap: Optional[Dict[str, Optional[str]]] = None,
        track_prefix: str = "",
        wire_dtype: Any = None,
        receipt_dir: Optional[str] = None,
        host_memory_budget_mb: Optional[int] = None,
        staging_disk_budget_mb: Optional[int] = None,
        staging_disk_overhead_ratio: float = 1.05,
        fingerprint_param_samples: int = 16,
        fingerprint_element_samples: int = 256,
    ) -> None:
        super().__init__(
            backend=backend,
            bucket_size_mb=bucket_size_mb,
            flush_cache=flush_cache,
            lora_merged=lora_merged,
            adapter_name=adapter_name,
            name_remap=name_remap,
            track_prefix=track_prefix,
            wire_dtype=wire_dtype,
        )
        if not str(staging_dir or "").strip():
            raise ValueError("FSDPVLLMFullWeightSync requires a non-empty staging_dir")
        if host_memory_budget_mb is not None and int(host_memory_budget_mb) <= 0:
            raise ValueError("host_memory_budget_mb must be > 0 when configured")
        if staging_disk_budget_mb is not None and int(staging_disk_budget_mb) <= 0:
            raise ValueError("staging_disk_budget_mb must be > 0 when configured")
        if float(staging_disk_overhead_ratio) < 1.0:
            raise ValueError("staging_disk_overhead_ratio must be >= 1.0")
        if int(fingerprint_param_samples) <= 0 or int(fingerprint_element_samples) <= 0:
            raise ValueError("fingerprint sample counts must be > 0")

        self._rollout = rollout
        self._staging_root = Path(staging_dir).expanduser().resolve()
        self._receipt_dir = None if receipt_dir is None else Path(receipt_dir).expanduser().resolve()
        self._host_memory_budget = None if host_memory_budget_mb is None else int(host_memory_budget_mb) << 20
        self._staging_disk_budget = None if staging_disk_budget_mb is None else int(staging_disk_budget_mb) << 20
        self._staging_disk_overhead_ratio = float(staging_disk_overhead_ratio)
        self._fingerprint_param_samples = int(fingerprint_param_samples)
        self._fingerprint_element_samples = int(fingerprint_element_samples)

        self._staged_buckets: list[dict[str, Any]] = []
        self._staged_count = 0
        self._active_stage_dir: Optional[Path] = None
        self._active_sync_id: Optional[str] = None
        self._next_model_version = 1
        self._last_publication: Optional[dict[str, Any]] = None
        self._publication_history: list[dict[str, Any]] = []
        self._preflight_state: Optional[dict[str, Any]] = None

        self._actor_baseline: Optional[dict[str, str]] = None
        self._actor_candidate: Optional[dict[str, str]] = None
        self._actor_baseline_source: Optional[dict[str, Any]] = None
        self._actor_candidate_source: Optional[dict[str, Any]] = None
        self._actor_update_pending = False
        self._verification: dict[str, Any] = {
            "enabled": True,
            "baseline_recorded": False,
            "pending_update": False,
            "checks": 0,
            "verified": False,
            "rank": self._global_rank,
            "sampled_parameters": [],
            "local_changed": [],
            "global_changed": False,
            "last_sync_id": None,
            "last_model_version": None,
            "source_model_fingerprint": None,
        }

    @property
    def _global_rank(self) -> int:
        if self._dist_ready():
            import torch.distributed as dist

            return int(dist.get_rank())
        return int(self.rank_info.rank) if self.rank_info is not None else 0

    @staticmethod
    def _dist_ready() -> bool:
        try:
            import torch.distributed as dist

            return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        except Exception:
            return False

    def _broadcast_from_rank_zero(self, value: Any) -> Any:
        if not self._dist_ready():
            return value
        import torch.distributed as dist

        values = [value if self._global_rank == 0 else None]
        dist.broadcast_object_list(values, src=0)
        return values[0]

    def _coordinated_raise(self, operation: str, local_error: Optional[str]) -> None:
        reports = [{"rank": self._global_rank, "error": local_error}]
        if self._dist_ready():
            import torch.distributed as dist

            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, reports[0])
            reports = [report for report in gathered if report is not None]
        failures = [report for report in reports if report.get("error")]
        if failures:
            first = failures[0]
            raise RuntimeError(f"FSDPVLLMFullWeightSync {operation} failed on rank {first['rank']}: {first['error']}")

    @staticmethod
    def _format_bytes(amount: int) -> str:
        return f"{amount / (1024**3):.2f} GiB"

    def _wire_nbytes(self, tensor: Any) -> int:
        element_size = tensor.element_size()
        if self._wire_dtype is not None and tensor.is_floating_point():
            element_size = self._wire_dtype.itemsize
        return int(tensor.numel()) * int(element_size)

    def _estimate_staging_bytes(self) -> tuple[int, int]:
        """Estimate staging from parameter metadata without state-dict collectives."""
        model = self._backend.model
        tied_lm_head = bool(getattr(getattr(model, "config", None), "tie_word_embeddings", False))
        total = 0
        largest = 0
        tensors = list(model.named_parameters()) + list(model.named_buffers())
        for raw_name, tensor in tensors:
            name = str(raw_name).removeprefix("base_model.model.")
            if tied_lm_head and name == "lm_head.weight":
                continue
            if not self._lora_merged and (".lora_A." in name or ".lora_B." in name):
                continue
            size = self._wire_nbytes(tensor)
            total += size
            largest = max(largest, size)
        estimated_disk = int(total * self._staging_disk_overhead_ratio)
        largest_bucket = max(int(self._bucket_bytes), largest)
        return estimated_disk, largest_bucket

    @staticmethod
    def _available_host_memory() -> int:
        try:
            import psutil

            return int(psutil.virtual_memory().available)
        except Exception:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
            return page_size * available_pages

    def _run_preflight(self, *, tp_world_size: int) -> dict[str, Any]:
        """Check rank-0 host single-bucket memory and total staging-disk budgets."""
        state: Optional[dict[str, Any]] = None
        local_error: Optional[str] = None
        estimated_disk, materialized_bucket = self._estimate_staging_bytes()
        # Push keeps the loaded CPU bucket, one flattened source, and one
        # independent shared-memory tensor per rollout TP worker alive.
        single_bucket = materialized_bucket * (2 + int(tp_world_size))
        if self._global_rank == 0:
            try:
                self._staging_root.mkdir(parents=True, exist_ok=True)
                host_available = self._available_host_memory()
                disk_available = int(shutil.disk_usage(self._staging_root).free)
                effective_host = min(
                    host_available,
                    self._host_memory_budget if self._host_memory_budget is not None else host_available,
                )
                effective_disk = min(
                    disk_available,
                    self._staging_disk_budget if self._staging_disk_budget is not None else disk_available,
                )
                state = {
                    "staging_dir": str(self._staging_root),
                    "estimated_disk_bytes": estimated_disk,
                    "single_bucket_host_bytes": single_bucket,
                    "materialized_bucket_bytes": materialized_bucket,
                    "payload_fanout": int(tp_world_size),
                    "host_available_bytes": host_available,
                    "disk_available_bytes": disk_available,
                    "host_budget_bytes": self._host_memory_budget,
                    "disk_budget_bytes": self._staging_disk_budget,
                    "effective_host_bytes": effective_host,
                    "effective_disk_bytes": effective_disk,
                }
                failures = []
                if single_bucket > effective_host:
                    failures.append(
                        "single-bucket host demand "
                        f"{self._format_bytes(single_bucket)} exceeds effective available/budget "
                        f"{self._format_bytes(effective_host)}"
                    )
                if estimated_disk > effective_disk:
                    failures.append(
                        f"total staging estimate {self._format_bytes(estimated_disk)} exceeds "
                        f"effective free/budget {self._format_bytes(effective_disk)}"
                    )
                if failures:
                    raise RuntimeError("; ".join(failures))
            except BaseException as exc:
                local_error = f"{type(exc).__name__}: {exc}"
        self._coordinated_raise("preflight", local_error)
        state = self._broadcast_from_rank_zero(state)
        self._preflight_state = dict(state)
        return dict(state)

    def _cleanup_stage_files(self) -> None:
        stage_dir = self._active_stage_dir
        self._staged_buckets.clear()
        self._staged_count = 0
        self._active_stage_dir = None
        self._active_sync_id = None
        if self._global_rank == 0 and stage_dir is not None:
            shutil.rmtree(stage_dir, ignore_errors=True)

    def _aggregate_source_fingerprint(self, fingerprints: Mapping[str, str]) -> dict[str, Any]:
        reports: list[dict[str, Any]] = [{"rank": self._global_rank, "parameters": dict(fingerprints)}]
        if self._dist_ready():
            import torch.distributed as dist

            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, reports[0])
            reports = [report for report in gathered if report is not None]
        reports.sort(key=lambda report: int(report["rank"]))
        canonical = json.dumps(reports, sort_keys=True, separators=(",", ":")).encode()
        return {
            "algorithm": "sha256(sampled-local-fsdp-shards-v1)",
            "sha256": hashlib.sha256(canonical).hexdigest(),
            "world_size": len(reports),
            "parameter_samples_per_rank": self._fingerprint_param_samples,
            "element_samples_per_parameter": self._fingerprint_element_samples,
        }

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def record_initial_actor_fingerprint(self) -> dict[str, Any]:
        """Record sampled local actor shards immediately before first training."""
        fingerprints = self._capture_local_shard_fingerprints("initial fingerprint")
        self._actor_baseline = fingerprints
        self._actor_baseline_source = self._aggregate_source_fingerprint(fingerprints)
        self._actor_candidate = None
        self._actor_candidate_source = None
        self._actor_update_pending = False
        self._verification.update(
            {
                "baseline_recorded": True,
                "pending_update": False,
                "verified": False,
                "rank": self._global_rank,
                "sampled_parameters": sorted(fingerprints),
                "local_changed": [],
                "global_changed": False,
                "source_model_fingerprint": dict(self._actor_baseline_source),
            }
        )
        return dict(self._verification)

    @distributed(dispatch_mode=Dispatch.DP_SCATTER)
    def source_model_fingerprint(self) -> dict[str, Any]:
        """Return this rank's consensus digest over sampled actor local shards."""
        fingerprints = self._capture_local_shard_fingerprints("source model fingerprint")
        source = self._aggregate_source_fingerprint(fingerprints)
        return {**source, "rank": self._global_rank}

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def mark_actor_updated(self) -> None:
        """Mark that the next publication must prove a sampled actor change."""
        if self._actor_baseline is None:
            raise RuntimeError("actor fingerprint baseline was not recorded before training")
        self._actor_update_pending = True
        self._verification["pending_update"] = True

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def get_verification_state(self) -> dict[str, Any]:
        """Expose local-shard and last-publication verification state."""
        return {
            **self._verification,
            "preflight": dict(self._preflight_state) if self._preflight_state is not None else None,
            "publication": dict(self._last_publication) if self._last_publication is not None else None,
        }

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def verification_receipts(self) -> list[dict[str, Any]]:
        """Expose successful bucket/TP receipts for parity artifacts."""
        return [dict(publication) for publication in self._publication_history]

    def _capture_local_shard_fingerprints(self, operation: str) -> dict[str, str]:
        local_error: Optional[str] = None
        fingerprints: dict[str, str] = {}
        try:
            import torch
            from torch.distributed.tensor import DTensor

            candidates = [
                (name, parameter)
                for name, parameter in self._backend.model.named_parameters()
                if parameter.requires_grad and not parameter.is_meta
            ]
            # Prefer the smallest trainable tensors (typically norms/router
            # parameters) because they participate for every token; randomly
            # choosing inactive expert shards can falsely report no update.
            candidates.sort(key=lambda item: (int(item[1].numel()), item[0]))
            candidates = candidates[: self._fingerprint_param_samples]
            for name, parameter in candidates:
                local = parameter.detach()
                if isinstance(local, DTensor):
                    local = local.to_local()
                flat = local.reshape(-1)
                if flat.numel() == 0:
                    continue
                sample_count = min(int(flat.numel()), self._fingerprint_element_samples)
                if sample_count == 1:
                    indices = [0]
                else:
                    indices = [index * (int(flat.numel()) - 1) // (sample_count - 1) for index in range(sample_count)]
                index_tensor = torch.tensor(indices, dtype=torch.long, device=flat.device)
                sample = flat.index_select(0, index_tensor).contiguous().cpu()
                digest = hashlib.sha256()
                digest.update(name.encode())
                digest.update(str(tuple(parameter.shape)).encode())
                digest.update(str(parameter.dtype).encode())
                digest.update(sample.view(torch.uint8).numpy().tobytes())
                fingerprints[name] = digest.hexdigest()
            if not fingerprints:
                raise RuntimeError("no trainable local actor shards were available for fingerprinting")
        except BaseException as exc:
            local_error = f"{type(exc).__name__}: {exc}"
        self._coordinated_raise(operation, local_error)
        return fingerprints

    def _verify_pending_actor_update(self) -> None:
        if not self._actor_update_pending:
            return
        if self._actor_baseline is None:
            self._coordinated_raise("post-update fingerprint", "baseline is missing")
            return

        current = self._capture_local_shard_fingerprints("post-update fingerprint")
        local_changed = sorted(name for name, digest in current.items() if self._actor_baseline.get(name) != digest)
        global_changed = bool(local_changed)
        if self._dist_ready():
            import torch
            import torch.distributed as dist

            # FSDP may expose a composite/undefined process-group backend even
            # though its parameter collectives are NCCL-only. Match the loaded
            # actor device instead of inferring tensor placement from the label.
            device = (
                torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
            )
            changed = torch.tensor(int(global_changed), dtype=torch.int32, device=device)
            dist.all_reduce(changed, op=dist.ReduceOp.MAX)
            global_changed = bool(changed.item())
        self._coordinated_raise(
            "post-update fingerprint assertion",
            None if global_changed else "no sampled actor local shard changed after optimizer update",
        )
        self._actor_candidate = current
        self._actor_candidate_source = self._aggregate_source_fingerprint(current)
        self._verification.update(
            {
                "checks": int(self._verification["checks"]) + 1,
                "verified": True,
                "local_changed": local_changed,
                "global_changed": global_changed,
                "source_model_fingerprint": dict(self._actor_candidate_source),
            }
        )

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def extract(self) -> None:
        """Materialize on every FSDP rank; stage CPU buckets on global rank 0 only."""
        import torch

        if self._active_sync_id is not None or self._staged_count:
            self._coordinated_raise("extract", "called with unconsumed staged buckets")
        self._verify_pending_actor_update()

        target_tp_world_size: Optional[int] = None
        local_error: Optional[str] = None
        if self._global_rank == 0:
            try:
                receiver = getattr(self._rollout, "tensor_weight_sync_target", self._rollout)
                target_tp_world_size = int(getattr(receiver, "weight_payload_fanout", 0))
                if target_tp_world_size <= 0:
                    raise RuntimeError(f"direct vLLM reported invalid TP payload fanout {target_tp_world_size}")
            except BaseException as exc:
                local_error = f"{type(exc).__name__}: {exc}"
        self._coordinated_raise("target TP preflight", local_error)
        target_tp_world_size = int(self._broadcast_from_rank_zero(target_tp_world_size))
        self._run_preflight(tp_world_size=target_tp_world_size)

        sync_id = self._broadcast_from_rank_zero(uuid.uuid4().hex if self._global_rank == 0 else None)
        stage_dir = self._staging_root / f"sync-{sync_id}"
        local_error = None
        if self._global_rank == 0:
            try:
                stage_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
            except BaseException as exc:
                local_error = f"{type(exc).__name__}: {exc}"
        self._coordinated_raise("staging directory creation", local_error)

        self._active_sync_id = str(sync_id)
        self._active_stage_dir = stage_dir
        staged: list[dict[str, Any]] = []
        try:
            for materialization_index, (bucket, _base_is_last) in enumerate(self._iter_buckets()):
                local_error = None
                if self._global_rank == 0:
                    cpu_bucket = None
                    by_dtype = None
                    dtype_bucket = None
                    tensor = None
                    try:
                        by_dtype = {}
                        for name, tensor in bucket:
                            wire_dtype = self._wire_dtype or tensor.dtype
                            by_dtype.setdefault(wire_dtype, []).append((str(name), tensor))
                        for wire_dtype, dtype_bucket in by_dtype.items():
                            staged_index = len(staged)
                            cpu_bucket = [
                                (
                                    name,
                                    tensor.detach().to(
                                        device="cpu",
                                        dtype=wire_dtype,
                                        copy=True,
                                    ),
                                )
                                for name, tensor in dtype_bucket
                            ]
                            data_path = stage_dir / f"bucket-{staged_index:06d}.pt"
                            temporary_path = data_path.with_suffix(".pt.tmp")
                            torch.save(cpu_bucket, temporary_path)
                            os.replace(temporary_path, data_path)
                            staged.append(
                                {
                                    "data_path": data_path,
                                    "header_path": stage_dir / f"bucket-{staged_index:06d}.json",
                                    "tensor_metadata": [tensor_header(name, tensor) for name, tensor in cpu_bucket],
                                }
                            )
                            cpu_bucket.clear()
                            cpu_bucket = None
                    except BaseException as exc:
                        local_error = f"{type(exc).__name__}: {exc}"
                    finally:
                        if cpu_bucket is not None:
                            cpu_bucket.clear()
                        if by_dtype is not None:
                            for pending in by_dtype.values():
                                pending.clear()
                            by_dtype.clear()
                        cpu_bucket = by_dtype = dtype_bucket = tensor = None
                del bucket
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self._coordinated_raise(f"materialization bucket {materialization_index} staging", local_error)

            local_error = None
            summary: Optional[dict[str, Any]] = None
            if self._global_rank == 0:
                try:
                    count = len(staged)
                    if count == 0:
                        raise RuntimeError("actor full-state walk produced no tensors")
                    for index, item in enumerate(staged):
                        tensor_metadata = item.pop("tensor_metadata")
                        header = {
                            "protocol": PROTOCOL,
                            "sync_id": str(sync_id),
                            "index": index,
                            "count": count,
                            "is_last": index == count - 1,
                            "model_version": int(self._next_model_version),
                            "tp_world_size": target_tp_world_size,
                            "fingerprint": bucket_fingerprint(tensor_metadata),
                            "tensors": tensor_metadata,
                        }
                        validate_bucket_header(
                            header,
                            payload_count=target_tp_world_size,
                            tp_world_size=target_tp_world_size,
                        )
                        temporary_header = item["header_path"].with_suffix(".json.tmp")
                        with temporary_header.open("w", encoding="utf-8") as stream:
                            json.dump(header, stream, sort_keys=True, separators=(",", ":"))
                            stream.flush()
                            os.fsync(stream.fileno())
                        os.replace(temporary_header, item["header_path"])
                        item["header"] = header
                    summary = {
                        "sync_id": str(sync_id),
                        "count": count,
                        "model_version": int(self._next_model_version),
                        "tp_world_size": target_tp_world_size,
                    }
                except BaseException as exc:
                    local_error = f"{type(exc).__name__}: {exc}"
            self._coordinated_raise("manifest finalization", local_error)
            summary = self._broadcast_from_rank_zero(summary)
            self._staged_count = int(summary["count"])
            self._staged_buckets = staged if self._global_rank == 0 else []
        except BaseException:
            self._cleanup_stage_files()
            raise

    def _load_staged_bucket(self, item: Mapping[str, Any]) -> list[tuple[str, Any]]:
        import torch

        named_tensors = torch.load(item["data_path"], map_location="cpu", weights_only=False)
        header_tensors = item["header"]["tensors"]
        if not isinstance(named_tensors, list) or len(named_tensors) != len(header_tensors):
            raise RuntimeError("staged bucket tensor count does not match its header")
        for position, ((name, tensor), metadata) in enumerate(zip(named_tensors, header_tensors, strict=True)):
            if (
                str(name) != metadata["name"]
                or [int(dim) for dim in tensor.shape] != list(metadata["shape"])
                or str(tensor.dtype) != metadata["dtype"]
                or int(tensor.numel()) != int(metadata["numel"])
                or tensor_sha256(tensor) != metadata["sha256"]
            ):
                raise RuntimeError(f"staged bucket tensor #{position} failed header/hash validation")
        return named_tensors

    def _write_receipt_artifact(self, publication: Mapping[str, Any]) -> None:
        if self._receipt_dir is None or self._global_rank != 0:
            return
        self._receipt_dir.mkdir(parents=True, exist_ok=True)
        target = self._receipt_dir / f"{publication['sync_id']}.json"
        temporary = target.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(publication, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)

    def _build_verification_publication(
        self,
        bucket_publications: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not bucket_publications:
            raise RuntimeError("cannot build a weight publication without bucket receipts")
        workers: dict[int, dict[str, Any]] = {}
        for bucket in bucket_publications:
            for receipt in bucket["receipts"]:
                tp_rank = int(receipt["rank"])
                worker = workers.setdefault(
                    tp_rank,
                    {
                        "tp_rank": tp_rank,
                        "consumed_tensor_count": 0,
                        "loaded": [],
                        "missing": [],
                        "unexpected": [],
                        "duplicate": [],
                        "loaded_missing": [],
                        "loaded_unexpected": [],
                        "committed_model_version": None,
                    },
                )
                worker["consumed_tensor_count"] += len(receipt.get("consumed", ()))
                worker["loaded"].extend(str(name) for name in receipt.get("loaded", ()))
                for field in (
                    "missing",
                    "unexpected",
                    "duplicate",
                    "loaded_missing",
                    "loaded_unexpected",
                ):
                    worker[field].extend(str(name) for name in receipt.get(field, ()))
            for receipt in bucket.get("commit_receipts", ()):
                tp_rank = int(receipt["rank"])
                if tp_rank not in workers:
                    raise RuntimeError(f"commit receipt references unknown TP rank {tp_rank}")
                workers[tp_rank]["committed_model_version"] = int(receipt["committed_model_version"])
        for worker in workers.values():
            worker["loaded"] = sorted(set(worker["loaded"]))
            for field in (
                "missing",
                "unexpected",
                "duplicate",
                "loaded_missing",
                "loaded_unexpected",
            ):
                worker[field] = sorted(set(worker[field]))

        return {
            "protocol": PROTOCOL,
            "sync_id": self._active_sync_id,
            "model_version": int(self._next_model_version),
            "tp_world_size": int(bucket_publications[-1]["tp_world_size"]),
            "payload": {
                "tensor_count": sum(int(bucket["tensor_count"]) for bucket in bucket_publications),
                "byte_count": sum(int(bucket["byte_count"]) for bucket in bucket_publications),
                "bucket_count": len(bucket_publications),
            },
            "workers": [workers[rank] for rank in sorted(workers)],
            "source_model_fingerprint": (
                dict(self._actor_candidate_source)
                if self._actor_candidate_source is not None
                else dict(self._actor_baseline_source or {})
            ),
            "reload_applied": bool(bucket_publications[-1].get("committed")),
            "prefix_cache_reset": bool(bucket_publications[-1].get("prefix_cache_reset")),
            "buckets": bucket_publications,
        }

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def push(self) -> None:
        """Read, publish, and release exactly one staged bucket file at a time."""
        import torch

        if self._active_sync_id is None or self._staged_count <= 0:
            self._coordinated_raise("push", "requires extract() first")
        success = False
        last_publication: Optional[dict[str, Any]] = None
        bucket_publications: list[dict[str, Any]] = []
        try:
            for index in range(self._staged_count):
                local_error: Optional[str] = None
                if self._global_rank == 0:
                    named_tensors = None
                    flat = None
                    payloads = None
                    payload_tensors = None
                    try:
                        from unirl.distributed.weight_sync.transfer.sgl_compat import (
                            FlattenedTensorBucket,
                            MultiprocessingSerializer,
                        )

                        item = self._staged_buckets[index]
                        header = item["header"]
                        named_tensors = self._load_staged_bucket(item)
                        flat = FlattenedTensorBucket(named_tensors=named_tensors)
                        receiver = getattr(self._rollout, "tensor_weight_sync_target", self._rollout)
                        fanout = int(getattr(receiver, "weight_payload_fanout", 0))
                        if fanout <= 0:
                            raise RuntimeError(f"direct vLLM reported invalid TP payload fanout {fanout}")
                        if fanout != int(header["tp_world_size"]):
                            raise RuntimeError(
                                f"direct vLLM TP fanout changed between extract and push: "
                                f"header={header['tp_world_size']}, current={fanout}"
                            )
                        payloads = []
                        payload_tensors = []
                        metadata = flat.get_metadata()
                        source_flat = flat.get_flattened_tensor()
                        for _tp_rank in range(fanout):
                            # CPU shared-memory handles are process-local and
                            # safely copied by each vLLM worker to its own GPU.
                            # One distinct storage per rank avoids one-shot FD
                            # reuse and cross-device CUDA IPC mappings.
                            rank_flat = source_flat.clone()
                            payload_tensors.append(rank_flat)
                            payloads.append(
                                MultiprocessingSerializer.serialize(
                                    {
                                        "flattened_tensor": rank_flat,
                                        "metadata": metadata,
                                    },
                                    output_str=True,
                                )
                            )
                        response = self._rollout.update_weights_from_tensor(
                            header=header,
                            payloads=payloads,
                            tp_world_size=fanout,
                            load_format="flattened_bucket",
                            flush_cache=bool(self._flush_cache and header["is_last"]),
                            track_prefix=self._track_prefix,
                        )
                        last_publication = validate_receipts(header, response, fanout=fanout)
                        if header["is_last"] and self._flush_cache and not last_publication.get("prefix_cache_reset"):
                            raise RuntimeError("direct vLLM committed weights without resetting the prefix cache")
                        bucket_publications.append(last_publication)
                        item["data_path"].unlink(missing_ok=True)
                        item["header_path"].unlink(missing_ok=True)
                    except BaseException as exc:
                        local_error = f"{type(exc).__name__}: {exc}"
                    finally:
                        del payloads, payload_tensors, flat, named_tensors
                        if torch.cuda.is_available():
                            torch.cuda.ipc_collect()
                            torch.cuda.empty_cache()
                self._coordinated_raise(f"bucket {index} publication", local_error)
            publication = self._broadcast_from_rank_zero(
                self._build_verification_publication(bucket_publications) if self._global_rank == 0 else None
            )
            last_publication = publication
            local_error = None
            if self._global_rank == 0:
                try:
                    self._write_receipt_artifact(publication)
                except BaseException as exc:
                    local_error = f"{type(exc).__name__}: {exc}"
            self._coordinated_raise("receipt artifact write", local_error)
            success = True
        finally:
            try:
                if success:
                    self._next_model_version += 1
                    self._last_publication = dict(last_publication or {})
                    self._publication_history.append(dict(self._last_publication))
                    if self._actor_update_pending:
                        self._actor_baseline = self._actor_candidate
                        self._actor_candidate = None
                        self._actor_baseline_source = self._actor_candidate_source
                        self._actor_candidate_source = None
                        self._actor_update_pending = False
                        self._verification.update(
                            {
                                "pending_update": False,
                                "last_sync_id": self._last_publication.get("sync_id"),
                                "last_model_version": self._last_publication.get("model_version"),
                            }
                        )
            finally:
                self._cleanup_stage_files()

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def sync(self) -> None:
        """Extract and push without retaining full-model CPU state."""
        self.extract()
        self.push()

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def cleanup(self) -> None:
        """Remove this rank's pending publication metadata and rank-0 staging files."""
        self._cleanup_stage_files()


__all__ = ["FSDPVLLMFullWeightSync"]
