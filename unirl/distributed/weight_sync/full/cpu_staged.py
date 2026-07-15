"""CPU-staged full-weight sync for memory-exclusive colocated engines.

This transport deliberately splits a sync into two lifecycle phases:

``extract()``
    Materialize the FSDP full state while the rollout engine is asleep and copy
    a versioned bf16 snapshot to CPU.

``push()``
    After the trainer has been offloaded and the engine woken, serialize fresh
    CPU buckets for each Omni stage and load them synchronously.  A serialized
    multiprocessing tensor handle is not reusable across consumers, so sharing
    one payload between AR and diffusion stages is explicitly forbidden here.

The initial supported topology is TP=1 for every selected stage.  That is the
only topology for which one serialized payload per stage is unambiguous.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from unirl.distributed.group.dispatch import Dispatch, distributed
from unirl.distributed.weight_sync.full.base import FullWeightSync

logger = logging.getLogger(__name__)

BAGEL_VLLM_OMNI_020_LOAD_PLAN = "bagel_vllm_omni_0_20"

# vLLM-Omni 0.20 BAGEL packs these HF source modules into one destination
# parameter.  The lane keeps sibling shards in separate loader calls so the
# returned destination-name set remains source-injective within every call.
_BAGEL_VLLM_OMNI_020_PACKED_SOURCES = (
    (".q_proj_moe_gen.", ".qkv_proj_moe_gen.", 0),
    (".k_proj_moe_gen.", ".qkv_proj_moe_gen.", 1),
    (".v_proj_moe_gen.", ".qkv_proj_moe_gen.", 2),
    (".q_proj.", ".qkv_proj.", 0),
    (".k_proj.", ".qkv_proj.", 1),
    (".v_proj.", ".qkv_proj.", 2),
    (".gate_proj.", ".gate_up_proj.", 0),
    (".up_proj.", ".gate_up_proj.", 1),
)
_BAGEL_VLLM_OMNI_020_PACKED_TARGETS = (
    ".qkv_proj_moe_gen.",
    ".qkv_proj.",
    ".gate_up_proj.",
)


class CPUStagedFullWeightSync(FullWeightSync):
    """Versioned CPU snapshot followed by a fresh per-stage tensor push."""

    requires_persistent_cpu_offload_on_single_device = True

    def __init__(
        self,
        *,
        backend: Any,
        rollout: Any,
        bucket_size_mb: int = 256,
        flush_cache: bool = True,
        lora_merged: bool = False,
        adapter_name: Optional[str] = None,
        name_remap: Optional[Dict[str, Optional[str]]] = None,
        track_prefix: str = "",
        wire_dtype: Any = "bf16",
        stage_ids: Sequence[int] = (0, 1),
        verify_names: Sequence[str] = (),
        load_plan: Optional[str] = None,
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

        import torch

        if self._wire_dtype != torch.bfloat16:
            raise ValueError(f"CPUStagedFullWeightSync currently requires wire_dtype='bf16'; got {self._wire_dtype!r}.")
        selected = tuple(dict.fromkeys(int(stage_id) for stage_id in stage_ids))
        if not selected:
            raise ValueError("CPUStagedFullWeightSync.stage_ids must not be empty.")
        if load_plan not in (None, BAGEL_VLLM_OMNI_020_LOAD_PLAN):
            raise ValueError(
                "CPUStagedFullWeightSync.load_plan must be null or "
                f"{BAGEL_VLLM_OMNI_020_LOAD_PLAN!r}; got {load_plan!r}."
            )

        self._rollout = rollout
        self._stage_ids = selected
        self._load_plan = load_plan
        self._verify_names = tuple(dict.fromkeys(str(name) for name in verify_names))
        if not self._verify_names:
            raise ValueError(
                "CPUStagedFullWeightSync.verify_names must contain at least one "
                "TP-flat parameter shared by every selected stage."
            )
        self._snapshot: Optional[List[Tuple[str, Any]]] = None
        self._snapshot_version: Optional[int] = None
        self._expected_checksums: Dict[str, str] = {}
        self._pending_stage_acks: Dict[int, int] = {}
        self._last_stage_acks: Dict[int, int] = {}

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def extract(self) -> int:
        """Copy one complete full-weight snapshot to CPU.

        The trainer owns lifecycle ordering and calls this only while the
        rollout engine is asleep.  A pending snapshot cannot be overwritten:
        doing so would make a partial push indistinguishable from a clean
        version transition.
        """
        import torch

        if self._snapshot is not None:
            raise RuntimeError(
                f"CPUStagedFullWeightSync.extract: snapshot v{self._snapshot_version} has not been pushed or discarded."
            )

        snapshot: List[Tuple[str, Any]] = []
        expected: Dict[str, str] = {}
        verify = set(self._verify_names)
        try:
            for name, tensor in self._iter_full_tensors():
                cpu_tensor = tensor.detach().to(device="cpu", copy=True).contiguous()
                snapshot.append((name, cpu_tensor))
                if name in verify:
                    from unirl.distributed.weight_sync.transfer.checksum import (
                        fingerprint_tensor,
                    )

                    expected[name] = fingerprint_tensor(cpu_tensor)
        except Exception:
            snapshot.clear()
            raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not snapshot:
            raise RuntimeError("CPUStagedFullWeightSync.extract produced an empty snapshot.")
        missing = sorted(verify.difference(expected))
        if missing:
            snapshot.clear()
            raise RuntimeError(
                f"CPUStagedFullWeightSync.extract could not find verify_names after name remapping: {missing}."
            )

        version = self.weight_version + 1
        self._snapshot = snapshot
        self._snapshot_version = version
        self._expected_checksums = expected
        self._pending_stage_acks = {}
        logger.info(
            "CPUStagedFullWeightSync extracted snapshot v%d (%d tensors, %.2f GiB).",
            version,
            len(snapshot),
            sum(t.numel() * t.element_size() for _, t in snapshot) / (1024**3),
        )
        return version

    @staticmethod
    def _bagel_vllm_omni_020_target(name: str) -> Tuple[int, str]:
        """Return the packed-load lane and destination name for one HF key."""
        if not name.startswith("language_model."):
            raise ValueError(f"{BAGEL_VLLM_OMNI_020_LOAD_PLAN} requires language_model.* source names; got {name!r}.")
        if any(target in name for target in _BAGEL_VLLM_OMNI_020_PACKED_TARGETS):
            raise ValueError(
                f"{BAGEL_VLLM_OMNI_020_LOAD_PLAN} requires unpacked HF source names; got already-packed name {name!r}."
            )
        for source, target, lane in _BAGEL_VLLM_OMNI_020_PACKED_SOURCES:
            if source in name:
                return lane, name.replace(source, target, 1)
        return 0, name

    def _planned_target(self, name: str) -> Tuple[int, str]:
        if self._load_plan == BAGEL_VLLM_OMNI_020_LOAD_PLAN:
            return self._bagel_vllm_omni_020_target(name)
        return 0, name

    def _append_sized_buckets(
        self,
        planned: List[List[Tuple[str, Any]]],
        named_tensors: Sequence[Tuple[str, Any]],
    ) -> None:
        bucket: List[Tuple[str, Any]] = []
        nbytes = 0
        for name, tensor in named_tensors:
            size = tensor.numel() * tensor.element_size()
            if bucket and nbytes + size >= self._bucket_bytes:
                planned.append(bucket)
                bucket, nbytes = [], 0
            bucket.append((name, tensor))
            nbytes += size
        if bucket:
            planned.append(bucket)

    def _iter_snapshot_buckets(self) -> Iterator[Tuple[List[Tuple[str, Any]], bool]]:
        snapshot = self._snapshot
        if snapshot is None:
            raise RuntimeError("CPUStagedFullWeightSync.push requires extract() first.")

        planned: List[List[Tuple[str, Any]]] = []
        if self._load_plan is None:
            self._append_sized_buckets(planned, snapshot)
        else:
            lanes: Dict[int, List[Tuple[str, Any]]] = {0: [], 1: [], 2: []}
            seen_sources: set[str] = set()
            seen_targets: Dict[int, set[str]] = {0: set(), 1: set(), 2: set()}
            for name, tensor in snapshot:
                if name in seen_sources:
                    raise RuntimeError(f"CPUStagedFullWeightSync load plan has duplicate source name {name!r}.")
                lane, target = self._planned_target(name)
                if target in seen_targets[lane]:
                    raise RuntimeError(
                        "CPUStagedFullWeightSync load plan is not source-injective: "
                        f"lane={lane}, duplicate target={target!r}."
                    )
                seen_sources.add(name)
                seen_targets[lane].add(target)
                lanes[lane].append((name, tensor))
            for lane in sorted(lanes):
                self._append_sized_buckets(planned, lanes[lane])

        for index, bucket in enumerate(planned):
            yield bucket, index == len(planned) - 1

    def _validate_topology(self) -> None:
        try:
            topology = {int(stage): int(tp) for stage, tp in self._rollout.tp_per_stage().items()}
        except (AttributeError, NotImplementedError) as exc:
            raise RuntimeError("CPUStagedFullWeightSync requires rollout.tp_per_stage() topology discovery.") from exc

        missing = [stage for stage in self._stage_ids if stage not in topology]
        if missing:
            raise RuntimeError(f"CPUStagedFullWeightSync stages {missing} are absent; topology={topology}.")
        unsupported = {stage: topology[stage] for stage in self._stage_ids if topology[stage] != 1}
        if unsupported:
            raise RuntimeError(f"CPUStagedFullWeightSync initially supports TP=1 per stage; got {unsupported}.")

    @staticmethod
    def _validate_bucket_ack(
        result: Any,
        *,
        stage_id: int,
        expected_received: int,
        expected_loaded_names: Optional[Sequence[str]] = None,
    ) -> None:
        if not isinstance(result, dict) or len(result) != 1:
            raise RuntimeError(
                f"CPUStagedFullWeightSync expected exactly one stage result for stage {stage_id}; got {result!r}."
            )
        result_stage, stage_result = next(iter(result.items()))
        try:
            matching_stage = not isinstance(result_stage, bool) and int(result_stage) == stage_id
        except (TypeError, ValueError):
            matching_stage = False
        if not matching_stage:
            raise RuntimeError(
                "CPUStagedFullWeightSync received an acknowledgement for the wrong stage: "
                f"expected={stage_id}, got={result_stage!r}."
            )

        leaves: List[Dict[str, Any]] = []

        def collect(value: Any) -> None:
            if isinstance(value, dict):
                if "received_count" in value or "loaded_count" in value:
                    leaves.append(value)
                else:
                    for nested in value.values():
                        collect(nested)
            elif isinstance(value, (list, tuple)):
                for nested in value:
                    collect(nested)

        collect(stage_result)
        if len(leaves) != 1:
            raise RuntimeError(
                "CPUStagedFullWeightSync expected exactly one TP=1 loader "
                f"acknowledgement for stage {stage_id}; got {result!r}."
            )
        ack = leaves[0]
        received_count = int(ack.get("received_count", -1))
        loaded_count = int(ack.get("loaded_count", -1))
        rejected = received_count != expected_received or loaded_count <= 0
        detail = ""
        if expected_loaded_names is not None:
            expected_names = tuple(expected_loaded_names)
            expected_set = set(expected_names)
            raw_loaded_names = ack.get("loaded_names")
            valid_names = isinstance(raw_loaded_names, (list, tuple)) and all(
                isinstance(name, str) for name in raw_loaded_names
            )
            loaded_names = tuple(raw_loaded_names) if valid_names else ()
            loaded_set = set(loaded_names)
            rejected = rejected or (
                len(expected_set) != expected_received
                or len(loaded_names) != len(loaded_set)
                or loaded_count != expected_received
                or loaded_set != expected_set
            )
            if not valid_names:
                detail = f", loaded_names={raw_loaded_names!r}"
            else:
                detail = (
                    f", missing={sorted(expected_set - loaded_set)!r}, unexpected={sorted(loaded_set - expected_set)!r}"
                )
        if rejected:
            raise RuntimeError(
                "CPUStagedFullWeightSync stage loader rejected a bucket: "
                f"stage={stage_id}, expected_received={expected_received}, "
                f"received={received_count}, loaded={loaded_count}{detail}."
            )

    def _push_stage(self, stage_id: int) -> None:
        from unirl.distributed.weight_sync.transfer.sgl_compat import (
            FlattenedTensorBucket,
            MultiprocessingSerializer,
        )

        for bucket, is_last in self._iter_snapshot_buckets():
            by_dtype: Dict[Any, List[Tuple[str, Any]]] = {}
            for name, tensor in bucket:
                by_dtype.setdefault(tensor.dtype, []).append((name, tensor))

            groups = list(by_dtype.values())
            for index, grouped in enumerate(groups):
                # Build new storage and a new pickle for every stage.  Reusing a
                # multiprocessing handle after another worker consumes it can
                # yield an invalid FD or shared-memory token.
                flat = FlattenedTensorBucket(named_tensors=[(name, tensor.clone()) for name, tensor in grouped])
                payload = MultiprocessingSerializer.serialize(
                    {
                        "flattened_tensor": flat.get_flattened_tensor(),
                        "metadata": flat.get_metadata(),
                    },
                    output_str=True,
                )
                result = self._rollout.update_weights_from_tensor(
                    serialized_named_tensors=[payload],
                    load_format="flattened_bucket",
                    flush_cache=(self._flush_cache and is_last and index == len(groups) - 1),
                    stage_ids=[stage_id],
                    track_prefix=self._track_prefix,
                )
                self._validate_bucket_ack(
                    result,
                    stage_id=stage_id,
                    expected_received=len(grouped),
                    expected_loaded_names=(
                        [self._planned_target(name)[1] for name, _ in grouped] if self._load_plan is not None else None
                    ),
                )

    def _verify_loaded_checksums(self) -> None:
        if not self._expected_checksums:
            return

        loaded = self._rollout.loaded_param_checksums(names=list(self._expected_checksums))
        for stage_id in self._stage_ids:
            rank_results = loaded.get(stage_id, loaded.get(str(stage_id)))
            if not isinstance(rank_results, list) or not rank_results:
                raise RuntimeError(
                    "CPUStagedFullWeightSync checksum read-back returned no ranks "
                    f"for stage {stage_id}: {rank_results!r}."
                )
            for rank, checksums in enumerate(rank_results):
                if not isinstance(checksums, dict):
                    raise RuntimeError(
                        "CPUStagedFullWeightSync checksum read-back returned an "
                        f"invalid stage {stage_id} rank {rank} payload: {checksums!r}."
                    )
                for name, expected in self._expected_checksums.items():
                    actual = checksums.get(name)
                    if actual != expected:
                        raise RuntimeError(
                            "CPUStagedFullWeightSync checksum mismatch at "
                            f"stage={stage_id}, rank={rank}, name={name!r}: "
                            f"expected={expected}, actual={actual}."
                        )

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def push(self) -> int:
        """Push the pending snapshot and require every selected stage to ack."""
        if self._snapshot is None or self._snapshot_version is None:
            raise RuntimeError("CPUStagedFullWeightSync.push requires extract() first.")

        version = self._snapshot_version
        self._validate_topology()
        try:
            for stage_id in self._stage_ids:
                self._push_stage(stage_id)
                # Every bucket returned a validated worker-loader ACK.
                self._pending_stage_acks[stage_id] = version

            expected_acks = {stage_id: version for stage_id in self._stage_ids}
            if self._pending_stage_acks != expected_acks:
                raise RuntimeError(
                    "CPUStagedFullWeightSync incomplete stage acknowledgements: "
                    f"expected={expected_acks}, got={self._pending_stage_acks}."
                )
            self._verify_loaded_checksums()
        except Exception:
            # Never retry a potentially partially-consumed snapshot.  The caller
            # keeps the engine asleep after this failure and extracts a new full
            # version before the next generation attempt.
            self._clear_pending_snapshot()
            raise

        self.weight_version = version
        self._last_stage_acks = dict(self._pending_stage_acks)
        self._clear_pending_snapshot()
        logger.info(
            "CPUStagedFullWeightSync committed snapshot v%d on stages %s.",
            version,
            list(self._stage_ids),
        )
        return version

    def _clear_pending_snapshot(self) -> None:
        self._snapshot = None
        self._snapshot_version = None
        self._expected_checksums = {}
        self._pending_stage_acks = {}

    @distributed(dispatch_mode=Dispatch.BROADCAST)
    def discard(self) -> None:
        """Drop a pending CPU snapshot after a lifecycle failure."""
        self._clear_pending_snapshot()

    @property
    def stage_versions(self) -> Dict[int, int]:
        """Last fully committed version for each selected stage."""
        return dict(self._last_stage_acks)


__all__ = ["BAGEL_VLLM_OMNI_020_LOAD_PLAN", "CPUStagedFullWeightSync"]
