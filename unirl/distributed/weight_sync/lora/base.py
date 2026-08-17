"""Shared base for the v2 LoRA weight-sync handlers."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from unirl.distributed.group.remote import Remote

logger = logging.getLogger(__name__)


def _extract_canonical_lora(backend: Any, *, param_prefix: str, adapter_name: str):
    """Extract canonical-format LoRA tensors + the PEFT config from the backend."""
    from unirl.distributed.weight_sync.payload import _peft_config_dict
    from unirl.utils.peft_merge import extract_lora_tensors

    model = backend.model
    weight_sync_dtype = getattr(backend, "weight_sync_dtype", None)
    lora_tensors = extract_lora_tensors(
        model, param_prefix=param_prefix, adapter_name=adapter_name, dtype=weight_sync_dtype
    )
    peft_config = _peft_config_dict(model, adapter_name)
    return lora_tensors, peft_config


class LoraWeightSyncBase(Remote):
    """Base for LoRA weight-sync handlers — extraction + verify; subclasses push."""

    def __init__(
        self,
        *,
        backend: Any,
        param_prefix: str = "",
        adapter_name: Optional[str] = None,
        verify: bool = False,
        track_prefix: str = "",
    ) -> None:
        super().__init__()
        self._backend = backend
        from unirl.utils.peft_merge import lora_targets_ep_experts

        if lora_targets_ep_experts(backend.model):
            raise ValueError(
                f"{type(self).__name__}: LoRA targeting EP-sharded fused experts "
                "is unsupported; target attention/shared non-EP modules instead."
            )
        self._param_prefix = str(param_prefix or "")
        self._adapter_name = str(adapter_name) if adapter_name is not None else str(backend.rollout_adapter_name)
        self._verify = bool(verify)
        self._track_prefix = str(track_prefix or "")

    def _extract(self):
        """Extract the canonical adapter (+ ``track_prefix``) and PEFT config."""
        lora_tensors, peft_config = _extract_canonical_lora(
            self._backend, param_prefix=self._param_prefix, adapter_name=self._adapter_name
        )
        if self._track_prefix:
            lora_tensors = {f"{self._track_prefix}.{k}": v for k, v in lora_tensors.items()}
        return lora_tensors, peft_config

    @staticmethod
    def _expected_checksums(lora_tensors: Dict[str, Any], peft_config: Dict):
        """Trainer-side expected ``(lora_A, lora_B)`` hash multisets."""
        from unirl.distributed.weight_sync.transfer.checksum import (
            compute_lora_checksums_post_optimize,
        )

        expected = compute_lora_checksums_post_optimize(lora_tensors, peft_config)
        exp_a = sorted(h for k, h in expected.items() if ".lora_A." in k)
        exp_b = sorted(h for k, h in expected.items() if ".lora_B." in k)
        return exp_a, exp_b

    def _assert_loaded(
        self,
        exp_a: List[str],
        exp_b: List[str],
        loaded: Dict,
        *,
        topology: Dict,
        label: str,
    ) -> None:
        """Assert one engine's loaded LoRA matches the expected multisets."""
        if not exp_a or not exp_b:
            raise RuntimeError(
                f"[LoRA-SYNC] verify FAILED on {label}: expected checksum sets must be non-empty "
                f"(lora_A={len(exp_a)}, lora_B={len(exp_b)})."
            )
        if not isinstance(topology, dict) or not topology:
            raise RuntimeError(f"[LoRA-SYNC] verify FAILED on {label}: rollout topology is empty.")
        try:
            expected_topology = {int(stage_id): int(tp) for stage_id, tp in topology.items()}
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"[LoRA-SYNC] verify FAILED on {label}: invalid rollout topology {topology!r}.") from exc
        if any(tp <= 0 for tp in expected_topology.values()):
            raise RuntimeError(f"[LoRA-SYNC] verify FAILED on {label}: invalid rollout topology {topology!r}.")

        if not isinstance(loaded, dict) or not loaded:
            raise RuntimeError(f"[LoRA-SYNC] verify FAILED on {label}: engine returned no loaded LoRA checksums.")
        try:
            loaded_by_stage = {int(stage_id): per_rank for stage_id, per_rank in loaded.items()}
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"[LoRA-SYNC] verify FAILED on {label}: invalid loaded stage keys.") from exc
        if set(loaded_by_stage) != set(expected_topology):
            raise RuntimeError(
                f"[LoRA-SYNC] verify FAILED on {label}: expected stages {sorted(expected_topology)}, "
                f"engine returned {sorted(loaded_by_stage)}."
            )

        for stage_id, tp in sorted(expected_topology.items()):
            per_rank = loaded_by_stage[stage_id]
            if not isinstance(per_rank, (list, tuple)) or len(per_rank) != tp:
                actual = len(per_rank) if isinstance(per_rank, (list, tuple)) else type(per_rank).__name__
                raise RuntimeError(
                    f"[LoRA-SYNC] verify FAILED on {label}, stage {stage_id}: "
                    f"expected {tp} TP rank readbacks, got {actual}."
                )
            for rank_idx, layer_map in enumerate(per_rank):
                if not isinstance(layer_map, dict) or not layer_map:
                    raise RuntimeError(
                        f"[LoRA-SYNC] verify FAILED on {label}, stage {stage_id} rank {rank_idx}: "
                        "engine returned no loaded LoRA layers."
                    )
                act_a = sorted(
                    checksum
                    for fields in layer_map.values()
                    for field, checksum in fields.items()
                    if field == "lora_a" or field.startswith("lora_a.")
                )
                act_b = sorted(
                    checksum
                    for fields in layer_map.values()
                    for field, checksum in fields.items()
                    if field == "lora_b" or field.startswith("lora_b.")
                )
                if act_a != exp_a or act_b != exp_b:
                    raise RuntimeError(
                        f"[LoRA-SYNC] verify FAILED on {label}, stage {stage_id} rank "
                        f"{rank_idx}: expected {len(exp_a)} lora_A / {len(exp_b)} lora_B "
                        f"hashes, engine loaded {len(act_a)} / {len(act_b)} "
                        f"(A_match={act_a == exp_a}, B_match={act_b == exp_b}). Likely a "
                        f"transport bug or a param_prefix mismatch ({self._param_prefix!r})."
                    )


__all__ = ["LoraWeightSyncBase"]
