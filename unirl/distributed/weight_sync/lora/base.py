"""Shared base for the v2 LoRA weight-sync handlers.

Both handlers read the trained adapter off the FSDP model identically (a
train-mesh collective) and verify it the same way; they differ only in how the
adapter reaches the engine:

- :class:`~unirl.distributed.weight_sync.lora.local.LocalLoraWeightSync` —
  same-Worker sibling, in-process push.
- :class:`~unirl.distributed.weight_sync.lora.remote.RemoteLoraWeightSync`
  — cross-process Ray push to non-sibling engines (separate slabs / HI3).

This base owns the transport-agnostic pieces — adapter extraction and the
post-load checksum compare — so subclasses implement only ``sync()`` (the push)
plus any connection setup. All model / vLLM-touching imports are deferred into
the methods so the driver can reference a class for ``remote(...)`` without
eagerly pulling torch-heavy or vLLM-only deps.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from unirl.distributed.group.remote import Remote

logger = logging.getLogger(__name__)


def _extract_canonical_lora(backend: Any, *, param_prefix: str, adapter_name: str):
    """Extract canonical-format LoRA tensors + the PEFT config from the backend.

    ``extract_lora_tensors`` redistributes each FSDP ``DTensor`` shard to a full
    tensor — a collective across the train process group — so the caller MUST run
    this on every train rank in lockstep (``BROADCAST``).
    """
    from unirl.distributed.weight_sync.payload import _peft_config_dict
    from unirl.utils.peft_merge import extract_lora_tensors

    model = backend.model
    lora_tensors = extract_lora_tensors(model, param_prefix=param_prefix)
    peft_config = _peft_config_dict(model, adapter_name)
    return lora_tensors, peft_config


class LoraWeightSyncBase(Remote):
    """Base for LoRA weight-sync handlers — extraction + verify; subclasses push.

    ``param_prefix`` is the pipeline prefix prepended to the canonical keys (e.g.
    ``"transformer."``; stripped engine-side by ``adapt_lora_for_sglang``).
    ``track_prefix`` (e.g. ``"ar"`` / ``"diffusion"``) further prefixes the keys so
    a :class:`~unirl.rollout.engine.composed.engine.ComposedRolloutEngine`
    can demux the update to one child; empty for a single-model trainer. ``verify``
    is a post-load checksum read-back, vLLM-Omni-only (the engine must expose
    ``loaded_lora_checksums``); ignored for SGLang.

    Subclasses add their own transport state (the sibling engine, or the cross-slab
    target handles) and implement ``sync()``.
    """

    def __init__(
        self,
        *,
        backend: Any,
        param_prefix: str = "",
        adapter_name: str = "default",
        verify: bool = False,
        track_prefix: str = "",
    ) -> None:
        super().__init__()
        self._backend = backend
        self._param_prefix = str(param_prefix or "")
        self._adapter_name = str(adapter_name or "default")
        self._verify = bool(verify)
        self._track_prefix = str(track_prefix or "")

    def _extract(self):
        """Extract the canonical adapter (+ ``track_prefix``) and PEFT config.

        A train-mesh collective (see :func:`_extract_canonical_lora`) — run on
        every train rank in lockstep.
        """
        lora_tensors, peft_config = _extract_canonical_lora(
            self._backend, param_prefix=self._param_prefix, adapter_name=self._adapter_name
        )
        # Prefix keys so a ComposedRolloutEngine can demux to one child.
        if self._track_prefix:
            lora_tensors = {f"{self._track_prefix}.{k}": v for k, v in lora_tensors.items()}
        return lora_tensors, peft_config

    @staticmethod
    def _expected_checksums(lora_tensors: Dict[str, Any], peft_config: Dict):
        """Trainer-side expected ``(lora_A, lora_B)`` hash multisets.

        ``lora_B`` is scaled by ``alpha/r`` to match the worker's post-``optimize``
        read-back. Returns sorted lists (multisets) compared against the engine's
        ``loaded_lora_checksums`` in :meth:`_assert_loaded`.
        """
        from unirl.rollout.engine.vllm_omni.weight_sync.checksum import (
            compute_lora_checksums_post_optimize,
        )

        expected = compute_lora_checksums_post_optimize(lora_tensors, peft_config)
        exp_a = sorted(h for k, h in expected.items() if ".lora_A." in k)
        exp_b = sorted(h for k, h in expected.items() if ".lora_B." in k)
        return exp_a, exp_b

    def _assert_loaded(self, exp_a: List[str], exp_b: List[str], loaded: Dict, *, label: str) -> None:
        """Assert one engine's loaded LoRA matches the expected multisets.

        The engine keys by vLLM-internal layer name + field (``lora_a`` /
        ``lora_b``), so a direct dict compare is impossible; instead compare the
        *multiset* of ``lora_A`` hashes and the multiset of ``lora_B`` hashes. With
        distinct per-layer weights (always true after a training step) multiset
        equality is a strong bit-equality proof and also catches a wrong
        ``param_prefix`` (which yields wrong / zero loaded layers). ``loaded`` is a
        ``{stage_id: [per_rank {layer: {field: hex}}]}`` map.
        """
        for stage_id, per_rank in loaded.items():
            for rank_idx, layer_map in enumerate(per_rank):
                act_a = sorted(f["lora_a"] for f in layer_map.values() if "lora_a" in f)
                act_b = sorted(f["lora_b"] for f in layer_map.values() if "lora_b" in f)
                if act_a != exp_a or act_b != exp_b:
                    raise RuntimeError(
                        f"[LoRA-SYNC] verify FAILED on {label}, stage {stage_id} rank "
                        f"{rank_idx}: expected {len(exp_a)} lora_A / {len(exp_b)} lora_B "
                        f"hashes, engine loaded {len(act_a)} / {len(act_b)} "
                        f"(A_match={act_a == exp_a}, B_match={act_b == exp_b}). Likely a "
                        f"transport bug or a param_prefix mismatch ({self._param_prefix!r})."
                    )


__all__ = ["LoraWeightSyncBase"]
