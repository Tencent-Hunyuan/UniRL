"""Base class for v2 full-(base-)weight sync handlers.

Counterpart to ``weight_sync/lora.py`` (LoRA-only). A ``FullWeightSync``
lives on the TRAIN slab as a sibling ``Remote`` of the FSDP backend (and, in
shared-process colocate, of the rollout engine) and pushes the freshly-trained
*full* weights into the rollout engine(s).

Subclasses pick a transport:
  - ``nccl.NCCLWeightSync``   — separate slabs (cross-node capable).
  - ``tensor.TensorWeightSync`` — colocate (serialized-tensor handoff).
  - ``ipc.IPCWeightSync``     — colocate (bucketed CUDA-IPC over ZMQ).

The base provides the transport-agnostic weight walk: full-tensor
materialization (FSDP shard → replicated, a train-mesh collective) and
size-bounded bucketing. All model / torch-heavy imports are deferred into the
methods so the driver can import this module to reference the class for
``remote(...)`` without eagerly pulling torch.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, List, Optional, Tuple

from unirl.config.require import require
from unirl.distributed.group.remote import Remote

logger = logging.getLogger(__name__)


def _one_star(s: object) -> bool:
    """True if ``s`` is a string containing exactly one ``"*"``."""
    return isinstance(s, str) and s.count("*") == 1


def _validate_name_remap(
    name_remap: Optional[Dict[str, Optional[str]]],
) -> Dict[str, Optional[str]]:
    """Validate + return the ordered ``name_remap`` rewrite rules (fail-closed)."""
    rules = dict(name_remap or {})
    keys = list(rules.keys())
    for i, (key, value) in enumerate(rules.items()):
        require(_one_star(key), f"name_remap key {key!r} needs one '*'.")
        require(value is None or _one_star(value), f"name_remap[{key!r}] value: null or one '*'; got {value!r}.")
        require(key != "*" or i == len(keys) - 1, f"name_remap '*' must be last; shadows {keys[i + 1 :]!r}.")
    return rules


def _apply_name_remap(name: str, name_remap: Dict[str, Optional[str]]) -> Optional[str]:
    """Apply the ordered single-``*`` rewrite; return the new name or None to drop."""
    for key, value in name_remap.items():
        pre, _, post = key.partition("*")
        # length guard: pre and post must not overlap inside name
        if name.startswith(pre) and name.endswith(post) and len(name) >= len(pre) + len(post):
            if value is None:
                return None
            vpre, _, vpost = value.partition("*")
            return vpre + name[len(pre) : len(name) - len(post)] + vpost
    return name


class FullWeightSync(Remote):
    """Base for full-weight sync Remotes.

    Subclasses implement ``sync()`` (push current weights) plus whatever
    one-time connection setup their transport needs. ``backend`` is the FSDP
    backend sibling whose ``.model`` is the weight source.

    ``lora_merged`` selects what gets pushed: ``False`` (default) syncs the raw
    base weights (meaningful for full fine-tuning); ``True`` folds the trained
    LoRA deltas into the base weights and pushes the merged full model
    (meaningful for a LoRA run served without a separate adapter).

    ``adapter_name`` names which PEFT adapter's delta is folded/pushed. ``None``
    (default) defers to the backend's ``rollout_adapter_name`` — the EMA shadow
    (``"old"``) for DiffusionNFT, which is what carries the EMA-smoothed weights
    into a SEPARATE rollout engine where the in-process ``apply_eval_ema`` swap
    cannot reach; ``"default"`` otherwise. A non-default adapter REQUIRES
    ``lora_merged=True``: a full-weight receiver loads merged base weights and
    does not apply a separately-shipped delta, so the shadow must be folded in
    before the push (the ctor fails closed otherwise).

    ``track_prefix`` routes the push to one child of a ``ComposedRolloutEngine``
    (e.g. ``"ar"`` / ``"diffusion"``); empty (default) targets a single-model
    engine. Each transport forwards it to the rollout so the receiver demuxes.

    ``wire_dtype`` (e.g. ``"bf16"``) casts floating tensors to the rollout
    engine's dtype in the weight walk — shard-side, BEFORE the FSDP
    all-gather — so the gather, the bucket sizing, and every transport move
    wire-width bytes. Set it in the ``sync:`` block whenever the training
    masters are wider than the engine (``model_precision: fp32`` →
    ``wire_dtype: bf16``). Receivers ``copy_``-cast on load, so this is a
    bandwidth/memory policy, not a correctness one; ``None`` (default) ships
    tensors as-is.
    """

    def __init__(
        self,
        *,
        backend,
        bucket_size_mb: int = 512,
        flush_cache: bool = True,
        lora_merged: bool = False,
        adapter_name: Optional[str] = None,
        name_remap: Optional[Dict[str, Optional[str]]] = None,
        track_prefix: str = "",
        wire_dtype: Any = None,
    ) -> None:
        super().__init__()
        # Deferred: unirl.utils.dtypes imports torch at module scope, and this
        # module must stay driver-importable without torch.
        from unirl.utils.dtypes import parse_torch_dtype

        self._backend = backend
        self._bucket_bytes = int(bucket_size_mb) * 1024 * 1024
        self._flush_cache = bool(flush_cache)
        self._lora_merged = bool(lora_merged)
        # Which PEFT adapter's weights to push. ``None`` defers to the backend's
        # single source of truth (``rollout_adapter_name``): the EMA shadow
        # ("old") for DiffusionNFT, else "default". See class docstring; a
        # non-default adapter must be merged in, so fail closed.
        self._adapter_name = str(adapter_name) if adapter_name is not None else str(backend.rollout_adapter_name)
        if self._adapter_name != "default" and not self._lora_merged:
            raise ValueError(
                f"FullWeightSync: adapter_name={self._adapter_name!r} requires "
                f"lora_merged=True (a non-default adapter must be folded into the "
                f"pushed base weights; got lora_merged=False)."
            )
        # The pushed adapter is invisible in the recipe when it defaults from the
        # backend (DiffusionNFT resolves to the EMA shadow "old"), so log it once:
        # a SEPARATE rollout engine sampling under "old" is the intended faithful
        # path, not a misconfiguration.
        if self._adapter_name != "default":
            logger.info(
                "%s: rollout engine will be synced from adapter %r (%s).",
                type(self).__name__,
                self._adapter_name,
                "explicit adapter_name" if adapter_name is not None else "backend.rollout_adapter_name",
            )
        # Ordered, first-match-wins name rewrites (glob -> replacement, or None to
        # drop); see _validate_name_remap / _apply_name_remap for the contract.
        self._name_remap = _validate_name_remap(name_remap)
        # Routes the update to one child of a ComposedRolloutEngine; empty for
        # a single-model engine. Forwarded by each transport's ``sync()``.
        self._track_prefix = str(track_prefix or "")
        # Wire dtype for the weight walk (None = ship as-is); see class docstring.
        self._wire_dtype = parse_torch_dtype(wire_dtype, field_name="wire_dtype", allow_none=True)
        self.weight_version = 0

    # ------------------------------------------------------------------
    # Transport-agnostic weight walk
    # ------------------------------------------------------------------

    def _iter_full_tensors(self) -> Iterator[Tuple[str, "object"]]:
        """Yield ``(name, full_tensor)`` one at a time (lazy → bounded memory).

        Both walks apply ``_to_full_tensor`` (redistribute each FSDP ``DTensor``
        shard to Replicate → a collective over the train mesh), so this MUST run
        on every train rank in lockstep; each yields a full (unsharded) CUDA
        tensor per param, in a deterministic order.

        ``lora_merged`` selects the walk:
          - ``True``  → ``merged_state_dict`` folds LoRA deltas into the base
            weights and yields the trained module's own keys (LoRA already
            absorbed, ``.base_layer.`` flattened away).
          - ``False`` → ``raw_state_dict`` base weights, skipping
            ``.lora_A``/``.lora_B``.

        ``adapter_name`` (resolved in the ctor from ``backend.rollout_adapter_name``
        when not given) names which adapter's delta the walk reads — DiffusionNFT
        points it at the EMA ``"old"`` shadow so the rollout engine receives the
        EMA-smoothed weights.

        Each emitted name is then rewritten by ``name_remap`` (see
        ``_apply_name_remap``): a ``None`` rule drops the param (e.g. a frozen
        tower not in the receiver), and the ``"*"`` catch-all nests the trained
        submodule's bare keys under the receiver's namespace (e.g.
        ``transformer.``). It is orthogonal to LoRA-merging.

        ``wire_dtype`` (if set) is applied inside the walk, so the cast
        happens shard-side before the redistribute and every consumer
        (``_iter_buckets`` sizing included) sees wire-width tensors.
        """
        from unirl.utils.peft_merge import merged_state_dict, raw_state_dict

        remap = self._name_remap
        if self._lora_merged:
            for name, tensor in merged_state_dict(
                self._backend.model, adapter_name=self._adapter_name, dtype=self._wire_dtype
            ):
                out = _apply_name_remap(name, remap)
                if out is not None:
                    yield out, tensor
            return

        for name, tensor in raw_state_dict(
            self._backend.model, adapter_name=self._adapter_name, dtype=self._wire_dtype
        ):
            if name.endswith(".lora_A") or name.endswith(".lora_B"):
                continue
            out = _apply_name_remap(name, remap)
            if out is not None:
                yield out, tensor

    def _iter_buckets(self) -> Iterator[Tuple[List[Tuple[str, "object"]], bool]]:
        """Yield ``(bucket, is_last)`` where ``bucket`` is a list of
        ``(name, tensor)`` up to ``bucket_size_mb``.

        ``is_last`` is True only for the final bucket — used to drive
        ``flush_cache`` on the receiver.
        """
        bucket: List[Tuple[str, object]] = []
        nbytes = 0
        for name, tensor in self._iter_full_tensors():
            size = tensor.numel() * tensor.element_size()
            if bucket and nbytes + size >= self._bucket_bytes:
                yield bucket, False
                bucket, nbytes = [], 0
            bucket.append((name, tensor))
            nbytes += size
        if bucket:
            yield bucket, True

    @property
    def _my_rank(self) -> int:
        ri = self.rank_info
        return ri.rank if ri is not None else 0


__all__ = ["FullWeightSync"]
