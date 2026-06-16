"""Re-home the ``sglang-drl`` fork's text-encoder *conditions* emission (LIN-365).

UniRL's GRPO recipes that run ``populate_conditions=true`` consume
engine-emitted text-encoder embeddings: the response translator
(``rollout/engine/sglang/response.py:_build_text_conditions``) reads, per
``GenerationResult``::

    result.prompt_embeds, result.pooled_prompt_embeds, result.encoder_attention_mask,
    result.negative_prompt_embeds, result.neg_pooled_prompt_embeds

Stock upstream ``GenerationResult`` / ``OutputBatch`` do NOT carry these
(fork-only), and upstream ``SamplingParams`` rejects ``return_prompt_embeds`` --
so the SD3 GRPO e2e crashes at
``SamplingParams.__init__() got an unexpected keyword argument
'return_prompt_embeds'``. This patch re-hosts the fork's conditions path on stock
upstream WITHOUT editing sglang source.

The flags themselves (``return_prompt_embeds`` / ``return_negative_prompt_embeds``)
are injected as ``SamplingParams`` fields by the sibling ``patch_sampling_io``
(see ``_SP_INJECT_FIELDS``); since ``Req`` has no such field, ``Req.__getattr__``
delegates the read to ``sampling_params``, so the worker sees them as
``result.return_prompt_embeds`` / ``result.return_negative_prompt_embeds``.

WHAT THIS PATCH DOES (all setattr / dataclass-field-injection / AROUND-wrap):

1. **OutputBatch + GenerationResult field injection.** Add the 6 embed fields to
   each dataclass (mirrors the fork's schedule_batch.py / entrypoints/utils.py
   diffs) so they round-trip through ``dataclasses.fields`` / ``replace`` and the
   scheduler<->driver IPC.

2. **Copy the 6 fields off the ``Req`` onto the OutputBatch, gated on the flags**,
   at the seam where the OutputBatch is actually built. In the MONOLITHIC path the
   terminal ``DecodingStage.forward(batch) -> OutputBatch`` constructs it directly,
   so ``GPUWorker._req_to_output_batch`` is bypassed (it fires only on the disagg
   raw-Req path) -- we therefore AROUND-wrap BOTH ``DecodingStage.forward`` (2a)
   and ``_req_to_output_batch`` (2b), sharing ``_copy_conditions``. Source-field
   mapping is the fork's (``gpu_worker.py`` OutputBatch construction diff)::

       prompt_embeds          <- result.prompt_embeds
       pooled_prompt_embeds   <- result.pooled_embeds
       encoder_attention_mask <- result.prompt_embeds_mask
       negative_prompt_embeds <- result.negative_prompt_embeds
       neg_pooled_prompt_embeds <- result.neg_pooled_embeds
       negative_attention_mask  <- result.negative_prompt_embeds_mask

   Upstream's ``TextEncodingStage.forward`` ALREADY populates the positive batch
   fields (``prompt_embeds`` / ``pooled_embeds`` / ``prompt_embeds_mask`` -- the
   embeds-aligned mask the DiT actually attends under) and, when CFG is active, the
   negative ones (``negative_prompt_embeds`` / ``neg_pooled_embeds`` /
   ``negative_prompt_embeds_mask``) -- so we only COPY, never re-encode.
   That is why no text-encoding AROUND-wrap is needed here (see RISKS for why the
   fork's zeros-fallback / ``_expand`` re-capture is intentionally dropped).

3. **AROUND-wrap ``GPUWorker._merge_expanded_output_batches``** (the grouped
   nopp>1 path) to concat the per-output embed fields dim-0 onto the merged
   OutputBatch -- upstream's merge helpers do not carry them. No-op in the single
   path (that path never calls merge).

4. **AROUND-wrap ``DiffGenerator._result_common``** to copy the idx-th output's
   embed slice from the (single or merged) OutputBatch into the per-result
   GenerationResult kwargs. Slicing mirrors the fork's ``_slice_embed_list`` /
   upstream's ``samples_out[idx]`` per-output convention: each field is a
   ``list[Tensor]`` (one per text encoder), sliced ``t[idx:idx+1]`` so each
   GenerationResult carries its own single-sample embeds and the response
   translator's dim-0 concat over results reconstructs the batch.

Idempotent; setattr / field-injection / AROUND-wrap only -- no sglang source edits.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import field

logger = logging.getLogger(__name__)

# The 6 conditions fields, in the fork's order. All default to None and are
# typed ``list[torch.Tensor] | None`` (one entry per text encoder) on
# OutputBatch; ``Any``-typed on GenerationResult to match its existing style.
_COND_FIELDS = (
    "prompt_embeds",
    "pooled_prompt_embeds",
    "encoder_attention_mask",
    "negative_prompt_embeds",
    "neg_pooled_prompt_embeds",
    "negative_attention_mask",
)

# result(Req) source attr -> OutputBatch dest attr (the fork's gpu_worker mapping).
# Positives gate on return_prompt_embeds; negatives on return_negative_prompt_embeds.
#
# NOTE (LIN-365): the emitted ``encoder_attention_mask`` carries the model's
# EMBEDS-ALIGNED mask (``prompt_embeds_mask`` — the very mask the server's DiT
# attends under, built by the text-encoding stage over the post-prefix-strip
# embeds), NOT the raw ``prompt_attention_mask`` (which for prefix-stripped models
# like Qwen-Image is longer than the embeds). The response translator mounts it
# only when its fused length matches the fused embeds
# (``utils.tracks.fuse_text_conditions``): Qwen-Image's single-encoder mask matches
# and flows through to replay; SD3's per-encoder mask fuses to 410 vs the merged
# 333-token embeds, so it is dropped there (SD3's ``predict_noise`` ignores the
# mask anyway — see the historic ~68x LoRA-gradient dilution that motivated this
# guard). This source-of-truth transmit + shape guard replaces both the old global
# mask-drop and the adapter-side all-ones backfill.
_POS_MAP = {
    "prompt_embeds": "prompt_embeds",
    "pooled_prompt_embeds": "pooled_embeds",
    "encoder_attention_mask": "prompt_embeds_mask",
}
_NEG_MAP = {
    "negative_prompt_embeds": "negative_prompt_embeds",
    "neg_pooled_prompt_embeds": "neg_pooled_embeds",
    "negative_attention_mask": "negative_prompt_embeds_mask",
}

# Sentinels.
_OUTPUT_BATCH_FIELDS_SENTINEL = "_unirl_conditions_output_batch_fields"
_GEN_RESULT_FIELDS_SENTINEL = "_unirl_conditions_gen_result_fields"
_REQ_TO_OB_SENTINEL = "_unirl_conditions_req_to_ob"
_DECODING_SENTINEL = "_unirl_conditions_decoding"
_MERGE_SENTINEL = "_unirl_conditions_merge"
_RESULT_COMMON_SENTINEL = "_unirl_conditions_result_common"


def patch_conditions() -> None:
    """Install the fork's text-encoder conditions emission on stock upstream.

    Import-safe (all sglang imports are local) and idempotent.
    """
    import sglang.multimodal_gen.runtime.entrypoints.diffusion_generator as dg_mod
    import sglang.multimodal_gen.runtime.entrypoints.utils as utils_mod
    import sglang.multimodal_gen.runtime.managers.gpu_worker as gw_mod
    import sglang.multimodal_gen.runtime.pipelines_core.schedule_batch as sb_mod
    from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import (
        DecodingStage,
    )

    # (1) Field injection on the two dataclasses that carry the conditions.
    #
    # Both are registered in ``__dataclass_fields__`` (so they round-trip through
    # ``fields`` / ``replace`` / ``asdict`` / pickle) AND have their generated
    # ``__init__`` wrapped to strip-then-reapply the 6 keys -- because once a field
    # lives in ``__dataclass_fields__``, ``dataclasses.replace`` passes EVERY field
    # as a kwarg to ``__init__``, and the frozen generated ``__init__`` would reject
    # the post-hoc ones. GenerationResult is additionally built directly with these
    # kwargs (``GenerationResult(**common, ...)`` in DiffGenerator.generate, fed by
    # our _result_common wrap). Same strip-then-reapply pattern patch_sampling_io
    # uses for SamplingParams.
    _inject_dataclass_fields(
        sb_mod.OutputBatch,
        _OUTPUT_BATCH_FIELDS_SENTINEL,
        type_str="list[torch.Tensor] | None",
    )
    _inject_dataclass_fields(
        utils_mod.GenerationResult,
        _GEN_RESULT_FIELDS_SENTINEL,
        type_str="Any",
    )

    # (2a) Monolithic path: copy embeds batch(Req) -> OutputBatch at the terminal
    #      decoding stage (where the OutputBatch is actually built).
    _wrap_decoding_stage(DecodingStage)

    # (2b) Disagg/raw-Req path: copy embeds Req -> OutputBatch in the per-Req
    #      conversion (this seam never fires on the monolithic decoding path).
    _wrap_req_to_output_batch(gw_mod.GPUWorker)

    # (3) Carry embeds through the grouped (nopp>1) merge.
    _wrap_merge_expanded_output_batches(gw_mod.GPUWorker)

    # (4) Copy/slice embeds OutputBatch -> GenerationResult per output index.
    _wrap_result_common(dg_mod.DiffGenerator)


# ------------------------------------------------------------------ #
# (1) Dataclass field injection (OutputBatch / GenerationResult)
# ------------------------------------------------------------------ #


def _make_dataclass_field(name: str, default, type_str: str):
    """Build a ``dataclasses.Field`` equivalent to ``name: type = default``.

    Mirrors ``patch_sampling_io._make_dataclass_field``: registered as a real
    (init=True) field so ``dataclasses.fields`` / ``replace`` / ``asdict`` treat
    it like any source-declared field.
    """
    f = field(default=default)
    f.name = name
    f.type = type_str
    f._field_type = dataclasses._FIELD
    return f


def _inject_dataclass_fields(cls, sentinel: str, *, type_str: str) -> None:
    """Register the 6 conditions fields onto a plain ``@dataclass`` ``cls``.

    Registration (``__dataclass_fields__`` entry + class-level ``None`` default)
    makes the fields visible to ``dataclasses.fields`` / ``replace`` / ``asdict``,
    makes ``getattr(obj, name)`` return ``None`` pre-construction, and lets pickle
    round-trip them via ``__dict__``.

    The dataclass-generated ``__init__`` is frozen at class-creation time and does
    not know the post-hoc fields; yet once a field is in ``__dataclass_fields__``,
    ``dataclasses.replace`` passes EVERY field as a kwarg, and
    ``GenerationResult`` is built directly as ``GenerationResult(**common, ...)``
    with our keys. So we wrap ``__init__`` to strip the 6 keys before the strict
    generated ``__init__`` runs, then re-apply via ``object.__setattr__`` -- the
    same strip-then-reapply pattern ``patch_sampling_io`` uses for SamplingParams.
    """
    if getattr(cls, sentinel, False):
        return

    own_fields = cls.__dict__.get("__dataclass_fields__")
    if own_fields is None:  # pragma: no cover - both are dataclasses
        own_fields = dict(getattr(cls, "__dataclass_fields__", {}))
        cls.__dataclass_fields__ = own_fields

    for name in _COND_FIELDS:
        if name not in own_fields:
            own_fields[name] = _make_dataclass_field(name, None, type_str)
        # Class-level default so getattr(obj, name) works even when our write
        # sites did not set it (flags off).
        if name not in cls.__dict__:
            setattr(cls, name, None)

    orig_init = cls.__dict__.get("__init__")
    if orig_init is not None and not getattr(orig_init, sentinel, False):

        def __init__(self, *args, __orig_init=orig_init, **kwargs):
            extra = {k: kwargs.pop(k) for k in _COND_FIELDS if k in kwargs}
            __orig_init(self, *args, **kwargs)
            for k, v in extra.items():
                object.__setattr__(self, k, v)

        setattr(__init__, sentinel, True)
        cls.__init__ = __init__

    setattr(cls, sentinel, True)


# ------------------------------------------------------------------ #
# (2) Req -> OutputBatch copy in _req_to_output_batch
# ------------------------------------------------------------------ #


def _wrap_req_to_output_batch(GPUWorker) -> None:
    """AROUND-wrap the ``@staticmethod`` Req -> OutputBatch conversion.

    Runs in both forward paths: ``_execute_forward_common`` (single) and
    ``_forward_group`` (grouped, per result before merge). Copies the embed
    fields off ``result`` (a ``Req``; reads delegate to ``sampling_params`` for
    the flags) onto the returned OutputBatch, gated on the flags. Verbatim source
    mapping from the fork's ``gpu_worker.py`` OutputBatch diff.
    """
    orig = GPUWorker.__dict__.get("_req_to_output_batch")
    if orig is None:
        raise AttributeError("GPUWorker._req_to_output_batch missing upstream")
    raw = orig.__func__ if isinstance(orig, staticmethod) else orig
    if getattr(raw, _REQ_TO_OB_SENTINEL, False):
        return

    def _req_to_output_batch(result):
        output_batch = raw(result)
        _copy_conditions(result, output_batch)
        return output_batch

    setattr(_req_to_output_batch, _REQ_TO_OB_SENTINEL, True)
    GPUWorker._req_to_output_batch = staticmethod(_req_to_output_batch)


def _copy_conditions(src, output_batch) -> None:
    """Copy the gated conditions fields off ``src`` (a Req) onto ``output_batch``.

    Shared by the decoding-stage wrap (monolithic path: the OutputBatch is built
    in ``DecodingStage.forward``) and ``_req_to_output_batch`` (disagg/raw-Req
    path). Source mapping is the fork's ``gpu_worker.py`` OutputBatch diff;
    positives gate on ``return_prompt_embeds``, negatives on
    ``return_negative_prompt_embeds`` (delegated to ``sampling_params``).
    """
    if getattr(src, "return_prompt_embeds", False):
        for dst, srcattr in _POS_MAP.items():
            setattr(output_batch, dst, _to_cpu_embed_list(getattr(src, srcattr, None)))
    if getattr(src, "return_negative_prompt_embeds", False):
        for dst, srcattr in _NEG_MAP.items():
            setattr(output_batch, dst, _to_cpu_embed_list(getattr(src, srcattr, None)))


def _wrap_decoding_stage(DecodingStage) -> None:
    """AROUND-wrap ``DecodingStage.forward`` to carry conditions onto its OutputBatch.

    In the monolithic path the pipeline's terminal stage is decoding, whose
    ``forward(batch) -> OutputBatch`` (decoding.py) builds the OutputBatch directly
    from the ``batch`` Req -- so ``GPUWorker._req_to_output_batch`` (which only runs
    on the disagg raw-Req path) never fires, and the conditions never reach the
    OutputBatch. The ``batch`` Req still carries ``prompt_embeds`` (set by
    SD3ConditioningStage and untouched by timestep/latent/denoising), so copy them
    onto the returned OutputBatch here, gated on the flags. Runs per-output in the
    grouped path (``run_grouped_requests`` -> ``forward`` per Req).
    """
    orig = DecodingStage.__dict__.get("forward")
    if orig is None:
        raise AttributeError("DecodingStage.forward missing upstream")
    if getattr(orig, _DECODING_SENTINEL, False):
        return

    def forward(self, batch, server_args):
        output_batch = orig(self, batch, server_args)
        _copy_conditions(batch, output_batch)
        return output_batch

    setattr(forward, _DECODING_SENTINEL, True)
    DecodingStage.forward = forward


def _to_cpu_embed_list(value):
    """Detach + move a per-encoder ``list[Tensor]`` embed field to CPU.

    The OutputBatch is pickled across the scheduler<->driver ZMQ boundary; rollout
    tensors are materialized to CPU before transport (see
    ``rollout_denoising_mixin``'s ``.cpu()`` on ``dit_trajectory`` /
    ``rollout_log_probs``). Text-encoder embeds come off the batch on GPU, so we
    mirror that contract here -- otherwise a CUDA tensor would have to cross the
    process boundary (CUDA-IPC fragile / cross-device). The response translator
    reads them with ``.detach().cpu()`` so CPU here is exactly what it expects.

    Returns ``None`` unchanged; preserves a possible bare tensor (defensive --
    upstream stores these as lists per encoder) and per-element ``None`` holes.
    """
    if value is None:
        return None
    import torch

    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, (list, tuple)):
        moved = [t.detach().cpu() if torch.is_tensor(t) else t for t in value]
        return moved if isinstance(value, list) else type(value)(moved)
    return value


# ------------------------------------------------------------------ #
# (3) Grouped-merge carry (nopp>1 path)
# ------------------------------------------------------------------ #


def _wrap_merge_expanded_output_batches(GPUWorker) -> None:
    """AROUND-wrap the grouped-output merge to carry conditions dim-0 concatenated.

    Upstream ``_merge_expanded_output_batches`` (and its collect/finalize helpers)
    does not carry the embed fields, so for an expanded ``num_outputs_per_prompt>1``
    request they would be dropped. We re-attach them by concatenating each
    field's per-encoder tensors across the per-output batches along dim-0, so the
    merged OutputBatch carries batch-dim-``N`` embeds that ``_result_common`` can
    then slice per output index.

    No-op in the single forward path -- that path returns the per-Req OutputBatch
    directly and never calls this method.
    """
    orig = GPUWorker.__dict__.get("_merge_expanded_output_batches")
    if orig is None:
        raise AttributeError("GPUWorker._merge_expanded_output_batches missing upstream")
    raw = orig.__func__ if isinstance(orig, staticmethod) else orig
    if getattr(raw, _MERGE_SENTINEL, False):
        return

    def _merge_expanded_output_batches(output_batches):
        merged = raw(output_batches)
        _merge_conditions(merged, output_batches)
        return merged

    setattr(_merge_expanded_output_batches, _MERGE_SENTINEL, True)
    GPUWorker._merge_expanded_output_batches = staticmethod(_merge_expanded_output_batches)


def _merge_conditions(merged, output_batches) -> None:
    """Concat each conditions field dim-0 across per-output batches onto ``merged``.

    Each field is ``list[Tensor]`` (per encoder); we concat the i-th encoder's
    tensor across all batches that carry it. If any batch is missing the field
    (None), the field is left None on ``merged`` -- positives are always present
    when ``return_prompt_embeds`` is set, negatives only under CFG.
    """
    import torch

    for name in _COND_FIELDS:
        per_batch = [getattr(ob, name, None) for ob in output_batches]
        if any(v is None for v in per_batch):
            continue
        if not per_batch:
            continue
        num_encoders = len(per_batch[0])
        # All batches must agree on encoder count to concat positionally.
        if any(len(v) != num_encoders for v in per_batch):
            logger.warning("conditions merge: inconsistent encoder count for %s; skipping", name)
            continue
        merged_list = []
        for enc_idx in range(num_encoders):
            tensors = [v[enc_idx] for v in per_batch]
            if any(t is None for t in tensors):
                merged_list.append(None)
            else:
                merged_list.append(torch.cat(tensors, dim=0))
        setattr(merged, name, merged_list)


# ------------------------------------------------------------------ #
# (4) OutputBatch -> GenerationResult copy/slice in _result_common
# ------------------------------------------------------------------ #


def _wrap_result_common(DiffGenerator) -> None:
    """AROUND-wrap ``DiffGenerator._result_common`` to add per-output embed slices.

    ``_result_common(req, output_batch, generation_time, output_index)`` returns
    the kwargs dict shared by every ``GenerationResult(**common, ...)`` call. We
    add the 6 conditions fields, slicing each per-encoder tensor ``t[idx:idx+1]``
    by ``output_index`` so each result carries its own single-sample embeds.

    Single path: ``output_batch`` is the per-Req batch (batch dim 1), idx=0 ->
    slice [0:1]. Grouped path: ``output_batch`` is the merged batch (batch dim N),
    idx in 0..N-1 -> slice [idx:idx+1]. The response translator concatenates over
    results (dim-0) to reconstruct the batch either way.
    """
    orig = DiffGenerator.__dict__.get("_result_common")
    if orig is None:
        raise AttributeError("DiffGenerator._result_common missing upstream")
    raw = orig.__func__ if isinstance(orig, staticmethod) else orig
    if getattr(raw, _RESULT_COMMON_SENTINEL, False):
        return

    def _result_common(req, output_batch, generation_time, output_index=None):
        common = raw(req, output_batch, generation_time, output_index)
        idx = 0 if output_index is None else int(output_index)
        for name in _COND_FIELDS:
            common[name] = _slice_embed_list(getattr(output_batch, name, None), idx)
        return common

    setattr(_result_common, _RESULT_COMMON_SENTINEL, True)
    DiffGenerator._result_common = staticmethod(_result_common)


def _slice_embed_list(embed_list, idx: int):
    """Slice the idx-th sample out of a per-encoder ``list[Tensor]`` field.

    Returns a new list with each tensor sliced ``t[idx:idx+1]`` (keeps the batch
    dim), or ``None`` when the field is absent. Mirrors the fork's
    ``_slice_embed_list`` in ``diffusion_generator.py``.
    """
    if embed_list is None:
        return None
    return [t[idx : idx + 1] if t is not None else None for t in embed_list]
