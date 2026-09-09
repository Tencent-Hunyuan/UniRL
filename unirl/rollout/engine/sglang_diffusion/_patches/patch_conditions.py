"""Re-home the ``sglang-drl`` fork's text-encoder *conditions* emission (LIN-365)."""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import field

logger = logging.getLogger(__name__)

# The condition fields default to None and are typed
# ``list[torch.Tensor] | None`` (one entry per text encoder) on
# OutputBatch; ``Any``-typed on GenerationResult to match its existing style.
#
# ``image_latent`` is represented as ``list[encoder=1][B, S_img, C]``.
# Expanded outputs belong to one prompt and share ``S_img``; mixed-resolution
# prompts remain separate GenerationResults and become ragged in the adapter.
#
# ``image_latent_sizes`` (Edit-Plus only) carries the prompt's
# ``vae_image_sizes`` (a ``list[tuple[int, int]]`` of pixel (W, H) pairs from
# upstream's ``preprocess_vae_image``) as ``list[encoder][source_image]``.
_COND_FIELDS = (
    "prompt_embeds",
    "audio_prompt_embeds",
    "pooled_prompt_embeds",
    "encoder_attention_mask",
    "negative_prompt_embeds",
    "negative_audio_prompt_embeds",
    "neg_pooled_prompt_embeds",
    "negative_attention_mask",
    "image_latent",
    "image_latent_sizes",
    "condition_image_latent_ids",
)

_POS_MAP = {
    "prompt_embeds": "prompt_embeds",
    "audio_prompt_embeds": "audio_prompt_embeds",
    "pooled_prompt_embeds": "pooled_embeds",
    "encoder_attention_mask": "prompt_embeds_mask",
}
_NEG_MAP = {
    "negative_prompt_embeds": "negative_prompt_embeds",
    "negative_audio_prompt_embeds": "negative_audio_prompt_embeds",
    "neg_pooled_prompt_embeds": "neg_pooled_embeds",
    "negative_attention_mask": "negative_prompt_embeds_mask",
}

_TOKEN_EMBED_DESTS = frozenset(
    {
        "prompt_embeds",
        "audio_prompt_embeds",
        "negative_prompt_embeds",
        "negative_audio_prompt_embeds",
    }
)

_OUTPUT_BATCH_FIELDS_SENTINEL = "_unirl_conditions_output_batch_fields"
_GEN_RESULT_FIELDS_SENTINEL = "_unirl_conditions_gen_result_fields"
_REQ_TO_OB_SENTINEL = "_unirl_conditions_req_to_ob"
_DECODING_SENTINEL = "_unirl_conditions_decoding"
_MERGE_SENTINEL = "_unirl_conditions_merge"
_RESULT_COMMON_SENTINEL = "_unirl_conditions_result_common"


def patch_conditions() -> None:
    """Install the fork's text-encoder conditions emission on stock upstream."""
    import sglang.multimodal_gen.runtime.entrypoints.diffusion_generator as dg_mod
    import sglang.multimodal_gen.runtime.entrypoints.utils as utils_mod
    import sglang.multimodal_gen.runtime.managers.gpu_worker as gw_mod
    import sglang.multimodal_gen.runtime.pipelines_core.schedule_batch as sb_mod
    from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import (
        DecodingStage,
    )

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

    _wrap_decoding_stage(DecodingStage)

    _wrap_req_to_output_batch(gw_mod.GPUWorker)

    _wrap_merge_expanded_output_batches(gw_mod.GPUWorker)

    _wrap_result_common(dg_mod.DiffGenerator)


def _make_dataclass_field(name: str, default, type_str: str):
    """Build a ``dataclasses.Field`` equivalent to ``name: type = default``."""
    f = field(default=default)
    f.name = name
    f.type = type_str
    f._field_type = dataclasses._FIELD
    return f


def _inject_dataclass_fields(cls, sentinel: str, *, type_str: str) -> None:
    """Register the condition fields onto a plain ``@dataclass`` ``cls``."""
    if getattr(cls, sentinel, False):
        return

    own_fields = cls.__dict__.get("__dataclass_fields__")
    if own_fields is None:  # pragma: no cover - both are dataclasses
        own_fields = dict(getattr(cls, "__dataclass_fields__", {}))
        cls.__dataclass_fields__ = own_fields

    for name in _COND_FIELDS:
        if name not in own_fields:
            own_fields[name] = _make_dataclass_field(name, None, type_str)
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


def _wrap_req_to_output_batch(GPUWorker) -> None:
    """AROUND-wrap the ``@staticmethod`` Req -> OutputBatch conversion."""
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
    """Copy the gated conditions fields off ``src`` (a Req) onto ``output_batch``."""
    if getattr(src, "return_prompt_embeds", False):
        _copy_mapped_conditions(src, output_batch, _POS_MAP)
    if getattr(src, "return_negative_prompt_embeds", False):
        _copy_mapped_conditions(src, output_batch, _NEG_MAP)
    # A request group contains replicas of one prompt, so its image latents
    # share one token grid and can remain a regular [B, S_img, C] tensor.
    image_latent = getattr(src, "image_latent", None)
    if image_latent is not None:
        values = image_latent if isinstance(image_latent, (list, tuple)) else [image_latent]
        output_batch.image_latent = _to_cpu_embed_list(values)

    vae_image_sizes = getattr(src, "vae_image_sizes", None)
    if vae_image_sizes is not None:
        output_batch.image_latent_sizes = [vae_image_sizes]

    condition_image_latent_ids = getattr(src, "condition_image_latent_ids", None)
    if condition_image_latent_ids is not None:
        values = (
            condition_image_latent_ids
            if isinstance(condition_image_latent_ids, (list, tuple))
            else [condition_image_latent_ids]
        )
        output_batch.condition_image_latent_ids = _to_cpu_embed_list(values)


def _copy_mapped_conditions(src, output_batch, mapping) -> None:
    """Copy each ``dst <- srcattr`` field, normalizing un-batched token embeds so"""
    for dst, srcattr in mapping.items():
        val = _to_cpu_embed_list(getattr(src, srcattr, None))
        if dst in _TOKEN_EMBED_DESTS:
            val = _ensure_batched_embed_list(val)
            val = _coalesce_duplicate_single_sample_encodes(val)
        setattr(output_batch, dst, val)


def _ensure_batched_embed_list(value):
    """Add a leading batch dim to any un-batched ``[seq, hidden]`` per-encoder tensor."""
    if not isinstance(value, (list, tuple)):
        return value
    out = [t if (t is None or t.dim() >= 3) else t.unsqueeze(0) for t in value]
    return out if isinstance(value, list) else type(value)(out)


def _coalesce_duplicate_single_sample_encodes(value):
    """Collapse shallow-copy duplicate prompt encodes."""
    import torch

    if not isinstance(value, (list, tuple)) or len(value) <= 1:
        return value
    if not all(torch.is_tensor(t) for t in value):
        return value

    first = value[0]
    first_shape = tuple(first.shape)
    if first.dim() < 1 or int(first.shape[0]) != 1:
        return value
    if any(tuple(t.shape) != first_shape for t in value[1:]):
        return value
    if not all(torch.equal(first, t) for t in value[1:]):
        return value
    return [first]


def _wrap_decoding_stage(DecodingStage) -> None:
    """AROUND-wrap ``DecodingStage.forward`` to carry conditions onto its OutputBatch."""
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
    """Detach + move a per-encoder ``list[Tensor]`` embed field to CPU."""
    if value is None:
        return None
    import torch

    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, (list, tuple)):
        moved = [t.detach().cpu() if torch.is_tensor(t) else t for t in value]
        return moved if isinstance(value, list) else type(value)(moved)
    return value


def _wrap_merge_expanded_output_batches(GPUWorker) -> None:
    """AROUND-wrap the grouped-output merge to carry conditions dim-0 concatenated."""
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
    """Concat each conditions field dim-0 across per-output batches onto ``merged``."""
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
            elif name == "image_latent_sizes":
                # Expanded outputs share one prompt and therefore one source grid.
                merged_list.append(tensors[0])
            else:
                merged_list.append(torch.cat(tensors, dim=0))
        setattr(merged, name, merged_list)


def _wrap_result_common(DiffGenerator) -> None:
    """AROUND-wrap ``DiffGenerator._result_common`` to add per-output embed slices."""
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
            val = getattr(output_batch, name, None)
            common[name] = val if name == "image_latent_sizes" else _slice_embed_list(val, idx)
        return common

    setattr(_result_common, _RESULT_COMMON_SENTINEL, True)
    DiffGenerator._result_common = staticmethod(_result_common)


def _slice_embed_list(embed_list, idx: int):
    """Slice the idx-th sample out of a per-encoder ``list[Tensor]`` field."""
    if embed_list is None:
        return None
    return [t[idx : idx + 1] if t is not None else None for t in embed_list]
