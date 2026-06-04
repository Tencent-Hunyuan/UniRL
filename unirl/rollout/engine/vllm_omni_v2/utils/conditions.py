"""Condition assembly from the worker-side ``custom_output`` captures.

Pure helpers the adapters' ``build_condition`` steps dispatch to. Each reads
a capture dict the paired pipeline subclass wrote (plain runtime attrs on
``DiffusionOutput`` don't survive vllm-omni's IPC boundary — only the
dataclass-routed ``custom_output`` does) and concatenates per-request entries
into one typed condition. Each returns ``None`` when any output is missing
its capture — the *adapter* decides that's fatal and raises with the
modality-specific diagnosis.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch

from unirl.models.hunyuan_image3.conditions import (
    HunyuanImage3FusedMultimodalCondition,
)
from unirl.types.conditions import Condition
from unirl.types.conditions.text import TextEmbedCondition


def build_fused_mm_condition(
    diff_outputs: Sequence[Any],
) -> Optional[HunyuanImage3FusedMultimodalCondition]:
    """Concat per-request ``fused_mm_capture`` dicts into one fused condition.

    Reads ``custom_output["fused_mm_capture"]`` — written by
    ``RLHunyuanImage3Pipeline`` after intercepting
    ``prepare_inputs_for_generation``. For think_recaption mode different
    prompts produce different AR output lengths → different ``L`` per
    capture; right-pad shorter sequences to ``max_L`` (pad 0 for input_ids,
    False for masks, 0.0 for rope_cache) so the dim-0 concat works. t2i
    scope: the it2i ``cond_*`` fields stay unpopulated.
    """
    if not diff_outputs:
        return None
    captures = [(getattr(d, "custom_output", None) or {}).get("fused_mm_capture") for d in diff_outputs]
    if any(c is None for c in captures):
        return None

    sequence_lengths = [int(c["input_ids"].shape[-1]) for c in captures]
    max_L = max(sequence_lengths)

    def _pad_to(t: Any, target_L: int, dim: int = -1, value: Any = 0) -> Any:
        if t is None or not isinstance(t, torch.Tensor):
            return t
        cur_L = t.shape[dim]
        if cur_L >= target_L:
            return t
        pad_size = target_L - cur_L
        ndim = t.ndim
        pad_spec = [0] * (2 * ndim)
        actual_dim = dim if dim >= 0 else ndim + dim
        pad_idx = (ndim - 1 - actual_dim) * 2
        pad_spec[pad_idx + 1] = pad_size
        return torch.nn.functional.pad(t, pad_spec, value=value)

    def _pad_attn_mask(mask: Any, target_L: int) -> Any:
        """Pad attention_mask [N, 1, L, L] → [N, 1, target_L, target_L]."""
        if mask is None or not isinstance(mask, torch.Tensor):
            return mask
        if mask.shape[-1] >= target_L:
            return mask
        N, H, L, _ = mask.shape
        padded = torch.zeros(N, H, target_L, target_L, dtype=mask.dtype, device=mask.device)
        padded[:, :, :L, :L] = mask
        return padded

    padded_captures = []
    for c, L_i in zip(captures, sequence_lengths):
        if L_i == max_L:
            padded_captures.append(c)
        else:
            padded_captures.append(
                {
                    "input_ids": _pad_to(c["input_ids"], max_L, dim=-1, value=0),
                    "attention_mask": _pad_attn_mask(c.get("attention_mask"), max_L),
                    "position_ids": _pad_to(c.get("position_ids"), max_L, dim=-1, value=0),
                    "gen_image_mask": _pad_to(c.get("gen_image_mask"), max_L, dim=-1, value=False),
                    "gen_timestep_scatter_index": c.get("gen_timestep_scatter_index"),
                    "rope_cache": (
                        (
                            _pad_to(c["rope_cache"][0], max_L, dim=-2, value=0.0),
                            _pad_to(c["rope_cache"][1], max_L, dim=-2, value=0.0),
                        )
                        if c.get("rope_cache") is not None and isinstance(c["rope_cache"], tuple)
                        else c.get("rope_cache")
                    ),
                }
            )

    fused_dict: Dict[str, Any] = {
        "input_ids": torch.cat([c["input_ids"] for c in padded_captures], dim=0),
        "attention_mask": torch.cat([c["attention_mask"] for c in padded_captures], dim=0),
        "position_ids": torch.cat([c["position_ids"] for c in padded_captures], dim=0),
        "gen_image_mask": torch.cat([c["gen_image_mask"] for c in padded_captures], dim=0),
        "gen_timestep_scatter_index": torch.cat([c["gen_timestep_scatter_index"] for c in padded_captures], dim=0),
    }
    cos_parts = [c["rope_cache"][0] for c in padded_captures]
    sin_parts = [c["rope_cache"][1] for c in padded_captures]
    fused_dict["rope_cache"] = (
        torch.cat(cos_parts, dim=0),
        torch.cat(sin_parts, dim=0),
    )

    # ``from_dict`` skips optional fields when absent; cond_* fields stay
    # ``None`` for t2i (out of scope for the it2i extension).
    return HunyuanImage3FusedMultimodalCondition.from_dict(fused_dict)


def build_sd3_text_condition(diff_outputs: Sequence[Any]) -> Optional[TextEmbedCondition]:
    """Concat per-request SD3 ``text_capture`` dicts into one TextEmbedCondition.

    Written by ``RLStableDiffusion3Pipeline`` after intercepting
    ``encode_prompt``. All per-request encodes share the same ``L`` (T5
    padding to ``max_sequence_length`` is fixed), so a plain dim-0 concat
    suffices.
    """
    if not diff_outputs:
        return None
    captures = [(getattr(d, "custom_output", None) or {}).get("text_capture") for d in diff_outputs]
    if any(c is None for c in captures):
        return None

    return TextEmbedCondition(
        embeds=torch.cat([c["prompt_embeds"] for c in captures], dim=0),
        pooled=torch.cat([c["pooled_prompt_embeds"] for c in captures], dim=0),
        attn_mask=None,  # SD3 uses fixed-length T5 padding; no attn mask needed
    )


def build_hv15_conditions(diff_outputs: Sequence[Any]) -> Optional[Dict[str, Condition]]:
    """Unpack per-request HunyuanVideo-1.5 dual-stream text conditions.

    Written by ``RLHunyuanVideo15Pipeline`` after intercepting
    ``encode_prompt`` — 8 tensors from the dual text encoder (Qwen2.5-VL MLLM
    + ByT5 glyph), mapped to ``text_mllm`` / ``text_glyph`` (+ negatives).
    Returns the conditions *dict* (keys aligned with
    ``HunyuanVideo15Conditions.from_dict``), NOT the typed wrapper — the
    trainer runs ``from_dict(track.conditions)`` itself. ``None`` when any
    output is missing the capture or lacks the dual-stream embeds.
    """
    if not diff_outputs:
        return None

    captures = [(getattr(d, "custom_output", None) or {}).get("text_capture") for d in diff_outputs]
    if any(c is None for c in captures):
        return None

    def _cat_field(field_name: str) -> Optional[torch.Tensor]:
        tensors = [c[field_name] for c in captures if c.get(field_name) is not None]
        if not tensors:
            return None
        return torch.cat(tensors, dim=0)

    prompt_embeds = _cat_field("prompt_embeds")
    prompt_embeds_mask = _cat_field("prompt_embeds_mask")
    prompt_embeds_2 = _cat_field("prompt_embeds_2")
    prompt_embeds_mask_2 = _cat_field("prompt_embeds_mask_2")
    negative_prompt_embeds = _cat_field("negative_prompt_embeds")
    negative_prompt_embeds_mask = _cat_field("negative_prompt_embeds_mask")
    negative_prompt_embeds_2 = _cat_field("negative_prompt_embeds_2")
    negative_prompt_embeds_mask_2 = _cat_field("negative_prompt_embeds_mask_2")

    cond_dict: Dict[str, Condition] = {}
    if prompt_embeds is not None:
        cond_dict["text_mllm"] = TextEmbedCondition(embeds=prompt_embeds, pooled=None, attn_mask=prompt_embeds_mask)
    if prompt_embeds_2 is not None:
        cond_dict["text_glyph"] = TextEmbedCondition(
            embeds=prompt_embeds_2, pooled=None, attn_mask=prompt_embeds_mask_2
        )
    if negative_prompt_embeds is not None:
        cond_dict["negative_text_mllm"] = TextEmbedCondition(
            embeds=negative_prompt_embeds, pooled=None, attn_mask=negative_prompt_embeds_mask
        )
    if negative_prompt_embeds_2 is not None:
        cond_dict["negative_text_glyph"] = TextEmbedCondition(
            embeds=negative_prompt_embeds_2, pooled=None, attn_mask=negative_prompt_embeds_mask_2
        )

    if "text_mllm" not in cond_dict or "text_glyph" not in cond_dict:
        return None
    return cond_dict


def build_ar_fused_condition(per_request: Sequence[Sequence[Any]]) -> Optional[Any]:
    """AR fused condition for ARGRPO replay: per-sample prompt token ids.

    Each AR request's Stage-0 output carries ``prompt_token_ids`` (vLLM runs
    prompts per-request with no batch padding, so this is the sample's TRUE,
    un-padded prompt). Right-pad to ``[B, max_len]`` and carry each sample's
    true length in the dedicated 1D ``prompt_lengths`` [B] field (NOT
    ``attention_mask`` — that's typed 4D and its concat does a 4D unpack).
    The teacher-forced replay slices ``input_ids[b, :prompt_lengths[b]]``, so
    the right-pad never leaks. Returns ``None`` if no Stage-0 output carries
    prompt tokens.
    """
    rows: List[List[int]] = []
    for outputs in per_request:
        ids = None
        for out in outputs:
            if getattr(out, "stage_id", None) == 0:
                ids = getattr(out, "prompt_token_ids", None)
                break
        rows.append([int(t) for t in ids] if ids else [])

    if not any(rows):
        return None

    bsz = len(rows)
    max_len = max(len(r) for r in rows)
    input_ids = torch.zeros((bsz, max_len), dtype=torch.long)
    prompt_lengths = torch.zeros((bsz,), dtype=torch.long)
    for b, r in enumerate(rows):
        if r:
            input_ids[b, : len(r)] = torch.tensor(r, dtype=torch.long)
            prompt_lengths[b] = len(r)
    return HunyuanImage3FusedMultimodalCondition(input_ids=input_ids, prompt_lengths=prompt_lengths)


__all__ = [
    "build_ar_fused_condition",
    "build_fused_mm_condition",
    "build_hv15_conditions",
    "build_sd3_text_condition",
]
