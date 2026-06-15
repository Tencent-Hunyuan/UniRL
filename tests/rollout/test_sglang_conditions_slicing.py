"""Shape-contract guards for the SGLang ``populate_conditions`` slice/merge path.

Per-encoder pooled (``[B, hidden]``) and attention masks (``[B, seq]``) must be
sliced/merged per output like any batched field, while single-encoder token-level
captions (``[seq, hidden]``, e.g. Z-Image/Qwen3) get a batch dim added at ingestion.
A rank-only slicer corrupts SD3 pooled into ``[1, n_enc, hidden]`` and drops group
rows in the merge; these tests pin the correct shapes for both families.

CPU-only; imports the real helpers (no sglang import needed).
"""

import types

import torch

from unirl.rollout.engine.sglang._patches.patch_conditions import (
    _COND_FIELDS,
    _ensure_batched_embed_list,
    _merge_conditions,
    _slice_embed_list,
)
from unirl.rollout.engine.sglang._text_fusion import fuse_text_encoder_outputs
from unirl.rollout.engine.sglang.response import _build_text_conditions


def _ob(**fields):
    """An OutputBatch-like holder with every conditions field defaulting to None."""
    ns = types.SimpleNamespace(**{f: None for f in _COND_FIELDS})
    for k, v in fields.items():
        setattr(ns, k, v)
    return ns


def _slice_then_fuse(per_encoder, idx, *, token_embed):
    if token_embed:
        per_encoder = _ensure_batched_embed_list(per_encoder)
    return fuse_text_encoder_outputs(_slice_embed_list(per_encoder, idx))


def test_sd3_pooled_not_corrupted_by_slice():
    # Two CLIP encoders, pooled [B, hidden] (rank 2, batched). Must fuse to [1, 1024]
    # for one output — NOT [1, 2, 512].
    pooled = [torch.randn(1, 512), torch.randn(1, 512)]
    assert tuple(_slice_then_fuse(pooled, 0, token_embed=False).shape) == (1, 1024)


def test_sd3_prompt_embeds_unaffected():
    # Token embeds [B, seq, hidden] (rank 3): ensure is a no-op; fuse concatenates seq.
    emb = [torch.randn(1, 77, 1536), torch.randn(1, 256, 1536)]
    assert tuple(_slice_then_fuse(emb, 0, token_embed=True).shape) == (1, 333, 1536)


def test_zimage_unbatched_caption_gets_batch_dim():
    # One Qwen3 encoder, caption [seq, hidden] (rank 2, NO batch dim).
    cap = [torch.randn(37, 2560)]
    assert tuple(_slice_then_fuse(cap, 0, token_embed=True).shape) == (1, 37, 2560)


def test_merge_keeps_all_group_rows_for_pooled():
    # Grouped merge of SD3 pooled across 3 output-batches of 4 must keep all 12 rows.
    obs = [_ob(pooled_prompt_embeds=[torch.randn(4, 512), torch.randn(4, 512)]) for _ in range(3)]
    merged = _ob()
    _merge_conditions(merged, obs)
    assert tuple(merged.pooled_prompt_embeds[0].shape) == (12, 512)


def test_merge_then_slice_zimage_caption_roundtrip():
    # Each per-output caption normalized to [1, seq, h]; merge -> [N, seq, h]; slicing
    # one output returns that output's full caption [1, seq, h].
    obs = [_ob(prompt_embeds=_ensure_batched_embed_list([torch.randn(37, 2560)])) for _ in range(2)]
    merged = _ob()
    _merge_conditions(merged, obs)
    assert tuple(merged.prompt_embeds[0].shape) == (2, 37, 2560)
    assert tuple(_slice_embed_list(merged.prompt_embeds, 1)[0].shape) == (1, 37, 2560)


def test_zimage_mask_recovered_from_zero_rows():
    # 3 valid tokens zero-padded to 5; mask must be [1,1,1,0,0].
    emb = torch.zeros(1, 5, 4)
    emb[0, :3] = torch.randn(3, 4)
    res = _ob(prompt_embeds=[emb])
    text_cond, _ = _build_text_conditions([res], model_family="z_image")
    assert text_cond.attn_mask.tolist() == [[1, 1, 1, 0, 0]]


def test_sd3_builds_no_mask():
    res = _ob(prompt_embeds=[torch.randn(1, 333, 1536)])
    text_cond, _ = _build_text_conditions([res], model_family="sd3")
    assert text_cond.attn_mask is None
