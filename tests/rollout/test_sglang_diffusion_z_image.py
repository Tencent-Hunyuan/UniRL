"""CPU-only guards for the ``sglang_diffusion`` Z-Image wiring.

Covers the pure-logic pieces of the Z-Image v2 rollout-engine port — no SGLang
import, no GPU:

* ``patch_weights_updater._apply_fused_param_mapping`` — separate diffusers
  ``to_q/to_k/to_v`` + ``w1/w3`` tensors must land in Z-Image's FUSED
  ``to_qkv`` / ``w13`` shards (the flat-reward bug otherwise).
* ``patch_conditions._ensure_batched_embed_list`` — un-batched ``[seq, hidden]``
  captions (single-prompt encode) get a batch dim; batched / pooled / masks are
  left as-is.
* ``adapters.z_image`` — the 6-D -> 5-D frame squeeze and the caption-mask
  backfill (so cross-padded zero rows are never denoised as real tokens).
"""

from types import SimpleNamespace

import pytest
import torch

from unirl.rollout.engine.sglang_diffusion._patches.patch_conditions import (
    _ensure_batched_embed_list,
)
from unirl.rollout.engine.sglang_diffusion._patches.patch_weights_updater import (
    _apply_fused_param_mapping,
    _resolve_param_names_mapping,
)
from unirl.rollout.engine.sglang_diffusion.adapters import registered_adapters
from unirl.rollout.engine.sglang_diffusion.adapters.z_image import (
    ZImageAdapter,
    _backfill_caption_mask,
    _to_image_form_trajectory,
)
from unirl.types.conditions.text import TextEmbedCondition

# The Z-Image fused-projection subset of ZImageArchConfig.param_names_mapping
# (weight-only entries; the lora_*/weight_scale_inv variants share the same shape).
_ZIMAGE_MAPPING = {
    r"(.*)\.attention\.to_q\.weight$": (r"\1.attention.to_qkv.weight", 0, 3),
    r"(.*)\.attention\.to_k\.weight$": (r"\1.attention.to_qkv.weight", 1, 3),
    r"(.*)\.attention\.to_v\.weight$": (r"\1.attention.to_qkv.weight", 2, 3),
    r"(.*)\.feed_forward\.w1\.weight$": (r"\1.feed_forward.w13.weight", 0, 2),
    r"(.*)\.feed_forward\.w3\.weight$": (r"\1.feed_forward.w13.weight", 1, 2),
}


class _FakeFusedModule:
    """Minimal stand-in for the fused Z-Image transformer module.

    ``_apply_fused_param_mapping`` reads ``type(module).param_names_mapping`` and
    ``dict(module.named_parameters())`` only; plain tensors (no ``weight_loader``)
    exercise the equal-split fallback in ``_write_fused_shard`` — the exact path
    Z-Image base (MHA, equal q/k/v) takes at tp_size=1.
    """

    param_names_mapping = _ZIMAGE_MAPPING

    def __init__(self, params):
        self._params = params

    def named_parameters(self):
        return list(self._params.items())


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #


def test_z_image_adapter_registered():
    assert "z_image" in registered_adapters()


# --------------------------------------------------------------------------- #
# Fused weight mapping
# --------------------------------------------------------------------------- #


def test_fused_param_mapping_routes_qkv_and_w13_shards():
    d, h = 4, 6
    qkv = torch.zeros(3 * d, d)
    w13 = torch.zeros(2 * h, d)
    module = _FakeFusedModule(
        {
            "layers.0.attention.to_qkv.weight": qkv,
            "layers.0.feed_forward.w13.weight": w13,
        }
    )
    named = [
        ("layers.0.attention.to_q.weight", torch.full((d, d), 1.0)),
        ("layers.0.attention.to_k.weight", torch.full((d, d), 2.0)),
        ("layers.0.attention.to_v.weight", torch.full((d, d), 3.0)),
        ("layers.0.feed_forward.w1.weight", torch.full((h, d), 7.0)),
        ("layers.0.feed_forward.w3.weight", torch.full((h, d), 8.0)),
    ]

    leftover = _apply_fused_param_mapping(module, named)

    assert leftover == []  # every separate tensor was consumed into a fused shard
    assert torch.equal(qkv[0:d], torch.full((d, d), 1.0))  # q -> shard 0
    assert torch.equal(qkv[d : 2 * d], torch.full((d, d), 2.0))  # k -> shard 1
    assert torch.equal(qkv[2 * d : 3 * d], torch.full((d, d), 3.0))  # v -> shard 2
    assert torch.equal(w13[0:h], torch.full((h, d), 7.0))  # w1 -> shard 0
    assert torch.equal(w13[h : 2 * h], torch.full((h, d), 8.0))  # w3 -> shard 1


def test_fused_param_mapping_passes_through_exact_matches():
    # A name already present as a real param must NOT be fused — it falls through
    # to the exact-match loader untouched.
    p = torch.zeros(4, 4)
    module = _FakeFusedModule({"layers.0.attention.to_qkv.weight": p, "norm.weight": torch.zeros(4)})
    payload = [("norm.weight", torch.ones(4))]
    assert _apply_fused_param_mapping(module, payload) == payload


def test_resolve_param_names_mapping_absent_is_empty():
    class _Plain:
        pass

    assert _resolve_param_names_mapping(_Plain()) == {}
    # No mapping -> _apply_fused_param_mapping is a pure pass-through.
    payload = [("transformer_blocks.0.attn.to_q.weight", torch.ones(2, 2))]
    assert _apply_fused_param_mapping(_Plain(), payload) == payload


# --------------------------------------------------------------------------- #
# Un-batched embed normalization
# --------------------------------------------------------------------------- #


def test_ensure_batched_adds_dim_for_unbatched_caption():
    out = _ensure_batched_embed_list([torch.randn(37, 2560)])
    assert tuple(out[0].shape) == (1, 37, 2560)


def test_ensure_batched_noop_for_batched_and_none():
    batched = torch.randn(4, 37, 2560)
    out = _ensure_batched_embed_list([batched, None])
    assert out[0] is batched  # already [B, seq, h] -> untouched (same object)
    assert out[1] is None
    assert _ensure_batched_embed_list(None) is None  # non-list passthrough


# --------------------------------------------------------------------------- #
# Trajectory frame squeeze
# --------------------------------------------------------------------------- #


def test_trajectory_frame_axis_squeezed():
    t6 = torch.randn(2, 13, 16, 1, 8, 8)  # [B, T+1, C, F=1, H, W]
    assert tuple(_to_image_form_trajectory(t6).shape) == (2, 13, 16, 8, 8)


def test_trajectory_5d_passthrough():
    t5 = torch.randn(2, 13, 16, 8, 8)
    assert _to_image_form_trajectory(t5) is t5


def test_trajectory_nonsingleton_frame_raises():
    with pytest.raises(ValueError):
        _to_image_form_trajectory(torch.randn(2, 13, 16, 2, 8, 8))


# --------------------------------------------------------------------------- #
# Caption mask backfill
# --------------------------------------------------------------------------- #


def test_backfill_mask_from_zero_rows():
    emb = torch.zeros(2, 5, 4)
    emb[0, :3] = 1.0  # sample 0: 3 real tokens, 2 zero-pad
    emb[1, :5] = 1.0  # sample 1: 5 real tokens
    out = _backfill_caption_mask(TextEmbedCondition(embeds=emb, pooled=None, attn_mask=None))
    assert out.attn_mask.tolist() == [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]


def test_backfill_leaves_real_mask_untouched():
    emb = torch.zeros(2, 5, 4)
    emb[0, :3] = 1.0
    real = torch.ones(2, 5, dtype=torch.long)
    tc = TextEmbedCondition(embeds=emb, pooled=None, attn_mask=real)
    assert _backfill_caption_mask(tc) is tc  # already has a mask -> returned as-is


def test_backfill_none_passthrough():
    assert _backfill_caption_mask(None) is None


# --------------------------------------------------------------------------- #
# Adapter integration (CPU, fake RawResults)
# --------------------------------------------------------------------------- #


def _make_adapter():
    cfg = SimpleNamespace(populate_conditions=True, target_modules=None)
    mc = SimpleNamespace(
        pretrained_model_ckpt_path="Tongyi-MAI/Z-Image",
        shift=6.0,
        weight_sync_param_name_prefix="transformer.",
    )
    return ZImageAdapter(cfg, mc, strategy=None)


def _res(**kw):
    fields = {
        f: None
        for f in (
            "prompt_embeds",
            "pooled_prompt_embeds",
            "encoder_attention_mask",
            "negative_prompt_embeds",
            "neg_pooled_prompt_embeds",
            "negative_attention_mask",
        )
    }
    fields.update(kw)
    return SimpleNamespace(**fields)


def test_build_condition_backfills_mask_for_cross_padded_captions():
    # Two single-prompt encodes (pre-trimmed, no mask) of different lengths: the
    # generic fuse zero-pads to the max with no mask -> build_condition must recover
    # the validity mask so replay's _caption_list trims the pad.
    adapter = _make_adapter()
    r0 = _res(prompt_embeds=[torch.ones(1, 3, 4)])
    r1 = _res(prompt_embeds=[torch.ones(1, 5, 4)])
    out = adapter.build_condition([r0, r1])
    text = out["text"]
    assert tuple(text.embeds.shape) == (2, 5, 4)
    assert text.attn_mask.tolist() == [[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]
