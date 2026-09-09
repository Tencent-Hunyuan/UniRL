"""Ulysses SP for HF autoregressive causal-LMs (e.g. Qwen3)."""

from __future__ import annotations

import functools
import logging
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)

SP_ATTN_IMPL = "veomni_flash_attention_2_with_sp"
_SP_ATTN_IMPL_CANDIDATES = (
    ("is_flash_attn_4_available", "veomni_flash_attention_4_with_sp"),
    ("is_flash_attn_3_available", "veomni_flash_attention_3_with_sp"),
    ("is_flash_attn_2_available", SP_ATTN_IMPL),
)


def is_ar_causal_lm(model: nn.Module) -> bool:
    """HF causal-LM shape: a decoder (``.model``) + ``.lm_head`` + ``.config``."""
    return hasattr(model, "lm_head") and hasattr(model, "model") and hasattr(model, "config")


def apply_ar_sequence_parallelism(model: nn.Module, sp_size: int) -> None:
    """Route attention through VeOmni Ulysses + wrap the decoder forward."""
    from unirl.train.backend.veomni import _compat

    _compat.ensure_attention_patch_installed()
    sp_attn_impl = _select_sp_attn_impl()
    _install_b1_dense_attn_patch(sp_attn_impl)

    _set_attn_impl(model.config, sp_attn_impl)
    for m in model.modules():
        cfg = getattr(m, "config", None)
        if cfg is not None:
            _set_attn_impl(cfg, sp_attn_impl)

    _wrap_decoder_forward(model.model)
    logger.info(
        "AR SP installed: attn_implementation=%s + decoder slice/gather wrapper (sp_size=%d)",
        sp_attn_impl,
        sp_size,
    )


def _select_sp_attn_impl() -> str:
    """Choose the newest FlashAttention backend installed in this environment."""
    import transformers.utils as transformers_utils

    for availability_probe, implementation in _SP_ATTN_IMPL_CANDIDATES:
        probe = getattr(transformers_utils, availability_probe, None)
        if callable(probe) and probe():
            return implementation
    return SP_ATTN_IMPL


def _install_b1_dense_attn_patch(sp_attn_impl: str) -> None:
    """B=1 dense path — KERNEL HALF, pairing with :func:`_sp_b1_dense_forward` (the boundary half)."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    orig = ALL_ATTENTION_FUNCTIONS[sp_attn_impl]
    if getattr(orig, "_unirl_b1_dense", False):
        return

    _PACK_KEYS = (
        "position_ids",
        "cu_seq_lens_q",
        "cu_seq_lens_k",
        "max_length_q",
        "max_length_k",
        "max_seqlen_q",
        "max_seqlen_k",
    )

    @functools.wraps(orig)
    def _b1_dense(*args: Any, **kwargs: Any):
        query = args[1] if len(args) > 1 else kwargs.get("query")
        if query is not None and query.shape[0] == 1:
            for k in _PACK_KEYS:
                kwargs.pop(k, None)
        return orig(*args, **kwargs)

    _b1_dense._unirl_b1_dense = True
    ALL_ATTENTION_FUNCTIONS.register(sp_attn_impl, _b1_dense)


def _sp_b1_dense_forward(orig, args, kwargs, true_len, mask2d, ps, spg):
    """B=1 dense path — BOUNDARY HALF, pairing with :func:`_b1_dense` (the kernel half)."""
    from veomni.distributed.sequence_parallel import gather_outputs, slice_input_tensor

    input_ids = kwargs.get("input_ids")
    inputs_embeds = kwargs.get("inputs_embeds")
    batch = (inputs_embeds if inputs_embeds is not None else input_ids).shape[0]

    real_idx = mask2d[0].nonzero(as_tuple=False).flatten()
    real_start, real_end = int(real_idx[0].item()), int(real_idx[-1].item()) + 1
    real_len = real_end - real_start
    sp = max(1, int(getattr(ps, "ulysses_size", 1)))
    padded_len = ((real_len + sp - 1) // sp) * sp
    pad = padded_len - real_len
    ref = input_ids if input_ids is not None else inputs_embeds
    if input_ids is not None:
        seq = input_ids[:, real_start:real_end]
        if pad:
            seq = torch.cat([seq, seq.new_zeros((batch, pad))], dim=1)
        kwargs["input_ids"] = slice_input_tensor(seq, dim=1, group=spg)
    if inputs_embeds is not None:
        emb = inputs_embeds[:, real_start:real_end]
        if pad:
            emb = torch.cat([emb, emb.new_zeros((batch, pad, emb.shape[-1]))], dim=1)
        kwargs["inputs_embeds"] = slice_input_tensor(emb, dim=1, group=spg)
    global_pos = torch.arange(padded_len, device=ref.device).unsqueeze(0).expand(batch, padded_len).contiguous()
    kwargs["position_ids"] = slice_input_tensor(global_pos, dim=1, group=spg)
    kwargs["attention_mask"] = None
    kwargs.pop("cache_position", None)
    out = orig(*args, **kwargs)
    hidden = gather_outputs(out.last_hidden_state, gather_dim=1, group=spg)
    hidden = hidden[:, :real_len, :]
    padded = hidden.new_zeros((batch, true_len, hidden.shape[-1]))
    padded[:, real_start:real_end, :] = hidden
    out.last_hidden_state = padded
    return out


def _set_attn_impl(cfg: Any, sp_attn_impl: str) -> None:
    if hasattr(cfg, "_attn_implementation"):
        cfg._attn_implementation = sp_attn_impl
    get_text = getattr(cfg, "get_text_config", None)
    if callable(get_text):
        try:
            tcfg = get_text()
            if tcfg is not None and tcfg is not cfg and hasattr(tcfg, "_attn_implementation"):
                tcfg._attn_implementation = sp_attn_impl
        except Exception:  # noqa: BLE001 — best-effort; absence is fine
            pass


def _sp_plain_forward(orig, args, kwargs, true_len, position_ids, spg):
    """Plain slice-in / gather-out (no padding, or B>1)."""
    from veomni.distributed.sequence_parallel import gather_outputs, slice_input_tensor

    input_ids = kwargs.get("input_ids")
    inputs_embeds = kwargs.get("inputs_embeds")
    if input_ids is not None:
        kwargs["input_ids"] = slice_input_tensor(input_ids, dim=1, group=spg)
    if inputs_embeds is not None:
        kwargs["inputs_embeds"] = slice_input_tensor(inputs_embeds, dim=1, group=spg)
    if position_ids is not None:
        kwargs["position_ids"] = slice_input_tensor(position_ids, dim=position_ids.dim() - 1, group=spg)
    kwargs.pop("cache_position", None)
    out = orig(*args, **kwargs)
    hidden = gather_outputs(out.last_hidden_state, gather_dim=1, group=spg)
    if hidden.shape[1] > true_len:
        hidden = hidden[:, :true_len, :]
    out.last_hidden_state = hidden
    return out


def _wrap_decoder_forward(decoder: nn.Module) -> None:
    """Wrap ``decoder.forward``: slice seq-dim inputs in, gather hidden out."""
    if getattr(decoder.forward, "_unirl_sp_wrapped", False):
        return

    orig = decoder.forward

    from unirl.train.backend.veomni import _compat

    _compat.ensure_installed()
    from veomni.distributed.parallel_state import get_parallel_state

    @functools.wraps(orig)
    def sp_forward(*args: Any, **kwargs: Any):
        ps = get_parallel_state()
        if not ps.ulysses_enabled:
            return orig(*args, **kwargs)
        spg = ps.sp_group

        input_ids = kwargs.get("input_ids")
        inputs_embeds = kwargs.get("inputs_embeds")
        position_ids = kwargs.get("position_ids")
        attention_mask = kwargs.get("attention_mask")

        if inputs_embeds is not None:
            true_len = inputs_embeds.shape[1]
        elif input_ids is not None:
            true_len = input_ids.shape[1]
        else:
            return orig(*args, **kwargs)

        batch = (inputs_embeds if inputs_embeds is not None else input_ids).shape[0]
        mask2d = attention_mask if (attention_mask is not None and attention_mask.dim() == 2) else None

        if batch == 1 and mask2d is not None and int(mask2d.sum().item()) < true_len:
            return _sp_b1_dense_forward(orig, args, kwargs, true_len, mask2d, ps, spg)
        return _sp_plain_forward(orig, args, kwargs, true_len, position_ids, spg)

    sp_forward._unirl_sp_wrapped = True
    decoder.forward = sp_forward


__all__ = ["apply_ar_sequence_parallelism", "is_ar_causal_lm", "SP_ATTN_IMPL"]
