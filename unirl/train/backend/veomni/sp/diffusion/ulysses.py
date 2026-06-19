"""Ulysses SP for diffusers transformers.

Two attention-SP mechanisms (auto-detected per model), plus a small per-model
boundary spec (slice the sequence-carrying inputs + RoPE in, gather hidden out):

* **dispatch-patch** (newer models that route attention through
  ``dispatch_attention_fn`` — qwen-image / flux2 / wan): wrap that kernel call
  with the Ulysses all-to-all (gather seq / scatter heads -> attention ->
  scatter seq / gather heads). Model-agnostic — the model's own processor still
  does projection / QK-norm / RoPE / stream-concat on the sliced streams, and
  full (non-causal) attention is order-invariant over the gathered set, so the
  joint all-to-all is correct (RoPE is applied before the kernel). Cross-attention
  (Wan text branch: sliced image query vs full text K/V) is detected by unequal
  q/k seq length and skipped.

* **processor-injection** (older models whose processor calls SDPA directly,
  e.g. SD3's ``JointAttnProcessor2_0``): replace the attention processor with
  :class:`SPAttentionProcessor` (port of mmrl), which does the per-stream
  all-to-all itself.

Built on VeOmni primitives + the folded mesh (no sp_size grad compensation;
docs/usp-derisk/sp_fsdp.py). v1 requires each stream's sequence divisible by
sp_size (no padding — full attention can't tolerate unmasked padding).
Validated: qwen-image (dispatch) relerr 2e-7.
"""

from __future__ import annotations

import logging
import sys
from typing import Any, Callable, Dict, NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

logger = logging.getLogger(__name__)


class _SP(NamedTuple):
    """Named VeOmni SP handles (see :func:`_sp`) -- attribute access instead of
    positional unpacking, so call sites read ``sp.slice_input_tensor`` not ``_, _, _``."""

    get_parallel_state: Callable
    slice_input_tensor: Callable
    gather_outputs: Callable
    gather_seq_scatter_heads: Callable
    gather_heads_scatter_seq: Callable


def _sp() -> _SP:
    """Lazy VeOmni handles (after the selective-import shim is installed)."""
    from unirl.train.backend.veomni import _compat

    _compat.ensure_installed()
    from veomni.distributed.parallel_state import get_parallel_state
    from veomni.distributed.sequence_parallel import gather_outputs, slice_input_tensor
    from veomni.distributed.sequence_parallel.ulysses import (
        gather_heads_scatter_seq,
        gather_seq_scatter_heads,
    )

    return _SP(
        get_parallel_state, slice_input_tensor, gather_outputs, gather_seq_scatter_heads, gather_heads_scatter_seq
    )


# ---------------------------------------------------------------------------
# Mechanism 1: Ulysses all-to-all around dispatch_attention_fn (newer models)
# ---------------------------------------------------------------------------


def _reject_attn_mask(args: Any, kwargs: Any) -> None:
    """diffusion SP v1 has no masked path; refuse if diffusers built an attn mask."""
    if kwargs.get("attn_mask", args[0] if args else None) is not None:
        raise NotImplementedError(
            "diffusion SP v1 does not support an attention mask under Ulysses; "
            "ensure each stream length is divisible by sp_size so no mask is built."
        )


def _make_ulysses_dispatch(orig_dispatch: Callable) -> Callable:
    import functools

    sp = _sp()
    get_parallel_state = sp.get_parallel_state
    gather_seq_scatter_heads = sp.gather_seq_scatter_heads
    gather_heads_scatter_seq = sp.gather_heads_scatter_seq

    @functools.wraps(orig_dispatch)
    def ulysses_dispatch(query: Tensor, key: Tensor, value: Tensor, *args: Any, **kwargs: Any):
        ps = get_parallel_state()
        # Intervene only for SP self-attention. Disabled -> passthrough. Cross-attention
        # (e.g. Wan text branch: sliced image Q vs full text K/V => unequal seq lengths)
        # attends locally per rank and also passes through -- the image Q is already
        # sliced to local by the boundary hook, so it must NOT be all-to-all'd.
        if not ps.ulysses_enabled or query.shape[1] != key.shape[1]:
            return orig_dispatch(query, key, value, *args, **kwargs)
        _reject_attn_mask(args, kwargs)  # v1: no masked path under Ulysses

        # Self-attention joint all-to-all: each rank gathers the full sequence and keeps
        # its slice of heads, attends over the full (order-invariant) set, then inverts.
        g = ps.sp_group
        query = gather_seq_scatter_heads(query, seq_dim=1, head_dim=2, group=g)
        key = gather_seq_scatter_heads(key, seq_dim=1, head_dim=2, group=g)
        value = gather_seq_scatter_heads(value, seq_dim=1, head_dim=2, group=g)
        out = orig_dispatch(query, key, value, *args, **kwargs)
        return gather_heads_scatter_seq(out, head_dim=2, seq_dim=1, group=g)

    ulysses_dispatch._unirl_sp_patched = True
    return ulysses_dispatch


def _patch_attention_dispatch(model: nn.Module) -> bool:
    """Wrap the model module's ``dispatch_attention_fn`` with the Ulysses all-to-all.

    Newer diffusers transformers route EVERY attention call through a single
    module-level ``dispatch_attention_fn(q, k, v, ...)`` indirection, so replacing
    that one global intercepts all attention in the model without touching any
    submodule. Returns False if the model's module has no such global (older models,
    e.g. SD3 -> caller falls back to processor injection). Idempotent via the
    ``_unirl_sp_patched`` flag on the wrapper.
    """
    module = sys.modules.get(type(model).__module__)
    if module is None or not hasattr(module, "dispatch_attention_fn"):
        return False
    if not getattr(module.dispatch_attention_fn, "_unirl_sp_patched", False):
        module.dispatch_attention_fn = _make_ulysses_dispatch(module.dispatch_attention_fn)
        logger.info("diffusion SP: patched %s.dispatch_attention_fn with Ulysses all-to-all", module.__name__)
    return True


# ---------------------------------------------------------------------------
# Mechanism 2: SP attention processor (older models, e.g. SD3 JointAttnProcessor2_0)
# ---------------------------------------------------------------------------


def apply_rotary_emb(x: Tensor, freqs_cis: tuple[Tensor, Tensor]) -> Tensor:
    """RoPE for diffusers Q/K, ``(B, S, H, D_h)``. Flux: cos/sin ``(S, D_h)`` (2D);
    Wan: ``(1, S, 1, D_h)`` (4D interleaved). Ported from mmrl."""
    cos, sin = freqs_cis
    cos, sin = cos.to(x.device), sin.to(x.device)
    if cos.ndim == 2:
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
        x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(-2)
        return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
    x1, x2 = x.to(torch.float64).unflatten(-1, (-1, 2)).unbind(-1)
    fc, fs = cos[..., 0::2].to(torch.float64), sin[..., 1::2].to(torch.float64)
    out = torch.empty_like(x, dtype=torch.float64)
    out[..., 0::2] = x1 * fc - x2 * fs
    out[..., 1::2] = x1 * fs + x2 * fc
    return out.to(x.dtype)


def _sdpa(q: Tensor, k: Tensor, v: Tensor, scale: float, attn_mask: Tensor | None) -> Tensor:
    o = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=attn_mask, scale=scale
    )
    return o.transpose(1, 2)


class SPAttentionProcessor:
    """SP-aware diffusers attention processor (Ulysses), port of mmrl.

    dual-stream joint (``add_q_proj``): all-to-all each [encoder, image] stream,
    re-cat in the original processor's order, attend, inverse; cross-attention:
    skip all-to-all; self-attention: full all-to-all.
    """

    def __init__(self, sp_group: Any, original_processor: Any = None):
        self.sp_group = sp_group
        self.original_processor = original_processor
        sp = _sp()
        self._gather_seq, self._gather_heads = sp.gather_seq_scatter_heads, sp.gather_heads_scatter_seq

    def _a2a_sh(self, x):  # scatter heads, gather seq
        return self._gather_seq(x, seq_dim=1, head_dim=2, group=self.sp_group)

    def _a2a_hs(self, x):  # scatter seq, gather heads
        return self._gather_heads(x, head_dim=2, seq_dim=1, group=self.sp_group)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        image_rotary_emb=None,
        rotary_emb=None,
        **kwargs,
    ):
        if image_rotary_emb is None and rotary_emb is not None:
            image_rotary_emb = rotary_emb
        is_cross = encoder_hidden_states is not None
        has_added_kv = hasattr(attn, "add_q_proj") and attn.add_q_proj is not None and is_cross
        b = hidden_states.shape[0]

        kv_in = hidden_states if (has_added_kv or not is_cross) else encoder_hidden_states
        query, key, value, inner = self._project_qkv(attn, hidden_states, kv_in, b)
        if has_added_kv:
            query, key, value = self._prepend_added_kv(attn, encoder_hidden_states, query, key, value, b, inner)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)
        scale = attn.scale if hasattr(attn, "scale") else (inner // attn.heads) ** -0.5

        if has_added_kv:
            return self._dual_stream_attend(
                attn, query, key, value, encoder_hidden_states.shape[1], scale, attention_mask, b, inner
            )
        return self._single_stream_attend(attn, query, key, value, is_cross, scale, attention_mask, b, inner)

    def _project_qkv(self, attn, hidden_states, kv_in, b):
        """Project Q/K/V and apply QK-norm before or after the head-reshape, matching the
        processor's ``normalized_shape`` (``(inner,)`` -> before view, ``(hd,)`` -> after)."""
        query = attn.to_q(hidden_states)
        key, value = attn.to_k(kv_in), attn.to_v(kv_in)
        inner = query.shape[-1]
        hd = inner // attn.heads
        nq = getattr(attn, "norm_q", None) is not None
        nk = getattr(attn, "norm_k", None) is not None
        norm_before_view = bool(
            nq and hasattr(attn.norm_q, "normalized_shape") and attn.norm_q.normalized_shape == (inner,)
        )
        if norm_before_view:
            if nq:
                query = attn.norm_q(query)
            if nk:
                key = attn.norm_k(key)
        query = query.view(b, -1, attn.heads, hd)
        key = key.view(b, -1, attn.heads, hd)
        value = value.view(b, -1, attn.heads, hd)
        if not norm_before_view:
            if nq:
                query = attn.norm_q(query)
            if nk:
                key = attn.norm_k(key)
        return query, key, value, inner

    def _prepend_added_kv(self, attn, encoder_hidden_states, query, key, value, b, inner):
        """Joint attention: project the encoder stream's added Q/K/V (+QK-norm) and prepend it."""
        hd = inner // attn.heads
        enc_q = attn.add_q_proj(encoder_hidden_states).view(b, -1, attn.heads, hd)
        enc_k = attn.add_k_proj(encoder_hidden_states).view(b, -1, attn.heads, hd)
        enc_v = attn.add_v_proj(encoder_hidden_states).view(b, -1, attn.heads, hd)
        if getattr(attn, "norm_added_q", None) is not None:
            enc_q = attn.norm_added_q(enc_q)
        if getattr(attn, "norm_added_k", None) is not None:
            enc_k = attn.norm_added_k(enc_k)
        return torch.cat([enc_q, query], 1), torch.cat([enc_k, key], 1), torch.cat([enc_v, value], 1)

    def _dual_stream_attend(self, attn, query, key, value, enc_len, scale, attention_mask, b, inner):
        """Joint streams [encoder|image] (or [image|encoder] for JointAttnProcessor2_0):
        split, all-to-all each stream, attend over the joint set in the processor's order,
        invert, split back into image/encoder outputs."""
        encoder_first = type(self.original_processor).__name__ != "JointAttnProcessor2_0"
        enc_q, img_q = query[:, :enc_len], query[:, enc_len:]
        enc_k, img_k = key[:, :enc_len], key[:, enc_len:]
        enc_v, img_v = value[:, :enc_len], value[:, enc_len:]
        enc_q, enc_k, enc_v = self._a2a_sh(enc_q), self._a2a_sh(enc_k), self._a2a_sh(enc_v)
        img_q, img_k, img_v = self._a2a_sh(img_q), self._a2a_sh(img_k), self._a2a_sh(img_v)
        if encoder_first:
            q, k, v = torch.cat([enc_q, img_q], 1), torch.cat([enc_k, img_k], 1), torch.cat([enc_v, img_v], 1)
        else:
            q, k, v = torch.cat([img_q, enc_q], 1), torch.cat([img_k, enc_k], 1), torch.cat([img_v, enc_v], 1)
        out = _sdpa(q, k, v, scale, attention_mask)
        full_enc, full_img = enc_q.shape[1], img_q.shape[1]
        enc_out, img_out = (
            (out[:, :full_enc], out[:, full_enc:]) if encoder_first else (out[:, full_img:], out[:, :full_img])
        )
        enc_out, img_out = self._a2a_hs(enc_out), self._a2a_hs(img_out)
        hs = attn.to_out[0](img_out.reshape(b, -1, inner))
        if len(attn.to_out) > 1:
            hs = attn.to_out[1](hs)
        enc = enc_out.reshape(b, -1, inner)
        if not getattr(attn, "context_pre_only", False):
            enc = attn.to_add_out(enc)
        return hs, enc

    def _single_stream_attend(self, attn, query, key, value, is_cross, scale, attention_mask, b, inner):
        """Cross-attention: attend locally (no all-to-all). Self-attention: full all-to-all."""
        if is_cross:
            out = _sdpa(query, key, value, scale, attention_mask)
        else:
            query, key, value = self._a2a_sh(query), self._a2a_sh(key), self._a2a_sh(value)
            out = self._a2a_hs(_sdpa(query, key, value, scale, attention_mask))
        hs = out.reshape(b, -1, inner)
        if getattr(attn, "to_out", None) is not None:
            hs = attn.to_out[0](hs)
            if len(attn.to_out) > 1:
                hs = attn.to_out[1](hs)
        return hs


def inject_sp_processors(model: nn.Module, sp_group: Any) -> int:
    import torch.distributed as dist

    sp_size = dist.get_world_size(sp_group)
    count = 0
    for name, module in model.named_modules():
        if hasattr(module, "set_processor") and hasattr(module, "get_processor"):
            heads = getattr(module, "heads", None)
            if heads is not None and heads % sp_size != 0:
                raise ValueError(
                    f"SP requires num_heads % sp_size == 0, but '{name}' has {heads} heads, sp_size={sp_size}."
                )
            module.set_processor(SPAttentionProcessor(sp_group, original_processor=module.get_processor()))
            count += 1
    logger.info("diffusion SP: injected SPAttentionProcessor into %d attention modules", count)
    return count


# ---------------------------------------------------------------------------
# Per-model boundary hooks: slice streams + RoPE in, gather hidden out.
# Three models use the BLOCK-level pattern (slice at blocks[0], gather at norm_out)
# via _install_boundary_hooks; flux2 uses a MODEL-level pattern (slice both streams at
# model input, gather image at model output) because its dual->single + text-strip
# layout has no single block boundary -- kept inline in _wrap_flux2.
# ---------------------------------------------------------------------------


def _assert_seq_divisible(length: int, sp_size: int, what: str) -> None:
    """Fail fast if a to-be-sharded stream length is not a multiple of ``sp_size``.

    Diffusion SP v1 shards the sequence with no padded/masked attention path:
    ``slice_input_tensor`` zero-pads a non-divisible sequence on the right, and full
    (non-causal) joint attention then attends to that pad, silently corrupting every
    real token's output (the post-gather truncation at ``norm_out`` cannot undo it).
    Refuse instead of corrupting. The *text/encoder* stream is the usual offender
    (SD3's fixed 77+256=333; qwen-image's variable prompt length); keeping it
    un-sharded is the planned latent-only-sharding follow-up.
    """
    if length % sp_size != 0:
        raise ValueError(
            f"diffusion SP: {what} length {length} is not divisible by sp_size={sp_size}. "
            "Ulysses v1 has no padded/masked path, so a non-divisible stream would be "
            "zero-padded and corrupt full attention. Use a resolution / max_sequence_length "
            "whose per-stream token count is a multiple of sp_size, or set sp_size=1 for this model."
        )


def _install_boundary_hooks(
    model, sp_group, blocks_attr, norm_out_attr, rope_hook=None, rope_attr="pos_embed", slice_encoder=True
):
    """Slice the image (+ optionally text) stream at the first block, gather the
    image stream at the output norm. ``slice_encoder=False`` keeps the text full
    (Wan cross-attention). Handles kwargs (qwen/sd3) and positional (wan) calls.
    """
    sp = _sp()
    get_parallel_state, slice_input_tensor, gather_outputs = (
        sp.get_parallel_state,
        sp.slice_input_tensor,
        sp.gather_outputs,
    )
    # block0_pre records the pre-slice image length; norm_out_pre reads it to drop SP
    # divisibility padding after the gather.
    state: Dict[str, int] = {}
    import torch.distributed as dist

    sp_size = dist.get_world_size(sp_group)

    def block0_pre(_m, args, kwargs):
        if not get_parallel_state().ulysses_enabled:
            return None
        new_args = list(args)
        if "hidden_states" in kwargs:
            _assert_seq_divisible(kwargs["hidden_states"].shape[1], sp_size, "image stream")
            state["img_len"] = kwargs["hidden_states"].shape[1]
            kwargs["hidden_states"] = slice_input_tensor(kwargs["hidden_states"], dim=1, group=sp_group)
        elif new_args:
            _assert_seq_divisible(new_args[0].shape[1], sp_size, "image stream")
            state["img_len"] = new_args[0].shape[1]
            new_args[0] = slice_input_tensor(new_args[0], dim=1, group=sp_group)
        if slice_encoder:
            if "encoder_hidden_states" in kwargs:
                _assert_seq_divisible(kwargs["encoder_hidden_states"].shape[1], sp_size, "text/encoder stream")
                kwargs["encoder_hidden_states"] = slice_input_tensor(
                    kwargs["encoder_hidden_states"], dim=1, group=sp_group
                )
            elif len(new_args) >= 2:
                _assert_seq_divisible(new_args[1].shape[1], sp_size, "text/encoder stream")
                new_args[1] = slice_input_tensor(new_args[1], dim=1, group=sp_group)
        return tuple(new_args), kwargs

    def norm_out_pre(_m, args, kwargs):
        if not get_parallel_state().ulysses_enabled:
            return None
        hs = gather_outputs(args[0], gather_dim=1, group=sp_group)
        tl = state.get("img_len")
        if tl is not None and hs.shape[1] > tl:
            hs = hs[:, :tl]
        return (hs, *args[1:]), kwargs

    getattr(model, blocks_attr)[0].register_forward_pre_hook(block0_pre, with_kwargs=True)
    getattr(model, norm_out_attr).register_forward_pre_hook(norm_out_pre, with_kwargs=True)
    if rope_hook is not None:
        getattr(model, rope_attr).register_forward_hook(rope_hook)


def _make_rope_slice_hook(sp_group, dim: int):
    """Slice RoPE freqs across SP ranks (each rank keeps its stream's slice). Handles a
    single freqs tensor or a (cos, sin)/(vid, txt) tuple. pos_embed runs full (before
    slicing), so the freqs arriving here are full-length."""
    sp = _sp()
    get_parallel_state, slice_input_tensor = sp.get_parallel_state, sp.slice_input_tensor

    def hook(_m, _inp, out):
        if not get_parallel_state().ulysses_enabled:
            return out
        if isinstance(out, (tuple, list)):
            return type(out)(slice_input_tensor(t, dim=dim, group=sp_group) for t in out)
        return slice_input_tensor(out, dim=dim, group=sp_group)

    return hook


FORWARD_WRAPPERS: Dict[str, Callable[[nn.Module, Any], None]] = {}


def register(class_name: str) -> Callable[[Callable], Callable]:
    """Register a per-model boundary wrapper under its diffusers class name.

    Used as a decorator in ``models/<name>.py``; the populated registry is consumed by
    :func:`apply_diffusion_sequence_parallelism`, which MRO-walks it by class name.
    """

    def deco(fn: Callable) -> Callable:
        FORWARD_WRAPPERS[class_name] = fn
        return fn

    return deco


def apply_diffusion_sequence_parallelism(model: nn.Module, sp_size: int) -> None:
    """Attention SP (dispatch-patch or processor-injection, auto-detected) + boundary hooks."""
    sp_group = _sp().get_parallel_state().sp_group

    if not _patch_attention_dispatch(model):  # newer models
        inject_sp_processors(model, sp_group)  # older models (SD3)

    # Walk the MRO, not just type(model).__name__: after veomni_parallelize the
    # instance's class is a dynamic FSDP2 subclass (e.g. FSDPSD3Transformer2DModel)
    # whose base is the real transformer class, so the bare-name registry would
    # miss. The MRO still contains the original (e.g. SD3Transformer2DModel).
    cls = type(model).__name__
    wrapper = next((FORWARD_WRAPPERS[k.__name__] for k in type(model).__mro__ if k.__name__ in FORWARD_WRAPPERS), None)
    if wrapper is None:
        raise NotImplementedError(
            f"diffusion SP: attention SP wired, but no boundary spec for {cls} "
            f"(MRO: {[k.__name__ for k in type(model).__mro__]}). "
            f"Add a models/<name>.py with @register(...) under unirl.train.backend.veomni.sp.diffusion."
        )
    wrapper(model, sp_group)
    logger.info("diffusion SP installed for %s (sp_size=%d)", cls, sp_size)


def is_diffusers_transformer(model: nn.Module) -> bool:
    return any(hasattr(m, "set_processor") and hasattr(m, "get_processor") for m in model.modules())
