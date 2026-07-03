"""SDPA-based ``flash_attn_varlen_func`` drop-in for the vendored BAGEL modeling.

The vendored BAGEL transformer (``vendor/modeling/bagel/{qwen2_navit,siglip_navit}.py``)
calls ``flash_attn_varlen_func`` in its GENERATION / inference attention path
(``PackedAttention(MoT).forward_inference`` and the SigLIP vision encoder). The
flash-attn build that exports that symbol is not always installed — e.g. an sglang
stack that ships ``flash-attn-4`` (whose API differs and has no
``flash_attn_varlen_func``). This module reimplements the exact varlen contract
those call sites rely on on top of
:func:`torch.nn.functional.scaled_dot_product_attention`, so BAGEL runs with **no
flash-attn dependency**. (The training / replay attention path already uses SDPA /
``flex_attention`` directly and does NOT route through this function.)

Only the subset of the flash-attn signature the BAGEL call sites use is implemented:
packed varlen q/k/v indexed by ``cu_seqlens``, grouped-query attention (q has more
heads than k/v), and ``causal`` with **bottom-right** alignment — so a KV-cache
decode step (``Lq=1, Lk=N``) correctly attends to all N keys.
``F.sdpa(is_causal=True)`` top-left-aligns and would be wrong for ``Lq != Lk``, so
the causal mask is always built explicitly.

Correctness-first: the per-sequence Python loop is slower than a fused flash kernel
for autoregressive generation. A batched block-diagonal mask is a possible follow-up.
"""

from __future__ import annotations

from typing import List, Optional

import torch
from torch.nn.functional import scaled_dot_product_attention


def _seqlens(cu_seqlens: torch.Tensor) -> List[int]:
    """Per-sequence lengths from cumulative offsets ``[0, l0, l0+l1, ...]``."""
    cu = cu_seqlens.to(torch.int64).tolist()
    return [cu[i + 1] - cu[i] for i in range(len(cu) - 1)]


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    **_: object,
) -> torch.Tensor:
    """Varlen scaled-dot-product attention over packed sequences (flash-attn API).

    Args mirror ``flash_attn.flash_attn_varlen_func`` for the subset BAGEL uses:

    - ``q``: ``(total_q, num_heads, head_dim)`` — all query tokens, concatenated.
    - ``k`` / ``v``: ``(total_k, num_kv_heads, head_dim)`` — all key / value tokens.
      ``num_kv_heads`` may be ``< num_heads`` (GQA); KV heads are expanded so query
      head ``h`` attends KV head ``h // (num_heads // num_kv_heads)``.
    - ``cu_seqlens_q`` / ``cu_seqlens_k``: ``(batch + 1,)`` cumulative token offsets
      segmenting the packed q / k|v into per-sequence blocks.
    - ``causal``: bottom-right-aligned causal mask within each sequence.
    - ``softmax_scale``: attention scale; ``None`` -> SDPA default ``1/sqrt(head_dim)``
      (matches flash-attn's default). ``max_seqlen_*`` are accepted for API parity.

    Returns ``(total_q, num_heads, head_dim)`` in ``q``'s dtype — the layout the call
    sites then ``.reshape(-1, hidden)``.
    """
    num_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    head_dim = q.shape[2]
    n_rep = num_heads // num_kv_heads

    q_lens = _seqlens(cu_seqlens_q)
    k_lens = _seqlens(cu_seqlens_k)

    out = q.new_empty((q.shape[0], num_heads, head_dim))
    q_off = 0
    k_off = 0
    for lq, lk in zip(q_lens, k_lens):
        qi = q[q_off : q_off + lq]  # (lq, num_heads, head_dim)
        ki = k[k_off : k_off + lk]  # (lk, num_kv_heads, head_dim)
        vi = v[k_off : k_off + lk]

        if n_rep > 1:  # GQA: expand KV heads to match query heads (contiguous grouping)
            ki = ki.repeat_interleave(n_rep, dim=1)  # -> (lk, num_heads, head_dim)
            vi = vi.repeat_interleave(n_rep, dim=1)

        # SDPA expects (batch, num_heads, seq, head_dim).
        qh = qi.transpose(0, 1).unsqueeze(0)  # (1, num_heads, lq, head_dim)
        kh = ki.transpose(0, 1).unsqueeze(0)
        vh = vi.transpose(0, 1).unsqueeze(0)

        attn_mask = None
        if causal:
            # Bottom-right alignment: query i (0..lq) attends key j (0..lk) iff
            # j <= i + (lk - lq). lq==lk -> usual lower-triangular; decode (lq=1,lk=N)
            # -> all N keys visible. Every row keeps >=1 key (lk>=lq here), so no
            # all-masked row / NaN. Bool mask: True = attend.
            qi_idx = torch.arange(lq, device=q.device).unsqueeze(1)  # (lq, 1)
            ki_idx = torch.arange(lk, device=q.device).unsqueeze(0)  # (1, lk)
            attn_mask = ki_idx <= (qi_idx + (lk - lq))

        oi = scaled_dot_product_attention(
            qh,
            kh,
            vh,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            scale=softmax_scale,
        )  # (1, num_heads, lq, head_dim)
        out[q_off : q_off + lq] = oi.squeeze(0).transpose(0, 1)  # (lq, num_heads, head_dim)
        q_off += lq
        k_off += lk

    return out


__all__ = ["flash_attn_varlen_func"]
