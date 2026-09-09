"""SDPA-based ``flash_attn_varlen_func`` drop-in for the vendored BAGEL modeling."""

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
    """Varlen scaled-dot-product attention over packed sequences (flash-attn API)."""
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
        qi = q[q_off : q_off + lq]
        ki = k[k_off : k_off + lk]
        vi = v[k_off : k_off + lk]

        if n_rep > 1:
            ki = ki.repeat_interleave(n_rep, dim=1)
            vi = vi.repeat_interleave(n_rep, dim=1)

        qh = qi.transpose(0, 1).unsqueeze(0)
        kh = ki.transpose(0, 1).unsqueeze(0)
        vh = vi.transpose(0, 1).unsqueeze(0)

        attn_mask = None
        if causal:
            # Use bottom-right causal masking so decode rows see all prior keys.
            qi_idx = torch.arange(lq, device=q.device).unsqueeze(1)
            ki_idx = torch.arange(lk, device=q.device).unsqueeze(0)
            attn_mask = ki_idx <= (qi_idx + (lk - lq))

        oi = scaled_dot_product_attention(
            qh,
            kh,
            vh,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            scale=softmax_scale,
        )
        out[q_off : q_off + lq] = oi.squeeze(0).transpose(0, 1)
        q_off += lq
        k_off += lk

    return out


__all__ = ["flash_attn_varlen_func"]
