"""Block-causal execution helpers for the diffusers WAN transformer.

The adapter keeps the original ``WanTransformer3DModel`` parameters and state
dict intact.  It only replaces the runtime forward path of each self-attention
layer, so a normal WAN2.1 checkpoint can initialize a causal generator without
checkpoint conversion.

Two execution modes are supported:

``causal full forward``
    The complete latent video is processed once with block-wise causal
    attention. Tokens inside one temporal block remain bidirectional.

``cached block forward``
    One temporal block is processed at a time. Detached K/V tensors from
    previously committed blocks form the history. Repeated denoising forwards
    read the history without mutating it; a separate ``commit_cache=True``
    forward stores the final generated block.

The cache is deliberately a plain Python object. It is neither a parameter nor
a persistent buffer and therefore never enters FSDP state dicts.
"""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


def _apply_rotary(
    hidden_states: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(hidden_states)


def _shifted_rope(rope: Any, hidden_states: torch.Tensor, *, frame_offset: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Diffusers ``WanRotaryPosEmbed.forward`` with a temporal offset."""
    _, _, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = rope.patch_size
    ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w
    split_sizes = [rope.t_dim, rope.h_dim, rope.w_dim]
    cos_t, cos_h, cos_w = rope.freqs_cos.split(split_sizes, dim=1)
    sin_t, sin_h, sin_w = rope.freqs_sin.split(split_sizes, dim=1)
    end = frame_offset + ppf
    if end > int(cos_t.shape[0]):
        raise ValueError(
            f"WAN causal RoPE needs temporal positions [{frame_offset}, {end}), "
            f"but rope_max_seq_len is only {cos_t.shape[0]}."
        )

    def expand(t: torch.Tensor, h: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        t = t[frame_offset:end].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        h = h[:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        w = w[:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
        return torch.cat([t, h, w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)

    return expand(cos_t, cos_h, cos_w), expand(sin_t, sin_h, sin_w)


def block_causal_attention_mask(
    *,
    num_frames: int,
    tokens_per_frame: int,
    frames_per_block: int,
    device: torch.device,
) -> torch.Tensor:
    """Return an SDPA bool mask: block-bidirectional, block-to-block causal."""
    if frames_per_block < 1:
        raise ValueError(f"frames_per_block must be >= 1; got {frames_per_block}.")
    frame_ids = torch.arange(num_frames, device=device).repeat_interleave(tokens_per_frame)
    block_ids = torch.div(frame_ids, frames_per_block, rounding_mode="floor")
    return block_ids[:, None] >= block_ids[None, :]


@dataclass
class WAN21CausalCache:
    """Detached per-layer self-attention history."""

    keys: List[Optional[torch.Tensor]]
    values: List[Optional[torch.Tensor]]
    frames: int = 0

    @classmethod
    def empty(cls, num_layers: int) -> "WAN21CausalCache":
        return cls(keys=[None] * num_layers, values=[None] * num_layers)

    def clear(self) -> None:
        self.keys[:] = [None] * len(self.keys)
        self.values[:] = [None] * len(self.values)
        self.frames = 0


@dataclass
class _CausalState:
    frames_per_block: int
    num_frames: int
    tokens_per_frame: int
    layer_index: int
    cache: Optional[WAN21CausalCache] = None
    commit_cache: bool = False
    attention_mask: Optional[torch.Tensor] = None


class WAN21CausalAttnProcessor:
    """WAN self-attention processor with block mask and optional history K/V."""

    _attention_backend = None
    _parallel_config = None

    def __init__(self, layer_index: int):
        self.layer_index = int(layer_index)

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        causal_state: Optional[_CausalState] = None,
        **_: Any,
    ) -> torch.Tensor:
        if encoder_hidden_states is not None:
            raise ValueError("WAN21CausalAttnProcessor is only valid for self-attention.")
        if causal_state is None:
            raise ValueError("WAN causal self-attention requires causal_state.")

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        query = attn.norm_q(query).unflatten(2, (attn.heads, -1))
        key = attn.norm_k(key).unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))
        if rotary_emb is not None:
            query = _apply_rotary(query, *rotary_emb)
            key = _apply_rotary(key, *rotary_emb)

        cache = causal_state.cache
        if cache is None:
            mask = causal_state.attention_mask
            if mask is None:
                mask = attention_mask
            if mask is not None:
                mask = mask[None, None]
            output = F.scaled_dot_product_attention(
                query.transpose(1, 2),
                key.transpose(1, 2),
                value.transpose(1, 2),
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=False,
            ).transpose(1, 2)
        else:
            layer = self.layer_index
            history_k = cache.keys[layer]
            history_v = cache.values[layer]
            key_all = key if history_k is None else torch.cat([history_k, key], dim=1)
            value_all = value if history_v is None else torch.cat([history_v, value], dim=1)
            output = F.scaled_dot_product_attention(
                query.transpose(1, 2),
                key_all.transpose(1, 2),
                value_all.transpose(1, 2),
                dropout_p=0.0,
                is_causal=False,
            ).transpose(1, 2)
            if causal_state.commit_cache:
                cache.keys[layer] = key_all.detach()
                cache.values[layer] = value_all.detach()

        output = output.flatten(2, 3).type_as(query)
        output = attn.to_out[0](output)
        return attn.to_out[1](output)


def _causal_block_forward(
    block: Any,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    rotary_emb: Tuple[torch.Tensor, torch.Tensor],
    *,
    causal_state: _CausalState,
) -> torch.Tensor:
    if temb.ndim == 4:
        values = (block.scale_shift_table.unsqueeze(0) + temb.float()).chunk(6, dim=2)
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = [v.squeeze(2) for v in values]
    else:
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            block.scale_shift_table + temb.float()
        ).chunk(6, dim=1)

    norm = (block.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
    attn_output = block.attn1(norm, None, None, rotary_emb, causal_state=causal_state)
    hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

    norm = block.norm2(hidden_states.float()).type_as(hidden_states)
    hidden_states = hidden_states + block.attn2(norm, encoder_hidden_states, None, None)

    norm = (block.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(hidden_states)
    ff_output = block.ffn(norm)
    return (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)


def _causal_bound_block_forward(
    self: Any,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    temb: torch.Tensor,
    rotary_emb: Tuple[torch.Tensor, torch.Tensor],
    *,
    causal_state: Optional[_CausalState] = None,
) -> torch.Tensor:
    """FSDP-visible block entry point for the causal implementation.

    Calling the helper directly from the transformer loop bypasses FSDP2's
    per-block pre-forward unshard hook and leaves parameters as DTensor shards.
    Binding this method onto each block and invoking ``block(...)`` keeps the
    normal module call boundary, so FSDP materializes the full block before the
    causal math and reshards it afterwards.
    """
    if causal_state is None:
        raise ValueError("WAN causal block forward requires causal_state.")
    return _causal_block_forward(
        self,
        hidden_states,
        encoder_hidden_states,
        temb,
        rotary_emb,
        causal_state=causal_state,
    )


def _causal_transformer_forward(
    self: Any,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    encoder_hidden_states_image: Optional[torch.Tensor] = None,
    return_dict: bool = True,
    attention_kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    from diffusers.models.modeling_outputs import Transformer2DModelOutput

    kwargs = dict(attention_kwargs or {})
    frames_per_block = int(kwargs.pop("frames_per_block", getattr(self, "_causal_frames_per_block", 1)))
    frame_offset = int(kwargs.pop("frame_offset", 0))
    cache = kwargs.pop("kv_cache", None)
    commit_cache = bool(kwargs.pop("commit_cache", False))
    if kwargs:
        raise ValueError(f"Unsupported WAN causal attention kwargs: {sorted(kwargs)}")

    batch_size, _, num_frames, height, width = hidden_states.shape
    p_t, p_h, p_w = self.config.patch_size
    post_frames = num_frames // p_t
    post_height = height // p_h
    post_width = width // p_w
    tokens_per_frame = post_height * post_width
    rotary_emb = _shifted_rope(self.rope, hidden_states, frame_offset=frame_offset)

    hidden_states = self.patch_embedding(hidden_states).flatten(2).transpose(1, 2)
    if timestep.ndim == 2:
        ts_seq_len = timestep.shape[1]
        timestep = timestep.flatten()
    else:
        ts_seq_len = None
    temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
        timestep, encoder_hidden_states, encoder_hidden_states_image, timestep_seq_len=ts_seq_len
    )
    timestep_proj = (
        timestep_proj.unflatten(2, (6, -1)) if ts_seq_len is not None else timestep_proj.unflatten(1, (6, -1))
    )
    if encoder_hidden_states_image is not None:
        encoder_hidden_states = torch.cat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

    full_mask = None
    if cache is None:
        full_mask = block_causal_attention_mask(
            num_frames=post_frames,
            tokens_per_frame=tokens_per_frame,
            frames_per_block=frames_per_block,
            device=hidden_states.device,
        )
    for layer_index, block in enumerate(self.blocks):
        state = _CausalState(
            frames_per_block=frames_per_block,
            num_frames=post_frames,
            tokens_per_frame=tokens_per_frame,
            layer_index=layer_index,
            cache=cache,
            commit_cache=commit_cache,
            attention_mask=full_mask,
        )
        # Mutable cache and activation recomputation are incompatible. Cached
        # rollout forwards intentionally bypass model gradient checkpointing.
        if cache is None and torch.is_grad_enabled() and self.gradient_checkpointing:
            hidden_states = self._gradient_checkpointing_func(
                lambda h, e, t, r: block(h, e, t, r, causal_state=state),
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                rotary_emb,
            )
        else:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                timestep_proj,
                rotary_emb,
                causal_state=state,
            )

    if commit_cache and cache is not None:
        cache.frames = max(cache.frames, frame_offset + post_frames)

    if temb.ndim == 3:
        shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
        shift, scale = shift.squeeze(2), scale.squeeze(2)
    else:
        shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)
    shift, scale = shift.to(hidden_states.device), scale.to(hidden_states.device)
    hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
    hidden_states = self.proj_out(hidden_states)
    hidden_states = hidden_states.reshape(batch_size, post_frames, post_height, post_width, p_t, p_h, p_w, -1)
    hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
    output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)
    if not return_dict:
        return (output,)
    return Transformer2DModelOutput(sample=output)


def enable_wan_block_causal(transformer: Any, *, frames_per_block: int = 1) -> Any:
    """Patch a diffusers WAN transformer in-place without changing its weights."""
    if frames_per_block < 1:
        raise ValueError(f"frames_per_block must be >= 1; got {frames_per_block}.")
    transformer._causal_frames_per_block = int(frames_per_block)
    for index, block in enumerate(transformer.blocks):
        block.attn1.set_processor(WAN21CausalAttnProcessor(index))
        block.forward = types.MethodType(_causal_bound_block_forward, block)
    transformer.forward = types.MethodType(_causal_transformer_forward, transformer)
    transformer._wan_block_causal_enabled = True
    return transformer


__all__ = [
    "WAN21CausalCache",
    "WAN21CausalAttnProcessor",
    "block_causal_attention_mask",
    "enable_wan_block_causal",
]
