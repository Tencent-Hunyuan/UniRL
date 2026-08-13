"""Navit-forward adapter over the PRISTINE official Bagel modeling."""

from __future__ import annotations

import sys
from typing import Any, Callable, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

__all__ = [
    "decode_text",
    "disable_inference_cache",
    "forward_flow",
    "init_und_context",
    "pack_und_forward_inputs",
    "prefill_text_split",
    "prefill_vit_split",
    "require_inference_dispatch",
    "score_response",
    "score_response_with_prompt",
    "und_replay_logits",
]


def disable_inference_cache(model: Any) -> None:
    """Turn off the TaylorSeer cache for the RL path (per-step determinism)."""
    try:
        model.language_model.model.enable_taylorseer = False
    except AttributeError:
        pass


def _raw(fn: Callable) -> Callable:
    """Undecorated form of a vendored ``@torch.no_grad`` method (via ``__wrapped__``)."""
    return getattr(fn, "__wrapped__", fn)


def _raw_forward_flow(model: Any):
    """The undecorated ``Bagel._forward_flow`` (bypasses upstream ``@torch.no_grad``)."""
    return _raw(type(model)._forward_flow)


def forward_flow(model: Any, **kwargs: Any) -> Any:
    """Velocity prediction via the pristine vendored ``Bagel._forward_flow``."""
    lm = model.language_model
    was_training = lm.training
    grad_enabled = torch.is_grad_enabled()
    if was_training:
        lm.eval()
    try:
        return _raw_forward_flow(model)(model, **kwargs)
    finally:
        if was_training and not grad_enabled:
            lm.train()


def require_inference_dispatch(model: Any) -> None:
    """Raise unless the MoT is in eval() mode (the navit forward-dispatch contract)."""
    lm = getattr(model, "language_model", None)
    if lm is not None and getattr(lm, "training", False):
        raise RuntimeError(
            "bagel.rl_ops: the MoT is in train() mode; the navit forward dispatches on "
            "self.training, so AR rollout/replay must run in eval() (with grads enabled "
            "for replay — same regime as BagelDiffusionStage.replay)."
        )


def init_und_context(model: Any) -> Dict[str, Any]:
    """Fresh empty KV context ``{kv_lens, ropes, past_key_values}`` (navit bs=1)."""
    lm_model = model.language_model.model
    num_layers = int(model.config.llm_config.num_hidden_layers)
    cache_cls = getattr(sys.modules[type(lm_model).__module__], "NaiveCache", None)
    if cache_cls is None:
        raise RuntimeError(
            f"bagel.rl_ops.init_und_context: module {type(lm_model).__module__!r} exports no "
            "NaiveCache; fake models must define one (per-layer key_cache/value_cache dicts)."
        )
    return {"kv_lens": [0], "ropes": [0], "past_key_values": cache_cls(num_layers)}


def _pack_text_ids(text_ids: torch.Tensor, *, kv_len: int, rope_start: int) -> Dict[str, torch.Tensor]:
    """``prepare_prompts``' packed-input bookkeeping for ONE pre-tokenized split."""
    n = int(text_ids.numel())
    return {
        "text_token_lens": torch.tensor([n], dtype=torch.int),
        "packed_text_ids": text_ids.to(dtype=torch.long),
        "packed_text_position_ids": torch.arange(rope_start, rope_start + n, dtype=torch.long),
        "packed_text_indexes": torch.arange(kv_len, kv_len + n, dtype=torch.long),
        "packed_key_value_indexes": torch.arange(kv_len, dtype=torch.long),
        "key_values_lens": torch.tensor([kv_len], dtype=torch.int),
    }


def _to_device(d: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Move every tensor value onto ``device`` (non-tensors pass through)."""
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in d.items()}


def prefill_text_split(
    model: Any, ctx: Dict[str, Any], *, text_ids: torch.Tensor, device: torch.device
) -> Dict[str, Any]:
    """Prefill one text split into the context; returns the advanced context."""
    kv_len, rope = int(ctx["kv_lens"][0]), int(ctx["ropes"][0])
    gi = _to_device(_pack_text_ids(text_ids, kv_len=kv_len, rope_start=rope), device)
    past = _raw(type(model).forward_cache_update_text)(model, ctx["past_key_values"], **gi)
    n = int(text_ids.numel())
    return {"kv_lens": [kv_len + n], "ropes": [rope + n], "past_key_values": past}


def prefill_vit_split(
    model: Any,
    ctx: Dict[str, Any],
    *,
    image_tensor: torch.Tensor,
    new_token_ids: Dict[str, int],
    device: torch.device,
) -> Dict[str, Any]:
    """Prefill one ViT image split into the context; returns the advanced context."""
    gi, newlens, new_rope = model.prepare_vit_images(
        curr_kvlens=ctx["kv_lens"],
        curr_rope=ctx["ropes"],
        images=[image_tensor],
        transforms=lambda x: x,
        new_token_ids=new_token_ids,
    )
    gi = _to_device(gi, device)
    past = _raw(type(model).forward_cache_update_vit)(model, ctx["past_key_values"], **gi)
    return {"kv_lens": newlens, "ropes": new_rope, "past_key_values": past}


def decode_text(
    model: Any,
    ctx: Dict[str, Any],
    *,
    start_token_id: int,
    sample_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
    max_new_tokens: int,
    stop_ids: List[int],
    device: torch.device,
) -> Tuple[List[int], List[float]]:
    """bs=1 per-token decode over a prefilled context; ``sample_fn`` sees ``logits [1, vocab]`` per step."""
    require_inference_dispatch(model)
    disable_inference_cache(model)
    lm = model.language_model
    kv_len, pos = int(ctx["kv_lens"][0]), int(ctx["ropes"][0])
    past = ctx["past_key_values"]
    stop_set = set(int(t) for t in stop_ids)

    max_new = int(max_new_tokens)
    all_indexes = torch.arange(kv_len + max_new, dtype=torch.long, device=device)
    all_positions = torch.arange(pos, pos + max_new, dtype=torch.long, device=device)
    all_kv_lens = torch.arange(kv_len, kv_len + max_new, dtype=torch.int, device=device)

    curr = torch.tensor([int(start_token_id)], dtype=torch.long, device=device)
    tokens: List[int] = []
    logps: List[float] = []
    done = False
    for j in range(max_new):
        emb = lm.model.embed_tokens(curr)
        out = lm.forward_inference(
            packed_query_sequence=emb,
            query_lens=torch.ones_like(curr),
            packed_query_position_ids=all_positions[j : j + 1],
            packed_query_indexes=all_indexes[kv_len + j : kv_len + j + 1],
            past_key_values=past,
            key_values_lens=all_kv_lens[j : j + 1],
            packed_key_value_indexes=all_indexes[: kv_len + j],
            update_past_key_values=True,
            is_causal=True,
            mode="und",
        )
        past = out.past_key_values
        logits = lm.lm_head(out.packed_query_sequence)
        token_id, log_prob = sample_fn(logits)
        tid = int(token_id.item())
        if not done:
            tokens.append(tid)
            logps.append(float(log_prob.item()))
            if tid in stop_set:
                done = True
        curr = token_id.to(device=device, dtype=torch.long).reshape(1)
    return tokens, logps


def score_response(
    model: Any,
    ctx: Dict[str, Any],
    *,
    response_ids: torch.Tensor,
    start_token_id: int,
    temperature: float = 1.0,
    logprob_chunk: int = 1024,
    device: torch.device,
) -> torch.Tensor:
    """Teacher-forced per-token log-probs of ``response_ids``, chunked lm_head, grad-capable; returns fp32 ``[n]``."""
    require_inference_dispatch(model)
    disable_inference_cache(model)
    lm = model.language_model
    kv_len, pos = int(ctx["kv_lens"][0]), int(ctx["ropes"][0])
    n = int(response_ids.numel())
    if n == 0:
        return torch.zeros(0, dtype=torch.float32, device=device)

    response_ids = response_ids.to(device=device, dtype=torch.long)
    start = torch.tensor([int(start_token_id)], dtype=torch.long, device=device)
    query_ids = torch.cat([start, response_ids[:-1]], dim=0)

    emb = lm.model.embed_tokens(query_ids)
    out = lm.forward_inference(
        packed_query_sequence=emb,
        query_lens=torch.tensor([n], dtype=torch.int, device=device),
        packed_query_position_ids=torch.arange(pos, pos + n, dtype=torch.long, device=device),
        packed_query_indexes=torch.arange(kv_len, kv_len + n, dtype=torch.long, device=device),
        past_key_values=ctx["past_key_values"],
        key_values_lens=torch.tensor([kv_len], dtype=torch.int, device=device),
        packed_key_value_indexes=torch.arange(kv_len, dtype=torch.long, device=device),
        update_past_key_values=False,
        is_causal=True,
        mode="und",
    )
    hidden = out.packed_query_sequence

    temp = float(temperature) if float(temperature) > 0.0 else 1.0

    def _chunk_logp(h: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        logits = lm.lm_head(h).float() / temp
        return logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, dim=-1)

    use_ckpt = torch.is_grad_enabled() and hidden.requires_grad
    parts: List[torch.Tensor] = []
    for s in range(0, n, int(logprob_chunk)):
        h, tgt = hidden[s : s + int(logprob_chunk)], response_ids[s : s + int(logprob_chunk)]
        if use_ckpt:
            parts.append(checkpoint(_chunk_logp, h, tgt, use_reentrant=False))
        else:
            parts.append(_chunk_logp(h, tgt))
    return torch.cat(parts, dim=0)


def score_response_with_prompt(
    model: Any,
    ctx: Dict[str, Any],
    *,
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
    start_token_id: int,
    temperature: float = 1.0,
    logprob_chunk: int = 1024,
    device: torch.device,
) -> torch.Tensor:
    """Inference-mode replay scorer: one grad ``forward_inference`` attending to a frozen no_grad image context."""
    require_inference_dispatch(model)
    disable_inference_cache(model)
    lm = model.language_model
    kv_len, pos = int(ctx["kv_lens"][0]), int(ctx["ropes"][0])
    n = int(response_ids.numel())
    if n == 0:
        return torch.zeros(0, dtype=torch.float32, device=device)

    response_ids = response_ids.to(device=device, dtype=torch.long)
    prompt = torch.as_tensor(prompt_ids, dtype=torch.long, device=device).reshape(-1)
    start = torch.tensor([int(start_token_id)], dtype=torch.long, device=device)
    query_ids = torch.cat([prompt, start, response_ids[:-1]], dim=0)
    m = int(query_ids.numel())

    emb = lm.model.embed_tokens(query_ids)
    out = lm.forward_inference(
        packed_query_sequence=emb,
        query_lens=torch.tensor([m], dtype=torch.int, device=device),
        packed_query_position_ids=torch.arange(pos, pos + m, dtype=torch.long, device=device),
        packed_query_indexes=torch.arange(kv_len, kv_len + m, dtype=torch.long, device=device),
        past_key_values=ctx["past_key_values"],
        key_values_lens=torch.tensor([kv_len], dtype=torch.int, device=device),
        packed_key_value_indexes=torch.arange(kv_len, dtype=torch.long, device=device),
        update_past_key_values=False,
        is_causal=True,
        mode="und",
    )
    hidden = out.packed_query_sequence[-n:]

    temp = float(temperature) if float(temperature) > 0.0 else 1.0

    def _chunk_logp(h: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        logits = lm.lm_head(h).float() / temp
        return logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, dim=-1)

    use_ckpt = torch.is_grad_enabled() and hidden.requires_grad
    parts: List[torch.Tensor] = []
    for s in range(0, n, int(logprob_chunk)):
        h, tgt = hidden[s : s + int(logprob_chunk)], response_ids[s : s + int(logprob_chunk)]
        if use_ckpt:
            parts.append(checkpoint(_chunk_logp, h, tgt, use_reentrant=False))
        else:
            parts.append(_chunk_logp(h, tgt))
    return torch.cat(parts, dim=0)


def pack_und_forward_inputs(
    model: Any,
    *,
    new_token_ids: Dict[str, Any],
    splits: List[Dict[str, Any]],
    response_input: torch.Tensor,
    device: torch.device,
    vit_transform: Callable[[Any], Any] = lambda x: x,
) -> Dict[str, Any]:
    """Train-mode packing: one und sample ``[*ordered splits | response_input]`` with a nested attention mask."""
    from .vendor.data.data_utils import prepare_attention_mask_per_sample

    text_ids: List[int] = []
    text_indexes: List[int] = []
    position_ids: List[int] = []
    vit_tokens_parts: List[torch.Tensor] = []
    vit_position_ids_parts: List[torch.Tensor] = []
    vit_token_indexes: List[int] = []
    vit_seqlens_parts: List[torch.Tensor] = []
    split_lens: List[int] = []
    attn_modes: List[str] = []
    pos = 0
    rope = 0

    def _append_text_block(ids: List[int]) -> None:
        nonlocal pos, rope
        for tid in ids:
            text_ids.append(int(tid))
            text_indexes.append(pos)
            position_ids.append(rope)
            pos += 1
            rope += 1
        split_lens.append(len(ids))
        attn_modes.append("causal")

    for sp in splits:
        kind = sp.get("kind")
        if kind == "vit":
            vit_input, _, _ = model.prepare_vit_images(
                curr_kvlens=[0],
                curr_rope=[rope],
                images=[sp["image"]],
                transforms=vit_transform,
                new_token_ids=new_token_ids,
            )
            img_block_len = int(vit_input["packed_seqlens"][0].item())
            text_ids.extend(int(t) for t in vit_input["packed_text_ids"].tolist())
            text_indexes.extend(pos + int(t) for t in vit_input["packed_text_indexes"].tolist())
            position_ids.extend(int(p) for p in vit_input["packed_position_ids"].tolist())
            vit_token_indexes.extend(pos + int(t) for t in vit_input["packed_vit_token_indexes"].tolist())
            vit_tokens_parts.append(vit_input["packed_vit_tokens"])
            vit_position_ids_parts.append(vit_input["packed_vit_position_ids"])
            vit_seqlens_parts.append(vit_input["vit_token_seqlens"])
            pos += img_block_len
            rope += 1
            split_lens.append(img_block_len)
            attn_modes.append("full")
        elif kind == "text":
            _append_text_block([int(t) for t in sp["ids"].tolist()])
        else:
            raise ValueError(f"pack_und_forward_inputs: unknown split kind {kind!r}; expected 'text' or 'vit'.")

    resp_start = pos
    _append_text_block([int(t) for t in response_input.tolist()])
    ce_loss_indexes = list(range(resp_start, resp_start + int(response_input.shape[0])))

    seqlen = pos
    nested_mask = prepare_attention_mask_per_sample(split_lens, attn_modes, device=device)

    return {
        "seqlen": seqlen,
        "sample_lens": [seqlen],
        "packed_text_ids": torch.tensor(text_ids, dtype=torch.long, device=device),
        "packed_text_indexes": torch.tensor(text_indexes, dtype=torch.long, device=device),
        "packed_position_ids": torch.tensor(position_ids, dtype=torch.long, device=device),
        "nested_attention_masks": [nested_mask],
        "packed_vit_tokens": (
            torch.cat(vit_tokens_parts, dim=0).to(device=device, dtype=model.dtype) if vit_tokens_parts else None
        ),
        "packed_vit_position_ids": (
            torch.cat(vit_position_ids_parts, dim=0).to(device) if vit_position_ids_parts else None
        ),
        "packed_vit_token_indexes": (
            torch.tensor(vit_token_indexes, dtype=torch.long, device=device) if vit_token_indexes else None
        ),
        "vit_token_seqlens": (torch.cat(vit_seqlens_parts, dim=0).to(device) if vit_seqlens_parts else None),
        "ce_loss_indexes": torch.tensor(ce_loss_indexes, dtype=torch.long, device=device),
    }


def und_replay_logits(model: Any, packed: Dict[str, Any]) -> torch.Tensor:
    """Train-mode grad-carrying und TRAINING forward; returns response-position logits ``[R, V]``."""
    lm = model.language_model
    packed_text_embedding = lm.model.embed_tokens(packed["packed_text_ids"])
    packed_sequence = packed_text_embedding.new_zeros((packed["seqlen"], model.hidden_size))
    packed_sequence[packed["packed_text_indexes"]] = packed_text_embedding

    packed_und_token_indexes = packed["packed_text_indexes"]
    if packed["packed_vit_tokens"] is not None:
        cu_seqlens = F.pad(torch.cumsum(packed["vit_token_seqlens"], dim=0), (1, 0)).to(torch.int32)
        max_seqlen = int(torch.max(packed["vit_token_seqlens"]).item())
        vit_embed = model.vit_model(
            packed_pixel_values=packed["packed_vit_tokens"],
            packed_flattened_position_ids=packed["packed_vit_position_ids"],
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        vit_embed = model.connector(vit_embed)
        vit_embed = vit_embed + model.vit_pos_embed(packed["packed_vit_position_ids"])
        packed_sequence[packed["packed_vit_token_indexes"]] = vit_embed
        packed_und_token_indexes = torch.cat([packed["packed_text_indexes"], packed["packed_vit_token_indexes"]], dim=0)

    last_hidden_state = lm(
        packed_sequence=packed_sequence,
        sample_lens=packed["sample_lens"],
        attention_mask=packed["nested_attention_masks"],
        packed_position_ids=packed["packed_position_ids"],
        packed_und_token_indexes=packed_und_token_indexes,
        packed_gen_token_indexes=None,
    )
    return lm.lm_head(last_hidden_state[packed["ce_loss_indexes"]])
