"""Navit-forward adapter over the PRISTINE official Bagel modeling.

The official ``ByteDance-Seed/Bagel`` ``_forward_flow`` is the velocity predictor
the RL path needs, but it (a) consumes a *packed* (navit) sequence + three KV-cache
contexts rather than a dense ``predict_noise(sample, sigma)`` and (b) carries an
upstream ``@torch.no_grad``. This module is the **thin adapter** that bridges those
two facts to UniRL's shared diffusion runtime — and nothing more:

- :func:`forward_flow`           grad-capable velocity via the pristine
                                 ``Bagel._forward_flow`` (bypasses ``@torch.no_grad``
                                 through ``functools.wraps``' ``__wrapped__``).
- :func:`disable_inference_cache` turns off TaylorSeer (per-step determinism for replay).

AR (text-out) adapters — same philosophy, for ``BagelARStage``:

- :func:`init_und_context` / :func:`prefill_text_split` / :func:`prefill_vit_split`
  build a fresh KV context from RAW prompt material (pre-tokenized ids / a
  vit-transformed image tensor) via the pristine ``forward_cache_update_text`` /
  ``forward_cache_update_vit`` reached through ``__wrapped__`` — grad-capable
  under ``enable_grad`` (AR replay trains the und path, so the prompt prefill
  must carry gradients), grad-free under rollout's ``no_grad``. One code path
  for rollout and replay ⇒ prefix K/V parity by construction.
- :func:`decode_text`            bs=1 per-token decode mirroring the vendored
                                 ``generate_text`` (bagel.py:929-1001) index
                                 bookkeeping, but emitting per-token FULL-softmax
                                 log-probs via the caller's sampling kernel
                                 (upstream returns token ids only).
- :func:`score_response`         one-shot teacher-forced replay scoring: query
                                 ``[bos] + response[:-1]`` attends causally to the
                                 prefilled context + itself (the same
                                 ``forward_inference(mode="und", is_causal=True)``
                                 call shape as the vendored text prefill), row j
                                 predicting ``response[j]`` — exactly the per-token
                                 rollout semantics, in one grad-capable pass.
- :func:`require_inference_dispatch` guards the eval()+grads replay regime (the
  navit decoder layers dispatch ``forward_train``/``forward_inference`` on
  ``self.training``; ``.train()`` mode would mis-route the packed kwargs).

Everything else the RL loop needs is UniRL's, NOT a flow_grpo port:

- the SDE transition + log-prob  → :class:`unirl.sde.kernels.FlowSDEStrategy`
- which steps run SDE            → :meth:`DiffusionSamplingParams.resolve_sde_indices`
                                   (``unirl.utils.scheduler_utils.AllSDEScheduler``)
- the σ / timestep schedule      → :class:`unirl.sde.runtime.FlowMatchSchedulePolicy`
- the initial noise x_T          → :class:`unirl.types.noise_recipe.NoiseRecipe`

so :class:`unirl.models.bagel.diffusion.BagelDiffusionStage` reads exactly like
``SD3DiffusionStage`` (central schedule + sde_indices + kernel + noise), with this
adapter supplying only the model-specific velocity call. ``vendor/`` stays
byte-pristine; an upstream bump is a re-vendor + import-rewrite with this file
untouched.

Gradients
---------
``Bagel._forward_flow`` carries ``@torch.no_grad`` upstream. :func:`forward_flow`
reaches the undecorated function via ``functools.wraps``' ``__wrapped__`` so replay
can backprop while the vendored file stays unedited (verified on torch 2.11: the
decorated form blocks grad even under ``enable_grad``; ``__wrapped__`` restores it).
Under an outer ``torch.no_grad()`` (e.g. rollout) it stays grad-free, so the same
function serves rollout, the ratio test, and training.
"""

from __future__ import annotations

import sys
from typing import Any, Callable, Dict, List, Tuple

import torch
from torch.utils.checkpoint import checkpoint

__all__ = [
    "decode_text",
    "disable_inference_cache",
    "forward_flow",
    "init_und_context",
    "prefill_text_split",
    "prefill_vit_split",
    "require_inference_dispatch",
    "score_response",
]


def disable_inference_cache(model: Any) -> None:
    """Turn off the TaylorSeer cache for the RL path (per-step determinism).

    The pristine ``_forward_flow`` reads ``self.language_model.model.enable_taylorseer``;
    the official ``generate_image`` sets it, but the RL loop calls ``_forward_flow``
    directly so we set the flag here (the cache would break per-step determinism →
    replay would not be bit-exact). Best-effort; ignored if the attribute path is
    absent (e.g. a fake model in unit tests).
    """
    try:
        model.language_model.model.enable_taylorseer = False
    except AttributeError:
        pass


def _raw(fn: Callable) -> Callable:
    """Undecorated form of a vendored ``@torch.no_grad`` method (via ``__wrapped__``).

    Bare ``@torch.no_grad`` applies ``functools.wraps`` so ``__wrapped__`` holds
    the original function (verified on torch 2.11/2.12); the fallback returns the
    function unchanged (e.g. undecorated fakes in unit tests).
    """
    return getattr(fn, "__wrapped__", fn)


def _raw_forward_flow(model: Any):
    """The undecorated ``Bagel._forward_flow`` (bypasses upstream ``@torch.no_grad``)."""
    return _raw(type(model)._forward_flow)


def forward_flow(model: Any, **kwargs: Any) -> Any:
    """Velocity prediction via the pristine vendored ``Bagel._forward_flow``.

    Bypasses upstream's ``@torch.no_grad`` (via ``__wrapped__``) so gradients flow
    during replay; under an outer ``torch.no_grad()`` it is still grad-free. The
    TaylorSeer cache kwargs (``model_pred_*``) are left at their ``None`` defaults —
    the RL path disables that cache (see :func:`disable_inference_cache`).

    ``model._forward_flow`` already does the CFG combine internally (gen / cfg_text /
    cfg_img contexts + ``cfg_text_scale`` / ``cfg_img_scale`` / ``cfg_renorm_*``), so
    the returned velocity is the CFG-combined ``v_t`` the SDE kernel consumes.
    """
    return _raw_forward_flow(model)(model, **kwargs)


# ---------------------------------------------------------------------------
# AR (text-out) adapters
# ---------------------------------------------------------------------------


def require_inference_dispatch(model: Any) -> None:
    """Raise unless the MoT is in eval() mode (the navit forward-dispatch contract).

    Every navit module routes ``forward_train`` vs ``forward_inference`` on
    ``self.training`` and ``Qwen2Model.forward_inference`` invokes decoder layers
    via ``__call__`` — so ``.train()`` mode mis-routes the packed inference kwargs
    into ``forward_train``. Replay runs in eval() with grads enabled, the same
    regime as ``BagelDiffusionStage.replay``.
    """
    lm = getattr(model, "language_model", None)
    if lm is not None and getattr(lm, "training", False):
        raise RuntimeError(
            "bagel.rl_ops: the MoT is in train() mode; the navit forward dispatches on "
            "self.training, so AR rollout/replay must run in eval() (with grads enabled "
            "for replay — same regime as BagelDiffusionStage.replay)."
        )


def init_und_context(model: Any) -> Dict[str, Any]:
    """Fresh empty KV context ``{kv_lens, ropes, past_key_values}`` (navit bs=1).

    Mirrors ``InterleaveInferencer.init_gen_context``. ``NaiveCache`` is resolved
    from the model's own modeling module (hi3 ``sys.modules`` trick) so this
    module never imports the vendored modeling (flash-attn) itself; fake models
    must export a ``NaiveCache`` from their module (see bagel_ar_cpu_check.py).
    """
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
    """``prepare_prompts``' packed-input bookkeeping for ONE pre-tokenized split.

    Byte-equivalent to the vendored ``Bagel.prepare_prompts`` (bagel.py:232-264)
    at bs=1, minus the tokenize+wrap step — ``text_ids`` are the final ids
    INCLUDING the ``bos/eos`` (``<|im_start|>``/``<|im_end|>``) wrap, so replay is
    tokenizer-independent and byte-aligned with rollout.
    """
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
    """Prefill one text split into the context; returns the advanced context.

    Runs the pristine ``forward_cache_update_text`` via ``__wrapped__`` so the
    same call is grad-capable under ``enable_grad`` (replay) and grad-free under
    ``no_grad`` (rollout).
    """
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
    """Prefill one ViT image split into the context; returns the advanced context.

    ``image_tensor`` is the ALREADY ``vit_transform``-ed ``[3, H, W]`` tensor (the
    conditions store the final transform output so rollout and replay consume
    byte-identical pixels); the pristine ``prepare_vit_images`` packer is reused
    verbatim with an identity transform. The cache update runs ``is_causal=False``
    inside — the non-causal image block within the causal stream, exactly as at
    rollout.
    """
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
    """bs=1 per-token decode over a prefilled context, emitting token+logp pairs.

    Reimplements the vendored ``generate_text`` loop (bagel.py:929-1001) — which
    returns token ids only — with the per-step ``sample_fn(logits [1, vocab]) →
    (token_id, full-softmax log-prob)`` kernel. Index bookkeeping is the bs=1
    collapse of the vendored multi-sample form: contiguous kv indexes
    ``arange(kv_len)``, query index ``[kv_len]``, position/kv_len advance by one
    per token. The returned token list INCLUDES the stop token (TextSegment
    convention); ``start_token_id`` (``new_token_ids['bos_token_id']``, as in the
    vendored ``prepare_start_tokens``) is the loop *input*, never recorded.

    Mutates ``ctx['past_key_values']`` in place (``update_past_key_values=True``)
    — callers prefill a fresh context per sample. Caller owns no_grad + autocast.
    """
    require_inference_dispatch(model)
    disable_inference_cache(model)
    lm = model.language_model
    kv_len, pos = int(ctx["kv_lens"][0]), int(ctx["ropes"][0])
    past = ctx["past_key_values"]
    stop_set = set(int(t) for t in stop_ids)

    # Hoisted index pools — per-step values are views (kv indexes grow by one
    # contiguous slot per token, so arange slices cover every step).
    max_new = int(max_new_tokens)
    all_indexes = torch.arange(kv_len + max_new, dtype=torch.long, device=device)
    all_positions = torch.arange(pos, pos + max_new, dtype=torch.long, device=device)
    all_kv_lens = torch.arange(kv_len, kv_len + max_new, dtype=torch.int, device=device)

    curr = torch.tensor([int(start_token_id)], dtype=torch.long, device=device)
    tokens: List[int] = []
    logps: List[float] = []
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
        logits = lm.lm_head(out.packed_query_sequence)  # [1, vocab]
        token_id, log_prob = sample_fn(logits)
        tid = int(token_id.item())
        tokens.append(tid)
        logps.append(float(log_prob.item()))
        if tid in stop_set:
            break
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
    """One-shot teacher-forced per-token log-probs of ``response_ids`` — grad-capable.

    Query ``[start] + response[:-1]`` (length n) attends causally to the prefilled
    context + itself (``is_causal=True``, ``update_past_key_values=False`` — the
    same ``forward_inference(mode="und")`` call shape as the vendored text
    prefill); flash-attn's bottom-right causal alignment makes row ``j`` attend to
    ``prefix + query[0..j]``, so row ``j`` predicts ``response[j]`` — exactly the
    per-token rollout semantics.

    Log-probs are the FULL softmax of ``lm_head(h).float() / T`` (gather −
    logsumexp), matching the rollout kernel's pre-truncation convention; the
    lm_head runs chunked (never materializing ``[n, vocab]`` whole) with per-chunk
    gradient checkpointing when grads are enabled. Returns fp32 ``[n]``. Caller
    owns the grad scope (eval() + ``enable_grad`` for replay) and autocast.
    """
    require_inference_dispatch(model)
    disable_inference_cache(model)
    lm = model.language_model
    kv_len, pos = int(ctx["kv_lens"][0]), int(ctx["ropes"][0])
    n = int(response_ids.numel())
    if n == 0:
        return torch.zeros(0, dtype=torch.float32, device=device)

    response_ids = response_ids.to(device=device, dtype=torch.long)
    start = torch.tensor([int(start_token_id)], dtype=torch.long, device=device)
    query_ids = torch.cat([start, response_ids[:-1]], dim=0)  # [n]

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
    hidden = out.packed_query_sequence  # [n, H]

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
