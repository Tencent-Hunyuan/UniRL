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
from types import MethodType
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from unirl.config.require import require

__all__ = [
    "decode_text",
    "detach_replay_tree",
    "disable_inference_cache",
    "forward_flow",
    "init_und_context",
    "install_layer_major_replay_dispatch",
    "move_replay_tree",
    "pack_und_forward_inputs",
    "prefill_prompt_text",
    "prefill_text_split",
    "prefill_vit_split",
    "rebuild_text_context_from_chunks",
    "validate_t2ti_replay_chunk_mode",
    "validate_t2ti_replay_execution_order",
    "require_inference_dispatch",
    "score_response",
    "score_response_with_prompt",
    "und_replay_logits",
]


T2TI_REPLAY_CHUNK_MODES = ("exact", "collapsed")
T2TI_REPLAY_EXECUTION_ORDERS = ("chunk_major", "layer_major")


def validate_t2ti_replay_chunk_mode(mode: Any) -> str:
    """Return a normalized T2TI cache-replay mode or raise.

    ``exact`` preserves the Stage-0 scheduler boundaries. ``collapsed`` preserves
    the initial prefill boundary and coalesces the decode tail into one causal
    update. The latter changes decode-kernel geometry, so callers must opt into it.
    """
    normalized = str(mode).strip().lower()
    require(
        normalized in T2TI_REPLAY_CHUNK_MODES,
        f"Bagel T2TI replay chunk mode must be one of {T2TI_REPLAY_CHUNK_MODES}; got {mode!r}.",
    )
    return normalized


def validate_t2ti_replay_execution_order(order: Any) -> str:
    """Validate how the exact chunk/layer replay DAG is traversed."""
    normalized = str(order).strip().lower()
    require(
        normalized in T2TI_REPLAY_EXECUTION_ORDERS,
        f"Bagel T2TI replay execution order must be one of {T2TI_REPLAY_EXECUTION_ORDERS}; got {order!r}.",
    )
    return normalized


def _layer_major_replay_forward_dispatch(self: Any, *args: Any, **kwargs: Any) -> Any:
    """Enter one wrapped decoder block and replay all native chunks inside it."""
    replay_chunks = kwargs.pop("_unirl_exact_replay_chunks", None)
    if replay_chunks is None:
        return self._unirl_original_forward(*args, **kwargs)

    require(not args, "BAGEL layer-major replay accepts keyword inputs only.")
    require(not bool(self.training), "BAGEL layer-major replay requires decoder layers in eval mode.")
    cache = kwargs.pop("past_key_values", None)
    require(cache is not None, "BAGEL layer-major replay requires a cache.")
    require(not kwargs, f"BAGEL layer-major replay received unexpected inputs: {sorted(kwargs)}.")
    outputs: List[torch.Tensor] = []
    for chunk in replay_chunks:
        hidden, cache = self.forward_inference(past_key_values=cache, **chunk)
        outputs.append(hidden)
    return tuple(outputs), cache


def install_layer_major_replay_dispatch(model: Any) -> None:
    """Install the replay dispatch before Accelerate/checkpoint/FSDP wrapping.

    The permanent instance dispatch falls through to the pristine bound
    ``forward`` for every normal BAGEL call. The special replay entry is still a
    single ``decoder_layer(...)`` invocation, so composable checkpointing and
    FSDP wrap the whole native-chunk loop and unshard each block only once.
    """
    try:
        layers = tuple(model.language_model.model.layers)
    except AttributeError as exc:
        raise ValueError("BAGEL layer-major replay could not find language_model.model.layers.") from exc
    require(bool(layers), "BAGEL layer-major replay requires at least one decoder layer.")
    for layer in layers:
        if bool(getattr(layer, "_unirl_layer_major_replay_installed", False)):
            continue
        require(
            callable(getattr(layer, "forward_inference", None)),
            f"BAGEL layer-major replay requires forward_inference on {type(layer).__name__}.",
        )
        layer._unirl_original_forward = layer.forward
        layer.forward = MethodType(_layer_major_replay_forward_dispatch, layer)
        layer._unirl_layer_major_replay_installed = True


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

    Training-mode contract: the vendored decoder layer dispatches train vs inference
    on ``self.training``, and ``_forward_flow`` goes through the ``forward_inference``
    (packed-query) signature, so the language model MUST be in ``eval()`` here. The two
    stages share one MoT instance within a single optimizer step (AR teacher-force sets
    train(); this diffusion replay needs eval()), so we cannot rely on the inherited
    mode. Force eval; under grad (replay) KEEP it eval so activation-checkpointing's
    recompute in the LATER ``.backward()`` still takes ``forward_inference`` (reverting
    to a stray train() would dispatch the packed-query kwargs into ``forward_train`` →
    "unexpected keyword argument 'packed_query_sequence'"). Restore only when no
    backward follows (rollout / no_grad).
    """
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


def prefill_prompt_text(
    model: Any,
    ctx: Dict[str, Any],
    *,
    prompt: str,
    tokenizer: Any,
    new_token_ids: Any,
    device: torch.device,
) -> Dict[str, Any]:
    """Tokenize and prefill one prompt with every packed tensor on ``device``.

    BAGEL's vendored ``prepare_prompts`` intentionally creates CPU tensors and
    its reference inferencer relies on Accelerate hooks to move them. Training
    removes those hooks before FSDP2 takes ownership, so trainside context
    construction must make the transfer explicit.
    """
    generation_input, kv_lens, ropes = model.prepare_prompts(
        curr_kvlens=ctx["kv_lens"],
        curr_rope=ctx["ropes"],
        prompts=[prompt],
        tokenizer=tokenizer,
        new_token_ids=new_token_ids,
    )
    generation_input = _to_device(generation_input, device)
    past = model.forward_cache_update_text(ctx["past_key_values"], **generation_input)
    return {"kv_lens": kv_lens, "ropes": ropes, "past_key_values": past}


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


def _rebuild_text_context_layer_major(
    model: Any,
    *,
    replay_chunks: Sequence[Sequence[int]],
    device: torch.device,
) -> Dict[str, Any]:
    """Traverse the exact chunk/layer replay DAG with one wrapped call per layer."""
    lm_model = model.language_model.model
    layers = tuple(lm_model.layers)
    require(bool(layers), "BAGEL layer-major replay requires decoder layers.")
    require(
        not bool(getattr(lm_model, "enable_taylorseer", False)),
        "BAGEL layer-major exact replay requires TaylorSeer to be disabled.",
    )
    require(
        all(bool(getattr(layer, "_unirl_layer_major_replay_installed", False)) for layer in layers),
        "BAGEL layer-major replay dispatch was not installed before checkpoint/FSDP wrapping.",
    )

    ctx = init_und_context(model)
    kv_len = int(ctx["kv_lens"][0])
    rope = int(ctx["ropes"][0])
    hidden_states: List[torch.Tensor] = []
    chunk_inputs: List[Dict[str, Any]] = []
    for chunk in replay_chunks:
        token_ids = torch.as_tensor(chunk, dtype=torch.long)
        generation_input = _to_device(_pack_text_ids(token_ids, kv_len=kv_len, rope_start=rope), device)
        hidden = lm_model.embed_tokens(generation_input["packed_text_ids"])
        cos, sin = lm_model.rotary_emb(hidden, generation_input["packed_text_position_ids"].unsqueeze(0))
        hidden_states.append(hidden)
        chunk_inputs.append(
            {
                "query_lens": generation_input["text_token_lens"],
                "packed_query_position_embeddings": (cos.squeeze(0), sin.squeeze(0)),
                "packed_query_indexes": generation_input["packed_text_indexes"],
                "key_values_lens": generation_input["key_values_lens"],
                "packed_key_value_indexes": generation_input["packed_key_value_indexes"],
                "update_past_key_values": True,
                "is_causal": True,
                "mode": "und",
            }
        )
        token_count = int(token_ids.numel())
        kv_len += token_count
        rope += token_count

    cache = ctx["past_key_values"]
    for layer in layers:
        per_chunk = tuple(
            {**static_inputs, "packed_query_sequence": hidden}
            for hidden, static_inputs in zip(hidden_states, chunk_inputs)
        )
        hidden_states, cache = layer(
            _unirl_exact_replay_chunks=per_chunk,
            past_key_values=cache,
        )
        require(
            len(hidden_states) == len(replay_chunks),
            "BAGEL layer-major replay returned the wrong number of chunk hidden states.",
        )

    return {"kv_lens": [kv_len], "ropes": [rope], "past_key_values": cache}


def _distributed_replay_chunk_target(local_count: int, *, device: torch.device) -> int:
    """Return the largest exact-replay traversal count in the DP world.

    The BAGEL production recipe is pure data parallelism (SP=1), so the default
    process group is also the FSDP group. A sequence-parallel rollout would need
    to synchronize on its data-parallel mesh instead.
    """
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() <= 1:
        return int(local_count)
    count = torch.tensor([int(local_count)], dtype=torch.int64, device=device)
    dist.all_reduce(count, op=dist.ReduceOp.MAX)
    return int(count.item())


def _cache_update_dependency_zero(cache: Any) -> torch.Tensor:
    """Build an exact zero whose graph touches the newest K/V from every layer."""
    dependencies: List[torch.Tensor] = []
    for store in (cache.key_cache, cache.value_cache):
        for layer_idx in sorted(store):
            value = store[layer_idx]
            if value is not None and value.numel() > 0:
                dependencies.append(value[-1].float().sum())
    require(bool(dependencies), "BAGEL collective replay padding produced no cache tensors.")
    return torch.stack(dependencies).sum() * 0.0


def _fork_cache_last_token(cache: Any) -> Any:
    """Fork a ``NaiveCache`` with one graph-connected K/V slot per layer."""
    trimmed = cache.fork()
    for store_name in ("key_cache", "value_cache"):
        source = getattr(cache, store_name)
        destination = getattr(trimmed, store_name)
        for layer_idx, value in source.items():
            destination[layer_idx] = None if value is None else value[-1:]
    return trimmed


def detach_replay_tree(value: Any, _memo: Optional[Dict[int, Any]] = None) -> Any:
    """Detach a BAGEL replay input tree while preserving cache aliasing.

    ``forward_kwargs`` contains ordinary tensor leaves plus vendored
    ``NaiveCache`` objects. PyTorch's pytree helpers do not know how to traverse
    that cache, so handle its K/V stores explicitly and memoize object identities:
    BAGEL intentionally aliases the positive ``gen`` and ``cfg_img`` caches.
    Storage is shared with the completed replay; no model-sized clone is made.
    """
    memo = {} if _memo is None else _memo
    object_id = id(value)
    if object_id in memo:
        return memo[object_id]
    if torch.is_tensor(value):
        detached = value.detach()
        memo[object_id] = detached
        return detached
    if isinstance(value, dict):
        detached_dict: Dict[Any, Any] = {}
        memo[object_id] = detached_dict
        detached_dict.update({key: detach_replay_tree(item, memo) for key, item in value.items()})
        return detached_dict
    if isinstance(value, list):
        detached_list: List[Any] = []
        memo[object_id] = detached_list
        detached_list.extend(detach_replay_tree(item, memo) for item in value)
        return detached_list
    if isinstance(value, tuple):
        detached_tuple = tuple(detach_replay_tree(item, memo) for item in value)
        memo[object_id] = detached_tuple
        return detached_tuple
    if callable(getattr(value, "fork", None)) and hasattr(value, "key_cache") and hasattr(value, "value_cache"):
        detached_cache = value.fork()
        memo[object_id] = detached_cache
        for store_name in ("key_cache", "value_cache"):
            source = getattr(value, store_name)
            destination = getattr(detached_cache, store_name)
            for layer_idx, tensor in source.items():
                destination[layer_idx] = detach_replay_tree(tensor, memo) if tensor is not None else None
        return detached_cache
    memo[object_id] = value
    return value


def move_replay_tree(
    value: Any,
    device: torch.device | str,
    _memo: Optional[Dict[int, Any]] = None,
) -> Any:
    """Copy a detached BAGEL replay tree to ``device`` without losing aliases.

    Prepared image replay contains ordinary containers and vendored
    ``NaiveCache`` instances, including an intentional alias between the
    positive ``gen`` and ``cfg_img`` caches. Copy every tensor leaf even when it
    is already on the destination device so the moved tree owns independent
    storage. This lets the prepared queue release its accelerator tree without
    sharing mutable tensor storage with either the source or a later hydration.

    The copy is synchronous by design: prepared CPU tensors are pageable, and a
    completed copy gives the consumer an unambiguous lifetime boundary. Tensor
    dtype and layout are preserved; every leaf is detached before transfer.
    """
    destination_device = torch.device(device)
    memo = {} if _memo is None else _memo
    object_id = id(value)
    if object_id in memo:
        return memo[object_id]
    if torch.is_tensor(value):
        moved = value.detach().to(device=destination_device, copy=True)
        memo[object_id] = moved
        return moved
    if isinstance(value, dict):
        moved_dict: Dict[Any, Any] = {}
        memo[object_id] = moved_dict
        moved_dict.update({key: move_replay_tree(item, destination_device, memo) for key, item in value.items()})
        return moved_dict
    if isinstance(value, list):
        moved_list: List[Any] = []
        memo[object_id] = moved_list
        moved_list.extend(move_replay_tree(item, destination_device, memo) for item in value)
        return moved_list
    if isinstance(value, tuple):
        moved_tuple = tuple(move_replay_tree(item, destination_device, memo) for item in value)
        memo[object_id] = moved_tuple
        return moved_tuple
    if callable(getattr(value, "fork", None)) and hasattr(value, "key_cache") and hasattr(value, "value_cache"):
        moved_cache = value.fork()
        memo[object_id] = moved_cache
        for store_name in ("key_cache", "value_cache"):
            source = getattr(value, store_name)
            destination = getattr(moved_cache, store_name)
            for layer_idx, tensor in source.items():
                destination[layer_idx] = (
                    move_replay_tree(tensor, destination_device, memo) if tensor is not None else None
                )
        return moved_cache
    memo[object_id] = value
    return value


def _pad_text_context_traversals(
    model: Any,
    ctx: Dict[str, Any],
    *,
    count: int,
    token_id: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Issue bounded cached traversals so every FSDP rank has equal depth."""
    if int(count) <= 0:
        return None

    # Keep the dummy suffix on the same cached update branch as the real trace.
    # Trimming every layer back to its newest slot bounds each update to a 1+1
    # attention geometry while preserving graph recurrence through the cache.
    # ``fork`` and the K/V views also keep the semantic context untouched.
    token_ids = torch.tensor([int(token_id)], dtype=torch.long, device=device)
    dummy_ctx = {
        "kv_lens": [1],
        "ropes": [int(x) for x in ctx["ropes"]],
        "past_key_values": _fork_cache_last_token(ctx["past_key_values"]),
    }
    for _ in range(int(count)):
        dummy_ctx = prefill_text_split(
            model,
            dummy_ctx,
            text_ids=token_ids,
            device=device,
        )
        dummy_ctx = {
            "kv_lens": [1],
            "ropes": dummy_ctx["ropes"],
            "past_key_values": _fork_cache_last_token(dummy_ctx["past_key_values"]),
        }
    if not torch.is_grad_enabled():
        return None
    return _cache_update_dependency_zero(dummy_ctx["past_key_values"])


def rebuild_text_context_from_chunks(
    model: Any,
    *,
    chunks: Sequence[Sequence[int]],
    expected_kv_length: int,
    expected_ropes: Sequence[int],
    device: torch.device,
    chunk_mode: str = "exact",
    execution_order: str = "chunk_major",
    collective_target_chunks: Optional[int] = None,
) -> Dict[str, Any]:
    """Rebuild a grad-capable BAGEL context from native cache-input tokens.

    ``chunks`` are captured at the vLLM Stage-0 runner boundary, after prompt
    tokenization and scheduling. ``chunk_mode="exact"`` preserves every native
    prefill/decode attention call. ``execution_order="layer_major"`` traverses
    the same two-dimensional chunk/layer DAG in the alternate topological order,
    entering each wrapped decoder block once while its parameters are resident.
    The explicit ``"collapsed"`` fast path keeps
    the native initial prefill and sends the same ordered decode IDs through one
    causal update. No cache tensor crosses the rollout boundary; this function creates a fresh
    ``NaiveCache`` on the trainer and lets autograd connect every reconstructed K/V
    tensor to the current policy weights. In exact mode, pure-DP workers agree on
    the largest per-sample chunk count before the first decoder call. Shorter
    traces issue bounded, cache-faithful decoder traversals so FSDP2 sees the same
    cached forward branch and backward collective depth on every rank. The dummy
    cache retains only one graph-connected K/V slot per layer and never advances
    the semantic context returned to the caller.

    The MoT inference kernels dispatch on ``module.training``.  Keep the language
    model in ``eval`` for grad-enabled replay, matching :func:`forward_flow`; do not
    restore it until backward has consumed any activation-checkpoint recomputation.
    """
    chunk_mode = validate_t2ti_replay_chunk_mode(chunk_mode)
    execution_order = validate_t2ti_replay_execution_order(execution_order)
    require(
        chunk_mode == "exact" or execution_order == "chunk_major",
        "BAGEL collapsed replay only supports chunk_major execution.",
    )
    require(
        collective_target_chunks is None or execution_order == "chunk_major",
        "BAGEL layer_major replay has no collective padding target.",
    )
    require(bool(chunks), "rebuild_text_context_from_chunks: chunks must be non-empty.")
    replay_chunks = tuple(tuple(int(token) for token in chunk) for chunk in chunks)
    for index, chunk in enumerate(replay_chunks):
        require(bool(chunk), f"rebuild_text_context_from_chunks: chunk {index} is empty.")
    if chunk_mode == "collapsed" and len(replay_chunks) > 1:
        replay_chunks = (
            replay_chunks[0],
            tuple(token for chunk in replay_chunks[1:] for token in chunk),
        )
    expected_kv_length = int(expected_kv_length)
    expected_ropes = tuple(int(x) for x in expected_ropes)
    require(bool(expected_ropes), "rebuild_text_context_from_chunks: expected_ropes must be non-empty.")

    lm = model.language_model
    was_training = bool(getattr(lm, "training", False))
    grad_enabled = torch.is_grad_enabled()
    if was_training:
        lm.eval()

    try:
        require_inference_dispatch(model)
        target_chunks = len(replay_chunks)
        if chunk_mode == "exact" and execution_order == "chunk_major":
            # Agree before the first decoder call. Computing this after the real
            # trace would already let unequal ranks desynchronize all-gathers.
            target_chunks = (
                _distributed_replay_chunk_target(len(replay_chunks), device=device)
                if collective_target_chunks is None
                else int(collective_target_chunks)
            )
            require(
                target_chunks >= len(replay_chunks),
                "rebuild_text_context_from_chunks: collective target cannot be shorter than the real trace: "
                f"{target_chunks} < {len(replay_chunks)}.",
            )
        if execution_order == "layer_major":
            ctx = _rebuild_text_context_layer_major(
                model,
                replay_chunks=replay_chunks,
                device=device,
            )
        else:
            ctx = init_und_context(model)
            for chunk in replay_chunks:
                token_ids = torch.as_tensor(chunk, dtype=torch.long)
                ctx = prefill_text_split(model, ctx, text_ids=token_ids, device=device)

        if chunk_mode == "exact" and execution_order == "chunk_major":
            dependency_zero = _pad_text_context_traversals(
                model,
                ctx,
                count=target_chunks - len(replay_chunks),
                token_id=replay_chunks[-1][-1],
                device=device,
            )
            if dependency_zero is not None:
                ctx["collective_pad_zero"] = dependency_zero

        actual_kv_length = int(ctx["kv_lens"][0])
        actual_ropes = tuple(int(x) for x in ctx["ropes"])
        require(
            actual_kv_length == expected_kv_length,
            "rebuild_text_context_from_chunks: reconstructed KV length does not match native transfer metadata: "
            f"{actual_kv_length} != {expected_kv_length}.",
        )
        require(
            actual_ropes == expected_ropes,
            "rebuild_text_context_from_chunks: reconstructed ropes do not match native transfer metadata: "
            f"{actual_ropes!r} != {expected_ropes!r}.",
        )
        return ctx
    finally:
        if was_training and not grad_enabled:
            lm.train()


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
    # Always run a FIXED ``max_new`` forwards (no early EOS break). Under an
    # FSDP-sharded MoT each forward triggers an all-gather collective; a
    # data-dependent forward count per rank (early break at a sample's own EOS)
    # desyncs the collective and deadlocks. A fixed loop makes every sample issue
    # an identical number of all-gathers — the same lockstep the Qwen-VL AR stage
    # uses. Recording stops at the first stop token, so the returned list is
    # unchanged (up to and including the stop token); forwards past it advance the
    # KV cache unrecorded.
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
        logits = lm.lm_head(out.packed_query_sequence)  # [1, vocab]
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
    """Inference-mode replay scorer: ONE grad ``forward_inference`` over ``[prompt + start +
    response[:-1]]`` attending to a (no_grad, frozen) image context ``ctx``.

    The caller prefills ONLY the image split into ``ctx`` under ``no_grad`` (frozen
    image understanding, ``is_causal=False`` as at rollout); the prompt text rides
    INSIDE this single grad forward, so the und path trains through prompt+response.
    Staying on ``forward_inference`` keeps the kernel matched to the rollout
    (``old_logp`` ratio ≈ 1), and a SINGLE grad forward keeps FSDP backward sound
    (no grad across two forwards). The last ``n`` query rows predict ``response[j]``;
    full-softmax ``log_softmax(lm_head(h)/T)`` gathered on the response tokens.
    """
    require_inference_dispatch(model)  # inference-mode replay stays in eval()
    disable_inference_cache(model)
    lm = model.language_model
    kv_len, pos = int(ctx["kv_lens"][0]), int(ctx["ropes"][0])
    n = int(response_ids.numel())
    if n == 0:
        return torch.zeros(0, dtype=torch.float32, device=device)

    response_ids = response_ids.to(device=device, dtype=torch.long)
    prompt = torch.as_tensor(prompt_ids, dtype=torch.long, device=device).reshape(-1)
    start = torch.tensor([int(start_token_id)], dtype=torch.long, device=device)
    query_ids = torch.cat([prompt, start, response_ids[:-1]], dim=0)  # [P + n]
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
    hidden = out.packed_query_sequence[-n:]  # the n response-predicting rows

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
    prompt_ids: List[int],
    image: Optional[Any],
    response_input: torch.Tensor,
    device: torch.device,
    vit_transform: Callable[[Any], Any] = lambda x: x,
) -> Dict[str, Any]:
    """Train-mode packing: one und sample ``[ViT image | prompt | response_input]`` for
    the MoT TRAINING forward (``forward_train`` layout) with a nested attention mask.

    Attention is ``full`` over the image block and ``causal`` over the text (built
    per-sample via ``prepare_attention_mask_per_sample``); the image block shares
    its rope position then prompt+response increment, matching the rollout KV build.
    ``ce_loss_indexes`` marks the response-input positions whose logits predict the
    response tokens. ``image`` is the already-``vit_transform``-ed tensor stored on
    the conditions, so ``vit_transform`` defaults to identity.
    """
    from .vendor.data.data_utils import prepare_attention_mask_per_sample

    text_ids: List[int] = []
    text_indexes: List[int] = []
    position_ids: List[int] = []
    vit_tokens = None
    vit_position_ids = None
    vit_token_indexes: List[int] = []
    vit_token_seqlens: Optional[torch.Tensor] = None
    split_lens: List[int] = []
    attn_modes: List[str] = []
    pos = 0
    rope = 0

    if image is not None:
        vit_input, _, _ = model.prepare_vit_images(
            curr_kvlens=[0],
            curr_rope=[0],
            images=[image],
            transforms=vit_transform,
            new_token_ids=new_token_ids,
        )
        img_block_len = int(vit_input["packed_seqlens"][0].item())
        text_ids.extend(int(t) for t in vit_input["packed_text_ids"].tolist())
        text_indexes.extend(int(t) for t in vit_input["packed_text_indexes"].tolist())
        position_ids.extend(int(p) for p in vit_input["packed_position_ids"].tolist())
        vit_token_indexes.extend(int(t) for t in vit_input["packed_vit_token_indexes"].tolist())
        vit_tokens = vit_input["packed_vit_tokens"].to(device=device, dtype=model.dtype)
        vit_position_ids = vit_input["packed_vit_position_ids"].to(device)
        vit_token_seqlens = vit_input["vit_token_seqlens"].to(device)
        pos = img_block_len
        rope = 1
        split_lens.append(img_block_len)
        attn_modes.append("full")

    text_block = list(prompt_ids) + [int(t) for t in response_input.tolist()]
    resp_start = pos + len(prompt_ids)
    for tid in text_block:
        text_ids.append(int(tid))
        text_indexes.append(pos)
        position_ids.append(rope)
        pos += 1
        rope += 1
    split_lens.append(len(text_block))
    attn_modes.append("causal")
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
        "packed_vit_tokens": vit_tokens,
        "packed_vit_position_ids": vit_position_ids,
        "packed_vit_token_indexes": (
            torch.tensor(vit_token_indexes, dtype=torch.long, device=device) if vit_token_indexes else None
        ),
        "vit_token_seqlens": vit_token_seqlens,
        "ce_loss_indexes": torch.tensor(ce_loss_indexes, dtype=torch.long, device=device),
    }


def und_replay_logits(model: Any, packed: Dict[str, Any]) -> torch.Tensor:
    """Train-mode grad-carrying und TRAINING forward; returns response-position logits ``[R, V]``.

    Mirrors ``Bagel.forward``'s understanding path (text embed + ViT embed → packed
    sequence → ``language_model`` MoT ``forward_train``) but returns ``lm_head``
    logits at the ce-loss (response) positions. The caller must have the language
    model in ``train()`` mode (so the navit dispatch routes ``forward_train``) and
    ``freeze_und=False``.
    """
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
