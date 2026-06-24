"""HunyuanImage3 AR stage: typed params + per-token kernel + rollout-level stage.

Four classes:

- ``HunyuanImage3ARParams`` — typed request-shape knobs (bot_task /
  max_tokens / temperature / top_p / top_k / stop_token_ids /
  cot_text / taylor_cache_*).
- ``HunyuanImage3ARState`` — per-call decode state threaded through the
  per-token loop (growing ``input_ids``, HF-style ``model_kwargs``,
  step index). AR mirror of ``HunyuanImage3DiffusionState``.
- ``HunyuanImage3ARStep`` — per-token transition kernel. Owns the model
  forward: ``init_state`` builds the KV cache + initial model_kwargs;
  ``step`` runs one token's forward + sampling + state advance;
  ``sample`` is the logits→token math kernel.
- ``HunyuanImage3ARStage`` — implements
  ``ARStage[HunyuanImage3ARConditions]``. Iterates the Step against the
  shared backbone in ``mode="gen_text"``, packs the results into a
  varlen ``TextSegment`` with ``cu_seqlens`` + per-step ``log_probs``.

PR 3 lands the **single-pass** AR autoregress. The multi-pass chain
(``bot_task ∈ {think, recaption, think_recaption, img_ratio}``) lands
in PR 4 — its outer-loop logic mirrors
``modeling_hunyuan_image_3.py:3237-3396``. Image-vocab token spans
emitted by the AR stage (the ``<img>`` splice handled at upstream
``modeling_hunyuan_image_3.py:3111``) ride in the same ``tokens``
tensor; the consumer (the diffusion stage in t2i / it2i) extracts them
via ``TextSegment.as_condition_with(reembed)`` — also wired in PR 4.

``replay()`` recomputes per-token log-probs for a stored rollout's
response tokens via a single teacher-forced forward over
``prompt + response`` (no KV cache). Used by GRPO/PPO-style training to
get gradient-flowing ``π_θ(token_t | prefix)``. Rollout's stored
log-probs (``segment.log_probs``) are full-softmax (ar.py:101-102), so
they're directly comparable to replay's output for π_old / π_θ
substitution.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from unirl.models.types.ar import ARSamplingParams, ARStage, ARStep
from unirl.types.segments import TextSegment

from .bundle import HunyuanImage3Bundle
from .conditions import HunyuanImage3ARConditions


@dataclass
class HunyuanImage3ARParams:
    """Per-request AR-mode knobs for HunyuanImage 3.0.

    Sampling defaults match the vllm-omni stage configs at
    ``vllm-omni/vllm_omni/model_executor/stage_configs/hunyuan_image3_*.yaml``.

    ``system_prompt`` / ``use_system_prompt`` mirror upstream
    ``HunyuanImage3ForCausalMM.generate_image``'s
    ``get_system_prompt(use_system_prompt, bot_task, system_prompt)``
    flow. ``use_system_prompt`` selects a built-in preset
    (``en_vanilla`` / ``en_recaption`` / ``en_think_recaption`` / ``dynamic``
    / ``None``) and ``system_prompt`` is the explicit string used when
    ``use_system_prompt='custom'`` (or as a fallback under
    ``use_system_prompt='dynamic'``). The bare HunyuanImage3 model is
    not a chat model -- gen_text without a t2i-shaped system prompt
    produces incoherent / repetitive output.
    """

    bot_task: str = "auto"  # auto | think | recaption | think_recaption | img_ratio
    max_tokens: int = 2048
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 1024
    stop_token_ids: List[int] = dc_field(default_factory=list)
    cot_text: Optional[str] = None

    # System-prompt knobs -- see class docstring.
    system_prompt: Optional[str] = None
    use_system_prompt: Optional[str] = None  # None -> read gen_config default

    # Taylor-cache acceleration knobs (forwarded to the model when supported).
    taylor_cache_interval: Optional[int] = None
    taylor_cache_order: Optional[int] = None


@dataclass
class HunyuanImage3ARState:
    """Per-call AR decode state, threaded through the per-token loop.

    Built by ``HunyuanImage3ARStep.init_state`` and advanced in place by
    each ``step`` call. Lifetime is a single ``autoregress`` call — never
    transported. AR mirror of ``HunyuanImage3DiffusionState``.
    """

    input_ids: torch.Tensor  # [B, T] long; grows by one column per step
    model_kwargs: Dict[str, Any]  # HF-style kwargs threaded across steps
    step_idx: int = 0


class HunyuanImage3ARStep(ARStep[HunyuanImage3Bundle, HunyuanImage3ARConditions, HunyuanImage3ARState]):
    """Per-token transition kernel — owns the model forward.

    ``init_state`` assembles the decode state (KV cache sized for
    ``prompt + max_new_tokens``, initial HF-style ``model_kwargs``) without
    running a forward; ``step`` performs one full token transition:
    ``prepare_inputs_for_generation`` → backbone forward in
    ``mode="gen_text"`` → next-token logits slice → ``sample`` → state
    advance. ``sample`` is the logits→token math kernel, honoring
    ``temperature`` / ``top_p`` / ``top_k`` from the construction args.
    """

    def __init__(
        self,
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> None:
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

    def sample(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample one token from a ``[B, vocab]`` logits tensor.

        Returns ``(token_id [B], log_prob [B])``. ``log_prob`` is the
        post-filter log-probability of the sampled token under the
        full softmax (so it's directly comparable to a replay-time
        full-softmax log-prob without filter masking).
        """
        if logits.dim() != 2:
            raise ValueError(f"HunyuanImage3ARStep.sample: expected logits shape [B, vocab], got {tuple(logits.shape)}")

        log_probs_full = F.log_softmax(logits.float(), dim=-1)
        scaled = logits.float() / max(self.temperature, 1e-6)

        # top-k filtering
        if self.top_k > 0 and self.top_k < scaled.shape[-1]:
            topk_vals, _ = torch.topk(scaled, self.top_k, dim=-1)
            kth = topk_vals[..., -1, None]
            scaled = torch.where(scaled < kth, torch.full_like(scaled, float("-inf")), scaled)

        # top-p filtering
        if self.top_p < 1.0:
            sorted_vals, sorted_idx = torch.sort(scaled, dim=-1, descending=True)
            cumprob = torch.softmax(sorted_vals, dim=-1).cumsum(dim=-1)
            cutoff = (cumprob > self.top_p).float()
            cutoff = torch.cat([torch.zeros_like(cutoff[..., :1]), cutoff[..., :-1]], dim=-1)
            mask = cutoff > 0
            sorted_vals = sorted_vals.masked_fill(mask, float("-inf"))
            scaled = torch.full_like(scaled, float("-inf")).scatter(-1, sorted_idx, sorted_vals)

        probs = F.softmax(scaled, dim=-1)
        token_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
        log_prob = log_probs_full.gather(-1, token_id.unsqueeze(-1)).squeeze(-1)
        return token_id, log_prob

    def init_state(
        self,
        model: HunyuanImage3Bundle,
        conditions: HunyuanImage3ARConditions,
        *,
        max_new_tokens: int,
    ) -> HunyuanImage3ARState:
        """Build the decode state for one ``autoregress`` call. No forward.

        Assumes ``conditions`` are validated (the stage is the only
        caller). Pre-builds the KV cache sized for
        ``prompt + max_new_tokens``, assembles the initial HF-style
        ``model_kwargs``, and resets the transformer's text-mode runtime
        attrs.
        """
        transformer = model.transformer
        fused = conditions.fused
        input_ids: torch.Tensor = fused.input_ids  # [B, L_prompt] long
        batch_size = int(input_ids.shape[0])

        # Pre-build a ``HunyuanStaticCache`` sized for prompt + max_new_tokens,
        # mirroring upstream ``_prepare_model_inputs`` (hunyuan.py:2326-2333).
        # Falls back to ``None`` (HF default DynamicCache) when the upstream
        # symbol isn't accessible -- e.g. fake-bundle unit tests.
        prompt_len = int(input_ids.shape[1])
        past_kv_initial = self._build_kv_cache(
            transformer, batch_size=batch_size, max_cache_len=prompt_len + int(max_new_tokens)
        )

        # i2t / it2i cond-vit fields — None for t2t. Reconstruct
        # ``vit_kwargs`` from the typed ``ImageEmbedCondition`` (the
        # upstream ViT module expects this dict shape).
        cond_vit = conditions.cond_vit
        cond_vit_images = cond_vit.embeds if cond_vit is not None else None
        vit_kwargs: Optional[Dict[str, Any]] = None
        if cond_vit is not None and (cond_vit.spatial_shapes is not None or cond_vit.attn_mask is not None):
            vit_kwargs = {
                "spatial_shapes": cond_vit.spatial_shapes,
                "attention_mask": cond_vit.attn_mask,
            }

        # i2t / it2i cond-VAE half (HI3-Instruct dual cond-image: VAE latents +
        # a cond timestep alongside the ViT patches). None for t2t. The forward
        # scatters these into the VAE <img> slots in any mode; without them the
        # 4096 VAE slots stay bare <img> embeddings → garbage comprehension.
        cond_vae = conditions.cond_vae
        cond_vae_images = cond_vae.latents if cond_vae is not None else None

        # Standard HF-style ``model_kwargs`` carried across the per-token
        # loop. Carries the rope tables, the 4D attention mask, and the
        # opaque tokenizer_output for the prefill ``_update_model_kwargs``
        # hook (which derives ``position_ids`` from ``real_pos`` for
        # right-padded batches).
        model_kwargs: Dict[str, Any] = {
            "mode": "gen_text",
            # Newer checkpoints' _update_model_kwargs_for_generation reads
            # model_kwargs["rope_image_info"] unconditionally; AR (gen_text) has
            # no image tokens, so pass an empty per-sample list.
            "rope_image_info": [[] for _ in range(batch_size)],
            "attention_mask": fused.attention_mask,  # [B, 1, L, L] bool
            "position_ids": fused.position_ids,  # [B, L] long
            "custom_pos_emb": fused.rope_cache,  # ([B, L, D], [B, L, D])
            "use_cache": True,
            "past_key_values": past_kv_initial,
            "cond_vit_images": cond_vit_images,
            "cond_vit_image_mask": fused.cond_vit_image_mask,
            "vit_kwargs": vit_kwargs,
            "cond_vae_images": cond_vae_images,
            "cond_vae_image_mask": fused.cond_vae_image_mask,
            "cond_timesteps": conditions.cond_timestep,
            "cond_timesteps_index": fused.cond_timestep_scatter_index,
        }
        if conditions.tokenizer_output is not None:
            model_kwargs["tokenizer_output"] = conditions.tokenizer_output

        # Newer checkpoints' gen_text forward reads runtime attrs off ``self`` that a
        # prior diffusion/FlowGRPO pass may leave stale (or never sets on this
        # snapshot). Reset for text generation: zero image tokens.
        transformer.post_token_len = None
        transformer.num_image_tokens = 0
        transformer.num_special_tokens = None

        return HunyuanImage3ARState(input_ids=input_ids, model_kwargs=model_kwargs)

    def step(
        self,
        model: HunyuanImage3Bundle,
        conditions: HunyuanImage3ARConditions,
        state: HunyuanImage3ARState,
    ) -> Tuple[torch.Tensor, torch.Tensor, HunyuanImage3ARState]:
        """One full token transition: forward → ``sample`` → state advance.

        Prefill (``state.step_idx == 0``) passes ``first_step=True`` to the
        backbone and gathers the predicting position from
        ``conditions.tokenizer_output.real_pos`` (right-padded batches);
        decode steps read ``logits[:, -1, :]``. Mutates ``state`` in place
        and returns ``(token_id [B], log_prob [B], state)``.
        """
        transformer = model.transformer
        device = state.input_ids.device
        batch_size = int(state.input_ids.shape[0])
        model_kwargs = state.model_kwargs

        # Cond-image scatter (i2t/it2i) only applies on the prefill: the cond
        # VAE + ViT embeds land in the full-sequence hidden states and are
        # cached, so decode steps (a single new token with no <img> positions)
        # pass None. ``_encode_cond_image``/the wrapper supply the VAE half
        # (cond_vae_images + cond_timesteps + cond_*_image_mask /
        # cond_timesteps_index) — the upstream forward asserts they come as a
        # set and scatters cond-VAE in gen_text mode too (not mode-gated).
        cond_kwargs: Dict[str, Any] = {}
        if state.step_idx == 0:
            cond_kwargs = {
                "cond_vit_images": model_kwargs.get("cond_vit_images"),
                "cond_vit_image_mask": model_kwargs.get("cond_vit_image_mask"),
                # Newer (Instruct) forward reads the ViT attn/spatial kwargs
                # under cond_vit_image_kwargs; older ones use vit_kwargs.
                "cond_vit_image_kwargs": model_kwargs.get("vit_kwargs"),
                "cond_vae_images": model_kwargs.get("cond_vae_images"),
                "cond_vae_image_mask": model_kwargs.get("cond_vae_image_mask"),
                "cond_timesteps": model_kwargs.get("cond_timesteps"),
                "cond_timesteps_index": model_kwargs.get("cond_timesteps_index"),
            }
        model_inputs = transformer.prepare_inputs_for_generation(
            state.input_ids,
            past_key_values=model_kwargs.get("past_key_values"),
            attention_mask=model_kwargs.get("attention_mask"),
            tokenizer_output=model_kwargs.get("tokenizer_output"),
            position_ids=model_kwargs["position_ids"],
            custom_pos_emb=model_kwargs["custom_pos_emb"],
            mode="gen_text",
            rope_image_info=model_kwargs.get("rope_image_info"),
            use_cache=True,
            **cond_kwargs,
        )
        with torch.no_grad():
            out = transformer(**model_inputs, first_step=(state.step_idx == 0))
        logits = getattr(out, "logits", None)
        if logits is None and isinstance(out, dict):
            logits = out.get("logits")
        if logits is None:
            raise RuntimeError("HunyuanImage3ARStep.step: model output has no .logits in mode='gen_text'.")

        # Under ``device_map="auto"`` the model's lm_head returns
        # logits on whichever shard owns it (often cuda:N≠0). Gather
        # the predicting-position slice on logits' own device, then
        # move the small ``[B, vocab]`` slice to the AR loop's home
        # device so subsequent sampling ops live on a single device.
        logits_device = logits.device
        if state.step_idx == 0 and conditions.tokenizer_output is not None:
            real_pos = getattr(conditions.tokenizer_output, "real_pos", None)
            if real_pos is not None:
                # ``real_pos`` is the *next* write position (one past
                # the last valid input token under right-padding); the
                # last valid input position is ``real_pos - 1``.
                real_pos_t = real_pos.to(device=logits_device, dtype=torch.long)
                if real_pos_t.dim() == 2:
                    real_pos_t = real_pos_t[:, -1]
                last_valid = (real_pos_t - 1).clamp(min=0, max=logits.shape[1] - 1)
                next_logits = logits[torch.arange(batch_size, device=logits_device), last_valid]
            else:
                next_logits = logits[:, -1, :]
        else:
            next_logits = logits[:, -1, :]  # [B, vocab]
        if next_logits.device != device:
            next_logits = next_logits.to(device)

        token_id, log_prob = self.sample(next_logits)

        # Advance: append the sampled token to the running input_ids, then
        # have the upstream helper advance position_ids / past_key_values.
        state.input_ids = torch.cat([state.input_ids, token_id.unsqueeze(-1)], dim=1)
        updated = transformer._update_model_kwargs_for_generation(out, model_kwargs)
        # Replace model_kwargs entirely. Upstream's
        # ``_update_model_kwargs_for_generation`` returns a *new* dict
        # that intentionally drops ``attention_mask`` and
        # ``tokenizer_output`` -- carrying the prompt's [B, 1, L, L]
        # 4D mask into decode steps would mismatch SDPA's expected
        # [B, H, q_len=1, kv_len] shape. Keep the cond_* / vit_kwargs
        # i2t/it2i pass-throughs alive across steps.
        new_kwargs: Dict[str, Any] = dict(updated)
        for carry in ("cond_vit_images", "cond_vit_image_mask", "vit_kwargs", "custom_pos_emb", "rope_image_info"):
            if carry not in new_kwargs and carry in model_kwargs:
                new_kwargs[carry] = model_kwargs[carry]
        new_kwargs["use_cache"] = True
        state.model_kwargs = new_kwargs
        state.step_idx += 1

        return token_id, log_prob, state

    @staticmethod
    def _build_kv_cache(transformer, *, batch_size: int, max_cache_len: int):
        """Pre-build a ``HunyuanStaticCache`` for the AR loop.

        Mirrors upstream ``hunyuan.py:2326-2333`` for ``mode="gen_text"``:
        ``dynamic=True`` (the cache slot count grows as new tokens land,
        bounded by ``max_cache_len``), ``dtype=bf16``. Falls back to
        ``None`` (HF default DynamicCache) when the upstream
        ``HunyuanStaticCache`` symbol isn't reachable -- e.g. fake-bundle
        unit tests where the transformer is just a stub ``nn.Module``.
        """
        import sys as _sys

        upstream_mod = _sys.modules.get(type(transformer).__module__)
        cache_cls = getattr(upstream_mod, "HunyuanStaticCache", None)
        if cache_cls is None:
            return None
        config = getattr(transformer, "config", None)
        if config is None:
            return None
        try:
            return cache_cls(
                config=config,
                batch_size=batch_size,
                max_cache_len=max_cache_len,
                dtype=torch.bfloat16,
                dynamic=True,
            )
        except Exception:  # noqa: BLE001 -- fall back to HF default cache
            return None


class HunyuanImage3ARStage(ARStage[HunyuanImage3ARConditions]):
    """Rollout-level AR stage: ``HunyuanImage3ARConditions → TextSegment``.

    Calls the shared HunyuanImage3 backbone with ``mode="gen_text"`` to
    perform autoregressive token generation. The unified-sequence input
    comes from ``conditions.fused.input_ids`` (the chat-template-built
    token sequence), with optional cond-image scatter for i2t / it2i via
    ``conditions.cond_vit`` + ``conditions.fused.cond_vit_image_mask``.

    PR 3 ships **single-pass** generation only — the ``bot_task`` knob
    in ``HunyuanImage3ARParams`` is read for stop-token selection but
    no multi-pass orchestration is performed. PR 4 lands the full
    ``think → recaption → img_ratio`` chain.
    """

    def __init__(
        self,
        *,
        model: HunyuanImage3Bundle,
    ) -> None:
        self.model = model

    def trainable_module(self) -> "torch.nn.Module":
        """Return the bare HI3 decoder — the FSDP/LoRA wrap target.

        Matches ``HunyuanImage3DiffusionStage.trainable_module`` (returns
        the same ``self.model.transformer.model`` object). HI3 is a
        unified MoE: AR (``mode='gen_text'``) and diffusion
        (``mode='gen_image'``) share the SAME decoder, so the multi-track
        builder's ``source_stage.trainable_module()`` resolves to the
        same nn.Module either way — LoRA injected via one stage is
        visible to the other.

        The HF wrapper (``HunyuanImage3ForCausalMM``) owns frozen VAE +
        ViT siblings that must NOT be FSDP-wrapped (mixed dtypes; not in
        either forward path). Returning the bare decoder under the
        wrapper avoids dragging those into the FSDP shard.
        """
        return self.model.transformer.model

    def autoregress(
        self,
        conditions: HunyuanImage3ARConditions,
        *,
        sampling_params: ARSamplingParams,
        params: Optional[HunyuanImage3ARParams] = None,
        **_kwargs: Any,
    ) -> TextSegment:
        """Run AR generation. Returns a varlen-packed ``TextSegment``.

        Iterates ``HunyuanImage3ARStep`` — which owns the per-token model
        forward — against the chat-template-built sequence carried in
        ``conditions.fused``. Required for ``tencent/HunyuanImage-3.0``
        weights — the model expects ``input_ids`` in ``mode="gen_text"``,
        not ``inputs_embeds``.

        Stop-token policy: any token in ``params.stop_token_ids`` (if
        provided) terminates that sample's generation. Falls back to
        ``sampling_params.stop_token_id`` otherwise.

        Padding note: unlike the qwen AR stages (which ``left_pad_prompt``),
        HI3 keeps the upstream right-padding and reads the prefill prediction
        at ``real_pos - 1``; decode-step ``position_ids`` are then advanced by
        the checkpoint's own ``_update_model_kwargs_for_generation`` (not
        vendored here). Mixed-length in-process batches therefore depend on
        that upstream position handling and have been validated only via the
        equal-length / two-engine (per-request, un-padded) rollout path. The
        matching ``replay`` recovers true positions from ``fused.prompt_lengths``.
        """
        fused = conditions.fused
        if fused is None or fused.input_ids is None:
            raise ValueError(
                "HunyuanImage3ARStage.autoregress: requires "
                "conditions.fused.input_ids — produced by "
                "HunyuanImage3TextEmbedStage.embed_for_ar(...)."
            )
        if fused.attention_mask is None or fused.position_ids is None or fused.rope_cache is None:
            raise ValueError(
                "HunyuanImage3ARStage.autoregress: input_ids path requires "
                "fused.attention_mask / position_ids / rope_cache to be set "
                "by HunyuanImage3TextEmbedStage.embed_for_ar(...)."
            )

        device = fused.input_ids.device
        batch_size = int(fused.input_ids.shape[0])

        stop_ids = self._resolve_stop_ids(params, sampling_params)
        step = HunyuanImage3ARStep(
            temperature=float(sampling_params.temperature),
            top_p=float(sampling_params.top_p),
            top_k=int(sampling_params.top_k),
        )
        max_new = int(sampling_params.max_new_tokens)
        state = step.init_state(self.model, conditions, max_new_tokens=max_new)

        generated_tokens: List[List[int]] = [[] for _ in range(batch_size)]
        per_token_logps: List[List[float]] = [[] for _ in range(batch_size)]
        finished = [False] * batch_size

        for _ in range(max_new):
            token_id, log_prob, state = step.step(self.model, conditions, state)
            for b in range(batch_size):
                if finished[b]:
                    continue
                tid = int(token_id[b].item())
                generated_tokens[b].append(tid)
                per_token_logps[b].append(float(log_prob[b].item()))
                if tid in stop_ids:
                    finished[b] = True
            if all(finished):
                break

        return _pack_text_segment(generated_tokens, per_token_logps, device=device)

    @staticmethod
    def _resolve_stop_ids(
        params: Optional[HunyuanImage3ARParams],
        sampling_params: ARSamplingParams,
    ) -> List[int]:
        if params is not None and params.stop_token_ids:
            return list(params.stop_token_ids)
        if sampling_params.stop_token_id is not None:
            return [int(sampling_params.stop_token_id)]
        return []

    def replay(
        self,
        conditions: HunyuanImage3ARConditions,
        *,
        segment: TextSegment,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Per-token log-prob replay over a stored rollout segment.

        One teacher-forced forward over ``prompt + response`` (no KV
        cache, no incremental loop), gather full-softmax log-probs at
        the predicting positions for each response token, return packed
        varlen ``[total_tokens]`` aligned with ``segment.log_probs``.

        Builds ``inputs_embeds`` for the forward by looking up the chat-
        template ``input_ids`` in the model's shared embedding table — exact
        for the text-only AR path (t2t / think_recaption). i2t / it2i replay
        with cond-image conditioning is **rejected** below: the cond VAE+ViT
        scatter is not re-applied here, so replaying those without it would
        compute logp on un-conditioned hidden states (a future training-side
        enhancement). The shipped two-engine recaption RL is text-only
        (``is_comprehension: false``), so this never triggers there.

        Caller controls grad / no_grad scope and ``.train()`` mode.
        Empty-response samples contribute zero tokens to the output.
        """
        fused = conditions.fused
        if fused is None or fused.input_ids is None:
            raise ValueError("HunyuanImage3ARStage.replay: conditions.fused.input_ids is None")
        if segment.tokens is None or segment.cu_seqlens is None or segment.lengths is None:
            raise ValueError(
                "HunyuanImage3ARStage.replay: segment requires tokens with "
                "framework-managed cu_seqlens (construct via TextSegment.pack)"
            )
        # Fail closed on cond-image replay: the rollout conditioned the response
        # on scattered VAE+ViT cond-image embeds, but this teacher-forced replay
        # rebuilds inputs_embeds from input_ids only (no cond scatter). Replaying
        # i2t/it2i this way would silently drop the image → wrong logp. Reject
        # until the scatter is ported here.
        if conditions.cond_vit is not None or conditions.cond_vae is not None:
            raise NotImplementedError(
                "HunyuanImage3ARStage.replay: cond-image (i2t / it2i) replay is not "
                "supported — the VAE+ViT cond scatter is not re-applied in the "
                "teacher-forced forward, so per-token logp would omit the image "
                "conditioning the rollout used. In-process comprehension/edit AR RL "
                "needs the cond-image scatter ported into replay first."
            )

        prompt_ids_padded = fused.input_ids  # [B, max_prompt_len], right-padded
        # Drive the forward on the MODEL's device, not the conditions' device:
        # the AR fused/segment come back from the engine via the transport store
        # as CPU tensors (and DP-shard keeps them on CPU), while the trainable
        # backbone lives on cuda. Using prompt_ids' device would feed CPU
        # input_ids into a cuda embedding → index_select device mismatch.
        device = self.model.transformer.model.wte.weight.device
        batch_size = int(prompt_ids_padded.shape[0])

        # Per-sample TRUE prompt lengths. The rollout sends each prompt without
        # batch padding (its own vLLM request in the two-engine adapter, or the
        # tokenizer's ``real_pos`` in the in-process ``embed_for_ar``); both
        # right-pad to ``[B, max_len]`` and carry the per-sample TRUE length in
        # ``fused.prompt_lengths`` [B]. Using the real length per sample is
        # REQUIRED — a single padded ``prompt_len`` would (1) let the response
        # attend prompt-region pad, (2) shift rope/positions (forward derives
        # them from arange over the padded length), and (3) slice the prediction
        # logits at the wrong column. We therefore replay ONE sample at a time
        # with no padding. Fail closed rather than silently fall back to the
        # padded length, which would corrupt per-token logp for short samples.
        if fused.prompt_lengths is None:
            raise ValueError(
                "HunyuanImage3ARStage.replay: fused.prompt_lengths is None. The "
                "per-sample TRUE prompt length is required to slice off the right-pad "
                "in a mixed-length batch; without it, replay would teacher-force on "
                "pad-shifted positions and silently corrupt the GRPO ratio. Populate "
                "it at rollout time — both the vLLM adapter (adapters/hi3.py) and the "
                "in-process embed_for_ar derive it from the tokenizer's real_pos."
            )
        prompt_lengths = [int(n) for n in fused.prompt_lengths.tolist()]

        resp_lengths = [int(n) for n in segment.lengths.tolist()]
        cu = [int(c) for c in segment.cu_seqlens.tolist()]

        transformer = self.model.transformer
        param_dtype = transformer.model.wte.weight.dtype
        neg_inf = torch.finfo(param_dtype).min

        flat: List[torch.Tensor] = []
        for b in range(batch_size):
            rl = resp_lengths[b]
            if rl == 0:
                continue
            pl = prompt_lengths[b]
            prompt_b = prompt_ids_padded[b, :pl].to(device=device, dtype=torch.long)
            resp_b = segment.tokens[cu[b] : cu[b] + rl].to(device=device, dtype=torch.long)
            full_ids = torch.cat([prompt_b, resp_b], dim=0).unsqueeze(0)  # [1, pl+rl]
            L_full = pl + rl

            # Pure text-only causal mask over the real (un-padded) sequence.
            causal = torch.tril(torch.ones((L_full, L_full), dtype=torch.bool, device=device))
            mask_4d = torch.full((1, 1, L_full, L_full), neg_inf, dtype=param_dtype, device=device)
            mask_4d.masked_fill_(causal.unsqueeze(0).unsqueeze(0), 0.0)

            # Reset image/rope runtime state — FlowGRPO earlier in the same
            # step sets num_image_tokens=4096; the text-only AR forward must run
            # with 0 image tokens or rope/attention indexing goes OOB → NaN. Per
            # forward because each sample's seq_len differs (forces rope rebuild).
            transformer.post_token_len = None
            transformer.num_special_tokens = None
            transformer.num_image_tokens = 0
            transformer.use_taylor_cache = False
            if hasattr(transformer, "cached_rope") and transformer.cached_rope is not None:
                for _rope_attr in ("seq_len", "rope_image_info", "cos_cache", "sin_cache"):
                    if hasattr(transformer.cached_rope, _rope_attr):
                        setattr(transformer.cached_rope, _rope_attr, None)

            out = transformer(
                input_ids=full_ids,
                attention_mask=mask_4d,
                mode="gen_text",
                past_key_values=None,
                use_cache=False,
                return_dict=True,
            )
            logits = getattr(out, "logits", None)
            if logits is None:
                raise RuntimeError("HunyuanImage3ARStage.replay: model output has no .logits")

            # logits[0, pl-1+t] predicts resp_b[t]. Use T=1 full-softmax to
            # match vLLM's recorded π_old (the [RATIO-PROBE-AR] diagnosis showed
            # vLLM logs T=1 logprobs; the old ``/temperature`` here added a
            # systematic +log(ratio_mean)≈+0.067 offset → AR ratio≈1.07. T=1 both
            # sides is the verl/OpenRLHF/TRL convention — temperature is a rollout
            # exploration knob, not part of the policy-gradient logp).
            raw_logits = logits[0, pl - 1 : pl - 1 + rl, :].float()
            log_probs_full = F.log_softmax(raw_logits, dim=-1)
            flat.append(log_probs_full.gather(-1, resp_b.unsqueeze(-1)).squeeze(-1))  # [rl], fp32

        if not flat:
            return torch.zeros(0, dtype=torch.float32, device=device)
        return torch.cat(flat, dim=0)


def _pack_text_segment(
    generated_tokens: List[List[int]],
    per_token_logps: List[List[float]],
    *,
    device: torch.device,
) -> TextSegment:
    """Pack per-sample lists of tokens / log-probs into a varlen ``TextSegment``.

    Delegates to :meth:`TextSegment.pack`, which packs the per-sample tensor
    lists along dim 0 and derives the framework-managed ``cu_seqlens``
    metadata. ``tokens`` / ``log_probs`` are packed per *token* across all
    segments, length ``sum(lengths)``; segment rows are 1:1 with samples.
    """
    return TextSegment.pack(
        tokens=[torch.tensor(toks, dtype=torch.long, device=device) for toks in generated_tokens],
        log_probs=[torch.tensor(lps, dtype=torch.float32, device=device) for lps in per_token_logps],
    )


__all__ = ["HunyuanImage3ARParams", "HunyuanImage3ARStage", "HunyuanImage3ARState", "HunyuanImage3ARStep"]
