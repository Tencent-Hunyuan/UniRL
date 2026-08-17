"""Qwen3.5 AR stage: typed params + per-token kernel + rollout-level stage."""

from __future__ import annotations

import inspect
import logging
from contextlib import nullcontext
from dataclasses import dataclass
from dataclasses import field as dc_field
from types import MethodType
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from unirl.models.types.ar import ARSamplingParams, ARStage, ARStep, left_pad_prompt
from unirl.types.segments import TextSegment
from unirl.utils.dtypes import parse_torch_dtype

from .bundle import Qwen3_5Bundle
from .conditions import Qwen3_5ARConditions

logger = logging.getLogger(__name__)

_SPARSE_PACKED_ATTN: Tuple[str, ...] = ()


def _reserved_generation_token_ids(transformer: Any) -> Tuple[int, ...]:
    """Image/video placeholders are prompt structure, never response actions."""
    config = transformer.config
    ids = []
    for name in ("image_token_id", "video_token_id"):
        token_id = getattr(config, name, None)
        if token_id is not None:
            ids.append(int(token_id))
    return tuple(dict.fromkeys(ids))


def _replay_aware_forward(
    self: Any,
    *,
    response_tokens: Optional[torch.Tensor] = None,
    prompt_len: Optional[int] = None,
    temperature: float = 1.0,
    autocast_dtype: Optional[torch.dtype] = None,
    **kw: Any,
) -> Any:
    """Dual-mode ``forward``: padded ``[B, T_max]`` fp32 log-probs, chunked so ``[B, L, vocab]`` never materializes."""
    if response_tokens is None:
        for klass in type(self).__mro__:
            f = klass.__dict__.get("forward")
            if f is not None and f is not _replay_aware_forward:
                return f(self, **kw)
        raise RuntimeError("_replay_aware_forward: no class-level forward found in the MRO")

    if torch.cuda.is_available():
        torch.backends.cuda.enable_cudnn_sdp(False)

    autocast_ctx = (
        torch.autocast("cuda", autocast_dtype) if autocast_dtype in (torch.float16, torch.bfloat16) else nullcontext()
    )
    with autocast_ctx:
        hidden = self.model(**kw, use_cache=False, return_dict=True).last_hidden_state

    T = float(temperature) if float(temperature) > 0.0 else 1.0

    T_max = int(response_tokens.size(1))
    resp_hidden = hidden[:, prompt_len - 1 : prompt_len - 1 + T_max, :]
    forbidden_ids = _reserved_generation_token_ids(self)

    def _logp_chunk(h: torch.Tensor, tok: torch.Tensor) -> torch.Tensor:
        lf = self.lm_head(h).float() / T
        if forbidden_ids:
            forbidden = torch.tensor(forbidden_ids, device=lf.device, dtype=torch.long)
            lf = lf.index_fill(-1, forbidden, float("-inf"))
        chosen = lf.gather(-1, tok.unsqueeze(-1)).squeeze(-1)
        return chosen - torch.logsumexp(lf, dim=-1)

    bsz = resp_hidden.size(0)
    chunk = max(64, 2048 // max(1, bsz))
    parts: List[torch.Tensor] = []
    for s in range(0, T_max, chunk):
        h = resp_hidden[:, s : s + chunk, :]
        tok = response_tokens[:, s : s + chunk]
        if torch.is_grad_enabled() and h.requires_grad:
            parts.append(checkpoint(_logp_chunk, h, tok, use_reentrant=False))
        else:
            parts.append(_logp_chunk(h, tok))
    if not parts:
        return resp_hidden.new_zeros((bsz, 0), dtype=torch.float32)
    return torch.cat(parts, dim=1)


@dataclass
class Qwen3_5ARParams:
    """Per-request AR-mode knobs for Qwen3.5."""

    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0
    stop_token_ids: List[int] = dc_field(default_factory=list)


class Qwen3_5ARStep(ARStep):
    """Per-token sampling kernel (mirrors Qwen3ARStep)."""

    def __init__(
        self,
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
        forbidden_token_ids: Tuple[int, ...] = (),
    ) -> None:
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.forbidden_token_ids = tuple(int(token_id) for token_id in forbidden_token_ids)

    def step(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if logits.dim() != 2:
            raise ValueError(f"Qwen3_5ARStep.step: expected logits shape [B, vocab], got {tuple(logits.shape)}")

        scaled = logits.float()
        if self.forbidden_token_ids:
            forbidden = torch.tensor(self.forbidden_token_ids, device=scaled.device, dtype=torch.long)
            scaled = scaled.index_fill(-1, forbidden, float("-inf"))

        if self.temperature <= 0.0:
            log_probs_full = F.log_softmax(scaled, dim=-1)
            token_id = log_probs_full.argmax(dim=-1)
            log_prob = log_probs_full.gather(-1, token_id.unsqueeze(-1)).squeeze(-1)
            return token_id, log_prob

        scaled = scaled / self.temperature
        log_probs_full = F.log_softmax(scaled, dim=-1)

        if self.top_k > 0 and self.top_k < scaled.shape[-1]:
            topk_vals, _ = torch.topk(scaled, self.top_k, dim=-1)
            kth = topk_vals[..., -1, None]
            scaled = torch.where(scaled < kth, torch.full_like(scaled, float("-inf")), scaled)

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


def _merge_per_sample(
    per_sample: Optional[List[Optional[torch.Tensor]]],
) -> Optional[torch.Tensor]:
    """Cat per-sample media tensors into a single flat tensor for the model."""
    if per_sample is None:
        return None
    parts = [t for t in per_sample if t is not None]
    return torch.cat(parts, dim=0) if parts else None


def _vision_rope_positions(
    transformer: Any,
    input_ids: torch.Tensor,
    *,
    image_grid_thw: Optional[torch.Tensor],
    attention_mask: torch.Tensor,
    mm_token_type_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Call Qwen3.5 ``get_rope_index`` → ``[3, bs, seq]`` (t, h, w)."""
    get_rope_index = transformer.model.get_rope_index
    cfg = transformer.config
    if mm_token_type_ids is None:
        mm_token_type_ids = torch.zeros_like(input_ids)
        image_token_id = getattr(cfg, "image_token_id", None)
        if image_token_id is not None:
            mm_token_type_ids[input_ids == image_token_id] = 1
    rope_parameters = inspect.signature(get_rope_index).parameters
    if "mm_token_type_ids" in rope_parameters:
        position_ids, _ = get_rope_index(
            input_ids,
            mm_token_type_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )
    else:
        position_ids, _ = get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )
    return position_ids


def _build_mm_token_type_ids(
    transformer: Any,
    input_ids: torch.Tensor,
) -> torch.Tensor:
    """Build ``mm_token_type_ids`` from token ids (text=0 / image=1)."""
    cfg = transformer.config
    mm = torch.zeros_like(input_ids)
    image_token_id = getattr(cfg, "image_token_id", None)
    if image_token_id is not None:
        mm[input_ids == image_token_id] = 1
    return mm


def _dense_flash_attention_kwargs(input_ids: torch.Tensor) -> Dict[str, Any]:
    """Flash-attention sequence metadata for dense ``[B, L]`` replay tensors."""
    if input_ids.dim() != 2:
        raise ValueError(f"_dense_flash_attention_kwargs expects [B, L] input_ids, got {tuple(input_ids.shape)}")
    batch_size, seq_len = int(input_ids.shape[0]), int(input_ids.shape[1])
    cu = torch.arange(batch_size + 1, device=input_ids.device, dtype=torch.int32) * int(seq_len)
    return {
        "cu_seq_lens_q": cu,
        "cu_seq_lens_k": cu,
        "max_length_q": int(seq_len),
        "max_length_k": int(seq_len),
    }


class Qwen3_5ARStage(ARStage[Qwen3_5ARConditions]):
    """Rollout-level AR stage for Qwen3.5 (dense + MoE, hybrid GDN attention)."""

    def __init__(
        self,
        *,
        model: Qwen3_5Bundle,
        autocast_precision: str = "bf16",
        logprob_precision: str = "fp32",
    ) -> None:
        self.model = model
        self.autocast_dtype = parse_torch_dtype(autocast_precision, field_name="Qwen3_5ARStage.autocast_precision")
        self.logprob_dtype = parse_torch_dtype(logprob_precision, field_name="Qwen3_5ARStage.logprob_precision")
        transformer = model.transformer
        if getattr(transformer.forward, "__func__", None) is not _replay_aware_forward:
            transformer.forward = MethodType(_replay_aware_forward, transformer)

    def trainable_module(self) -> "torch.nn.Module":
        return self.model.transformer

    def autoregress(
        self,
        conditions: Qwen3_5ARConditions,
        *,
        sampling_params: ARSamplingParams,
        params: Optional[Qwen3_5ARParams] = None,
        **_kwargs: Any,
    ) -> TextSegment:
        if conditions.prompt is None or conditions.prompt.input_ids is None:
            raise ValueError("Qwen3_5ARStage.autoregress: requires conditions.prompt.input_ids")
        if conditions.prompt.attention_mask is None:
            raise ValueError("Qwen3_5ARStage.autoregress: requires conditions.prompt.attention_mask")

        transformer = self.model.transformer
        input_ids: torch.Tensor = conditions.prompt.input_ids
        attention_mask: torch.Tensor = conditions.prompt.attention_mask
        device = input_ids.device

        pad_id = self.model.tokenizer.pad_token_id or 0
        input_ids, attention_mask = left_pad_prompt(input_ids, attention_mask, pad_id)
        batch_size = int(input_ids.shape[0])

        transformer.model.rope_deltas = None

        stop_ids = self._resolve_stop_ids(params, sampling_params)
        forbidden_ids = _reserved_generation_token_ids(transformer)
        step = Qwen3_5ARStep(
            temperature=float(sampling_params.temperature),
            top_p=float(sampling_params.top_p),
            top_k=int(sampling_params.top_k),
            forbidden_token_ids=forbidden_ids,
        )
        max_new = int(sampling_params.max_new_tokens)

        pv = _merge_per_sample(conditions.pixel_values)
        igt = _merge_per_sample(conditions.image_grid_thw)
        if pv is not None:
            pv = pv.to(device)
        if igt is not None:
            igt = igt.to(device)

        use_cache = not callable(getattr(transformer, "get_parallel_plan", None))

        model_kwargs: Dict[str, Any] = {
            "attention_mask": attention_mask,
            "use_cache": use_cache,
            "past_key_values": None,
        }
        if use_cache and pv is not None:
            model_kwargs["image_grid_thw"] = igt
            model_kwargs["mm_token_type_ids"] = _build_mm_token_type_ids(transformer, input_ids)
            model_kwargs["position_ids"] = transformer._prepare_position_ids_for_generation(
                input_ids,
                model_kwargs,
            )

        cur_input_ids = input_ids
        next_sequence_length = int(input_ids.shape[1])
        generated_tokens: List[List[int]] = [[] for _ in range(batch_size)]
        per_token_logps: List[List[float]] = [[] for _ in range(batch_size)]
        finished = [False] * batch_size
        is_first_step = True

        for _ in range(max_new):
            if use_cache:
                prep_kwargs: Dict[str, Any] = {
                    "past_key_values": model_kwargs.get("past_key_values"),
                    "attention_mask": model_kwargs.get("attention_mask"),
                    "position_ids": model_kwargs.get("position_ids"),
                    "mm_token_type_ids": model_kwargs.get("mm_token_type_ids"),
                    "next_sequence_length": next_sequence_length,
                    "use_cache": True,
                }
                if is_first_step:
                    if pv is not None:
                        prep_kwargs["pixel_values"] = pv
                    if igt is not None:
                        prep_kwargs["image_grid_thw"] = igt
                    prep_kwargs["is_first_iteration"] = True
                else:
                    prep_kwargs["is_first_iteration"] = False

                model_inputs = transformer.prepare_inputs_for_generation(cur_input_ids, **prep_kwargs)
            else:
                model_inputs = {
                    "input_ids": cur_input_ids,
                    "attention_mask": model_kwargs["attention_mask"],
                    "use_cache": False,
                }
                if pv is not None:
                    model_inputs["pixel_values"] = pv
                if igt is not None:
                    model_inputs["image_grid_thw"] = igt
                if pv is not None:
                    model_inputs["mm_token_type_ids"] = _build_mm_token_type_ids(transformer, cur_input_ids)
                    vision_pos = _vision_rope_positions(
                        transformer,
                        cur_input_ids,
                        image_grid_thw=igt,
                        attention_mask=model_kwargs["attention_mask"],
                        mm_token_type_ids=model_inputs["mm_token_type_ids"],
                    )
                    text_pos = model_kwargs["attention_mask"].long().cumsum(-1) - 1
                    text_pos.masked_fill_(model_kwargs["attention_mask"] == 0, 1)
                    model_inputs["position_ids"] = torch.cat([text_pos[None], vision_pos], dim=0)
                model_inputs.update(_dense_flash_attention_kwargs(cur_input_ids))

            with torch.no_grad():
                out = transformer(**model_inputs, return_dict=True)
            logits = out.logits
            next_logits = logits[:, -1, :]
            if next_logits.device != device:
                next_logits = next_logits.to(device)

            token_id, log_prob = step.step(next_logits)
            token_list = token_id.tolist()
            logp_list = log_prob.tolist()
            for b in range(batch_size):
                if finished[b]:
                    continue
                tid = int(token_list[b])
                generated_tokens[b].append(tid)
                per_token_logps[b].append(float(logp_list[b]))
                if tid in stop_ids:
                    finished[b] = True

            local_done = all(finished)
            if dist.is_initialized() and dist.get_world_size() > 1:
                done = torch.tensor([1 if local_done else 0], device=device)
                dist.all_reduce(done, op=dist.ReduceOp.MIN)
                local_done = done.item() == 1
            if local_done:
                break

            cur_input_ids = torch.cat([cur_input_ids, token_id.unsqueeze(-1)], dim=1)
            if use_cache:
                model_kwargs = transformer._update_model_kwargs_for_generation(out, model_kwargs)
                model_kwargs["use_cache"] = True
                next_sequence_length = 1
            else:
                next_mask = torch.ones(
                    (batch_size, 1),
                    dtype=model_kwargs["attention_mask"].dtype,
                    device=device,
                )
                model_kwargs["attention_mask"] = torch.cat(
                    [model_kwargs["attention_mask"], next_mask],
                    dim=1,
                )
            is_first_step = False

        return _pack_text_segment(generated_tokens, per_token_logps, device=device)

    def replay(
        self,
        conditions: Qwen3_5ARConditions,
        *,
        segment: TextSegment,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Replay per-token log-probs; returns packed varlen ``[total_tokens]``."""
        return self.padding_replay(conditions, segment=segment, temperature=temperature)

    def padding_replay(
        self,
        conditions: Qwen3_5ARConditions,
        *,
        segment: TextSegment,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Dense ``[B, P_max + T_max]`` padded replay with chunked logp."""
        if conditions.prompt is None or conditions.prompt.input_ids is None:
            raise ValueError("Qwen3_5ARStage.padding_replay: conditions.prompt.input_ids is None")
        if conditions.prompt.attention_mask is None:
            raise ValueError("Qwen3_5ARStage.padding_replay: conditions.prompt.attention_mask is None")
        if segment.tokens is None or segment.cu_seqlens is None or segment.lengths is None:
            raise ValueError("Qwen3_5ARStage.padding_replay: segment requires tokens with cu_seqlens")

        device = next(self.model.transformer.parameters()).device
        prompt_ids = conditions.prompt.input_ids.to(device)
        prompt_mask = conditions.prompt.attention_mask.to(device)
        batch_size = int(prompt_ids.shape[0])

        pv = _merge_per_sample(conditions.pixel_values)
        igt = _merge_per_sample(conditions.image_grid_thw)
        if pv is not None:
            pv = pv.to(device)
        if igt is not None:
            igt = igt.to(device)

        real_lens = prompt_mask.sum(dim=1).long()
        max_real_len = int(real_lens.max().item())
        prompt_ids = prompt_ids[:, :max_real_len]
        prompt_mask = prompt_mask[:, :max_real_len]

        pad_id = self.model.tokenizer.pad_token_id or 0

        if int(real_lens.min().item()) < max_real_len:
            left_padded_ids = torch.full_like(prompt_ids, pad_id)
            left_padded_mask = torch.zeros_like(prompt_mask)
            for b in range(batch_size):
                n_real = int(real_lens[b].item())
                if n_real == 0:
                    continue
                left_padded_ids[b, max_real_len - n_real :] = prompt_ids[b, :n_real]
                left_padded_mask[b, max_real_len - n_real :] = 1
            prompt_ids = left_padded_ids
            prompt_mask = left_padded_mask

        self.model.transformer.model.rope_deltas = None

        lengths = [int(n) for n in segment.lengths.tolist()]
        T_max = max(lengths) if lengths else 0
        response_tokens = torch.full((batch_size, T_max), pad_id, dtype=torch.long, device=device)
        response_mask = torch.zeros((batch_size, T_max), dtype=torch.long, device=device)
        cu = [int(c) for c in segment.cu_seqlens.tolist()]
        for b in range(batch_size):
            n = lengths[b]
            if n == 0:
                continue
            response_tokens[b, :n] = segment.tokens[cu[b] : cu[b] + n].to(device=device, dtype=torch.long)
            response_mask[b, :n] = 1

        for token_id in _reserved_generation_token_ids(self.model.transformer):
            if bool(((response_tokens == token_id) & response_mask.bool()).any().item()):
                raise ValueError(
                    "Qwen3_5ARStage.padding_replay: response contains reserved "
                    f"multimodal placeholder token id {token_id}; refusing to "
                    "rewrite the sampled action. Regenerate with placeholder "
                    "logit bias enabled."
                )

        if T_max > 0:
            full_ids = torch.cat([prompt_ids, response_tokens], dim=1)
            full_mask = torch.cat([prompt_mask, response_mask], dim=1)
        else:
            full_ids = prompt_ids
            full_mask = prompt_mask

        # Build mm_token_type_ids from the PROMPT only. Response tokens are sampled text and must be type 0.
        cfg = self.model.transformer.config
        mm_token_type_ids = torch.zeros_like(full_ids)
        image_token_id = getattr(cfg, "image_token_id", None)
        if image_token_id is not None:
            mm_token_type_ids[:, :max_real_len][prompt_ids == image_token_id] = 1

        vision_pos = _vision_rope_positions(
            self.model.transformer,
            full_ids,
            image_grid_thw=igt,
            attention_mask=full_mask,
            mm_token_type_ids=mm_token_type_ids,
        )
        text_pos = full_mask.long().cumsum(-1) - 1
        text_pos.masked_fill_(full_mask == 0, 1)
        position_ids = torch.cat([text_pos[None], vision_pos], dim=0)

        forward_kwargs: Dict[str, Any] = {
            "input_ids": full_ids,
            "attention_mask": full_mask,
            "position_ids": position_ids,
            "response_tokens": response_tokens,
            "prompt_len": max_real_len,
            "temperature": temperature,
            "autocast_dtype": (self.autocast_dtype if device.type == "cuda" else None),
        }
        forward_kwargs.update(_dense_flash_attention_kwargs(full_ids))
        if pv is not None:
            forward_kwargs["pixel_values"] = pv
        if igt is not None:
            forward_kwargs["image_grid_thw"] = igt

        per_token = self.model.transformer(**forward_kwargs)

        if T_max == 0:
            return torch.zeros(0, dtype=self.logprob_dtype, device=device)

        flat: List[torch.Tensor] = []
        for b in range(batch_size):
            n = lengths[b]
            if n == 0:
                continue
            flat.append(per_token[b, :n])
        if not flat:
            return torch.zeros(0, dtype=self.logprob_dtype, device=device)
        return torch.cat(flat, dim=0).to(dtype=self.logprob_dtype)

    def _resolve_stop_ids(
        self,
        params: Optional[Qwen3_5ARParams],
        sampling_params: ARSamplingParams,
    ) -> List[int]:
        ids: List[int] = []
        if params is not None and params.stop_token_ids:
            ids.extend(int(t) for t in params.stop_token_ids)
        if sampling_params.stop_token_id is not None:
            ids.append(int(sampling_params.stop_token_id))
        eos = self.model.tokenizer.eos_token_id
        if eos is not None:
            if isinstance(eos, (list, tuple)):
                ids.extend(int(t) for t in eos)
            else:
                ids.append(int(eos))
        seen: set = set()
        out: List[int] = []
        for t in ids:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out


def _pack_text_segment(
    generated_tokens: List[List[int]],
    per_token_logps: List[List[float]],
    *,
    device: torch.device,
) -> TextSegment:
    return TextSegment.pack(
        tokens=[torch.tensor(toks, dtype=torch.long, device=device) for toks in generated_tokens],
        log_probs=[torch.tensor(lps, dtype=torch.float32, device=device) for lps in per_token_logps],
    )


__all__ = ["Qwen3_5ARParams", "Qwen3_5ARStage", "Qwen3_5ARStep"]
