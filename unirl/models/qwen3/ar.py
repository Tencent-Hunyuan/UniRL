"""Qwen3 AR stage: typed params + per-token kernel + rollout-level stage."""

from __future__ import annotations

import functools
import logging
from contextlib import nullcontext
from dataclasses import dataclass
from dataclasses import field as dc_field
from types import MethodType
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from unirl.models.types.ar import ARSamplingParams, ARStage, ARStep, left_pad_prompt
from unirl.models.types.replay_result import ReplayResult
from unirl.types.segments import TextSegment
from unirl.utils.dtypes import parse_torch_dtype

from .bundle import Qwen3Bundle
from .conditions import Qwen3ARConditions

logger = logging.getLogger(__name__)

_SPARSE_PACKED_ATTN = ("flex_attention", "flash_attention_2", "flash_attention_3", "flash_attention_4")


@functools.lru_cache(maxsize=None)
def _warn_packed_disabled(attn_impl: str) -> None:
    """One-time warning (per distinct backend) when packed replay is skipped."""
    logger.warning(
        "packed-varlen replay disabled: attn_implementation=%r is not a "
        "sparse-block kernel; using the padded replay path. Set "
        "attn_implementation='flex_attention' (or 'flash_attention_2' with "
        "flash_attn installed) to enable packed replay.",
        attn_impl,
    )


def _packed_replay_supported(attn_impl: Optional[str]) -> bool:
    """Feature-detect the packed varlen replay prerequisites (review #43)."""
    if attn_impl not in _SPARSE_PACKED_ATTN:
        _warn_packed_disabled(str(attn_impl))
        return False
    try:
        from transformers.masking_utils import find_packed_sequence_indices  # noqa: F401
    except Exception:
        return False
    return True


def _replay_aware_forward(
    self: Any,
    *,
    response_tokens: Optional[torch.Tensor] = None,
    prompt_len: Optional[int] = None,
    temperature: float = 1.0,
    autocast_dtype: Optional[torch.dtype] = None,
    packed_predict_index: Optional[torch.Tensor] = None,
    return_values: bool = False,
    **kw: Any,
) -> Any:
    """Dual-mode ``forward``: padded ``[B, T_max]`` fp32 log-probs, chunked so ``[B, L, vocab]`` never materializes."""
    if response_tokens is None:
        for klass in type(self).__mro__:
            f = klass.__dict__.get("forward")
            if f is not None and f is not _replay_aware_forward:
                return f(self, **kw)
        raise RuntimeError("_replay_aware_forward: no class-level forward found in the MRO")

    _require_value_head_for_replay(self, return_values)

    # Disable cuDNN SDPA because its bf16 backward can produce NaN gradients.
    if torch.cuda.is_available():
        torch.backends.cuda.enable_cudnn_sdp(False)

    autocast_ctx = (
        torch.autocast("cuda", autocast_dtype) if autocast_dtype in (torch.float16, torch.bfloat16) else nullcontext()
    )
    with autocast_ctx:
        hidden = self.model(**kw, use_cache=False, return_dict=True).last_hidden_state

    T = float(temperature) if float(temperature) > 0.0 else 1.0
    value_head = getattr(self, "value_head", None) if return_values else None

    if packed_predict_index is not None:
        h_pred = hidden[0].index_select(0, packed_predict_index)
        targets = response_tokens

        def _flat_logp_chunk(h: torch.Tensor, tok: torch.Tensor) -> torch.Tensor:
            lf = self.lm_head(h).float() / T
            return lf.gather(-1, tok.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(lf, dim=-1)

        flat_parts: List[torch.Tensor] = []
        flat_chunk = 2048
        for s in range(0, int(h_pred.size(0)), flat_chunk):
            h = h_pred[s : s + flat_chunk]
            tok = targets[s : s + flat_chunk]
            if torch.is_grad_enabled() and h.requires_grad:
                flat_parts.append(checkpoint(_flat_logp_chunk, h, tok, use_reentrant=False))
            else:
                flat_parts.append(_flat_logp_chunk(h, tok))
        if not flat_parts:
            empty = hidden.new_zeros((0,), dtype=torch.float32)
            if value_head is None:
                return empty
            return ReplayResult(log_probs=empty, values=empty)
        log_probs = torch.cat(flat_parts, dim=0)
        if value_head is None:
            return log_probs
        value_parts = [value_head(h_pred[s : s + flat_chunk]) for s in range(0, int(h_pred.size(0)), flat_chunk)]
        values = torch.cat(value_parts, dim=0) if value_parts else log_probs.new_zeros(0)
        return ReplayResult(log_probs=log_probs, values=values)
    T_max = int(response_tokens.size(1))
    resp_hidden = hidden[:, prompt_len - 1 : prompt_len - 1 + T_max, :]

    def _logp_chunk(h: torch.Tensor, tok: torch.Tensor) -> torch.Tensor:
        lf = self.lm_head(h).float() / T
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
        empty = resp_hidden.new_zeros((bsz, 0), dtype=torch.float32)
        if value_head is None:
            return empty
        return ReplayResult(log_probs=empty, values=empty)
    log_probs = torch.cat(parts, dim=1)
    if value_head is None:
        return log_probs
    value_parts = [value_head(resp_hidden[:, s : s + chunk, :]) for s in range(0, T_max, chunk)]
    values = torch.cat(value_parts, dim=1) if value_parts else log_probs.new_zeros((bsz, 0))
    return ReplayResult(log_probs=log_probs, values=values)


def _require_value_head_for_replay(model: Any, return_values: bool) -> None:
    if return_values and getattr(model, "value_head", None) is None:
        raise ValueError(
            "Qwen3 replay: return_values=True requires a value head (set use_value_head=True in the pipeline config)"
        )


def _finalize_replay_output(
    out: Union[torch.Tensor, ReplayResult],
    *,
    segment: TextSegment,
    return_values: bool,
    logprob_dtype: torch.dtype,
) -> Union[torch.Tensor, ReplayResult]:
    """Cast log-probs and flatten padded critic values to segment order."""
    if not isinstance(out, ReplayResult):
        if return_values:
            raise ValueError("Qwen3ARStage.replay: return_values=True but critic returned no values")
        return out.to(dtype=logprob_dtype)

    log_probs = out.log_probs.to(dtype=logprob_dtype)
    if not return_values:
        return log_probs
    if out.values is None:
        raise ValueError("Qwen3ARStage.replay: return_values=True but critic returned no values")
    if log_probs.ndim == 1:
        return ReplayResult(log_probs=log_probs, values=out.values.float())
    if segment.lengths is None:
        raise ValueError("Qwen3ARStage.replay: segment requires lengths to flatten critic values")

    flat_values = [out.values[b, : int(length)] for b, length in enumerate(segment.lengths.tolist()) if int(length) > 0]
    values = torch.cat(flat_values, dim=0) if flat_values else out.values.new_zeros(0)
    return ReplayResult(log_probs=log_probs, values=values.float())


@dataclass
class Qwen3ARParams:
    """Per-request AR-mode knobs for Qwen3."""

    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0
    stop_token_ids: List[int] = dc_field(default_factory=list)


class Qwen3ARStep(ARStep):
    """Per-token sampling kernel."""

    def __init__(
        self,
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> None:
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)

    def step(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if logits.dim() != 2:
            raise ValueError(f"Qwen3ARStep.step: expected logits shape [B, vocab], got {tuple(logits.shape)}")

        if self.temperature <= 0.0:
            log_probs_full = F.log_softmax(logits.float(), dim=-1)
            token_id = log_probs_full.argmax(dim=-1)
            log_prob = log_probs_full.gather(-1, token_id.unsqueeze(-1)).squeeze(-1)
            return token_id, log_prob

        scaled = logits.float() / self.temperature

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


class Qwen3ARStage(ARStage[Qwen3ARConditions]):
    """Rollout-level AR stage for Qwen3."""

    def __init__(
        self,
        *,
        model: Qwen3Bundle,
        autocast_precision: str = "bf16",
        logprob_precision: str = "fp32",
    ) -> None:
        self.model = model
        self.autocast_dtype = parse_torch_dtype(autocast_precision, field_name="Qwen3ARStage.autocast_precision")
        self.logprob_dtype = parse_torch_dtype(logprob_precision, field_name="Qwen3ARStage.logprob_precision")
        transformer = model.transformer
        if getattr(transformer.forward, "__func__", None) is not _replay_aware_forward:
            transformer.forward = MethodType(_replay_aware_forward, transformer)

    def trainable_module(self) -> "torch.nn.Module":
        """Return the HF causal LM module — the FSDP/LoRA wrap target."""
        return self.model.transformer

    def autoregress(
        self,
        conditions: Qwen3ARConditions,
        *,
        sampling_params: ARSamplingParams,
        params: Optional[Qwen3ARParams] = None,
        **_kwargs: Any,
    ) -> TextSegment:
        """Run AR generation. Returns a varlen-packed ``TextSegment``."""
        if conditions.prompt is None or conditions.prompt.input_ids is None:
            raise ValueError(
                "Qwen3ARStage.autoregress: requires conditions.prompt.input_ids — "
                "produced by Qwen3ChatTemplateStage.embed(...)."
            )
        if conditions.prompt.attention_mask is None:
            raise ValueError(
                "Qwen3ARStage.autoregress: requires conditions.prompt.attention_mask — "
                "produced by Qwen3ChatTemplateStage.embed(...)."
            )

        transformer = self.model.transformer
        input_ids: torch.Tensor = conditions.prompt.input_ids
        attention_mask: torch.Tensor = conditions.prompt.attention_mask
        device = input_ids.device

        pad_id = self.model.tokenizer.pad_token_id or 0
        input_ids, attention_mask = left_pad_prompt(input_ids, attention_mask, pad_id)
        batch_size = int(input_ids.shape[0])

        stop_ids = self._resolve_stop_ids(params, sampling_params)
        step = Qwen3ARStep(
            temperature=float(sampling_params.temperature),
            top_p=float(sampling_params.top_p),
            top_k=int(sampling_params.top_k),
        )
        max_new = int(sampling_params.max_new_tokens)

        model_kwargs = {
            "attention_mask": attention_mask,
            "use_cache": True,
            "past_key_values": None,
        }
        cur_input_ids = input_ids
        next_sequence_length = int(input_ids.shape[1])

        generated_tokens: List[List[int]] = [[] for _ in range(batch_size)]
        per_token_logps: List[List[float]] = [[] for _ in range(batch_size)]
        finished = [False] * batch_size

        for _ in range(max_new):
            model_inputs = transformer.prepare_inputs_for_generation(
                cur_input_ids,
                next_sequence_length=next_sequence_length,
                past_key_values=model_kwargs.get("past_key_values"),
                attention_mask=model_kwargs.get("attention_mask"),
                use_cache=True,
            )
            with torch.no_grad():
                out = transformer(**model_inputs, return_dict=True)
            logits = out.logits
            next_logits = logits[:, -1, :]
            if next_logits.device != device:
                next_logits = next_logits.to(device)

            token_id, log_prob = step.step(next_logits)
            for b in range(batch_size):
                if finished[b]:
                    continue
                tid = int(token_id[b].item())
                generated_tokens[b].append(tid)
                per_token_logps[b].append(float(log_prob[b].item()))
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
            model_kwargs = transformer._update_model_kwargs_for_generation(out, model_kwargs)
            model_kwargs["use_cache"] = True
            next_sequence_length = 1
        return _pack_text_segment(generated_tokens, per_token_logps, device=device)

    def replay(
        self,
        conditions: Qwen3ARConditions,
        *,
        segment: TextSegment,
        temperature: float = 1.0,
        return_values: bool = False,
    ) -> Union[torch.Tensor, ReplayResult]:
        """Per-token log-prob replay; falls back to the dense ``[B, P_max + T_max]`` :meth:`padding_replay`."""
        _require_value_head_for_replay(self.model.transformer, return_values)
        attn_impl = getattr(getattr(self.model.transformer, "config", None), "_attn_implementation", None)
        if _packed_replay_supported(attn_impl):
            packed = self.packed_replay(
                conditions,
                segment=segment,
                temperature=temperature,
                return_values=return_values,
            )
            if packed is not None:
                return packed
        return self.padding_replay(
            conditions,
            segment=segment,
            temperature=temperature,
            return_values=return_values,
        )

    def packed_replay(
        self,
        conditions: Qwen3ARConditions,
        *,
        segment: TextSegment,
        temperature: float = 1.0,
        return_values: bool = False,
    ) -> Optional[Union[torch.Tensor, ReplayResult]]:
        """Packed-varlen replay (B > 1): zero padding anywhere."""
        if conditions.prompt is None or conditions.prompt.input_ids is None or conditions.prompt.attention_mask is None:
            return None
        if segment.tokens is None or segment.cu_seqlens is None or segment.lengths is None:
            return None
        device = next(self.model.transformer.parameters()).device
        prompt_ids = conditions.prompt.input_ids.to(device)
        prompt_mask = conditions.prompt.attention_mask.to(device)
        batch_size = int(prompt_ids.shape[0])
        if batch_size <= 1:
            return None

        lengths = [int(n) for n in segment.lengths.tolist()]
        pad_id = self.model.tokenizer.pad_token_id or 0
        real_prompt_lens_p = prompt_mask.long().sum(dim=-1)

        cu_p = [int(c) for c in segment.cu_seqlens.tolist()]
        flat_resp = segment.tokens.to(device=device, dtype=torch.long)
        streams: List[torch.Tensor] = []
        pos_parts: List[torch.Tensor] = []
        pred_parts: List[torch.Tensor] = []
        offset = 0
        for b in range(batch_size):
            n_p = int(real_prompt_lens_p[b].item())
            n_r = lengths[b]
            assert n_p >= 1, "packed_replay: stream has 0 real prompt tokens"
            seq = torch.cat([prompt_ids[b, :n_p], flat_resp[cu_p[b] : cu_p[b] + n_r]])
            streams.append(seq)
            pos_parts.append(torch.arange(seq.numel(), device=device))
            if n_r > 0:
                pred_parts.append(torch.arange(offset + n_p - 1, offset + n_p - 1 + n_r, device=device))
            offset += int(seq.numel())
        packed_ids = torch.cat(streams).unsqueeze(0)
        packed_pos = torch.cat(pos_parts).unsqueeze(0)
        predict_index = torch.cat(pred_parts) if pred_parts else torch.zeros(0, dtype=torch.long, device=device)
        bucket = 1024
        L = int(packed_ids.shape[1])
        target = ((L + bucket - 1) // bucket) * bucket
        attn_impl = getattr(getattr(self.model.transformer, "config", None), "_attn_implementation", None)
        if attn_impl != "flex_attention":
            target = L
        if target > L:
            n_fill = target - L
            fill_ids = torch.full((1, n_fill), pad_id, dtype=packed_ids.dtype, device=device)
            fill_pos = torch.arange(n_fill, device=device).unsqueeze(0)
            packed_ids = torch.cat([packed_ids, fill_ids], dim=1)
            packed_pos = torch.cat([packed_pos, fill_pos], dim=1)
        per_token_flat = self.model.transformer(
            input_ids=packed_ids,
            attention_mask=None,
            position_ids=packed_pos,
            response_tokens=flat_resp,
            packed_predict_index=predict_index,
            prompt_len=0,
            temperature=temperature,
            return_values=return_values,
            autocast_dtype=(self.autocast_dtype if device.type == "cuda" else None),
        )
        return _finalize_replay_output(
            per_token_flat,
            segment=segment,
            return_values=return_values,
            logprob_dtype=self.logprob_dtype,
        )

    def padding_replay(
        self,
        conditions: Qwen3ARConditions,
        *,
        segment: TextSegment,
        temperature: float = 1.0,
        return_values: bool = False,
    ) -> Union[torch.Tensor, ReplayResult]:
        """Dense ``[B, P_max + T_max]`` padded replay — the default / fallback path."""
        if conditions.prompt is None or conditions.prompt.input_ids is None:
            raise ValueError("Qwen3ARStage.replay: conditions.prompt.input_ids is None")
        if conditions.prompt.attention_mask is None:
            raise ValueError("Qwen3ARStage.replay: conditions.prompt.attention_mask is None")
        if segment.tokens is None or segment.cu_seqlens is None or segment.lengths is None:
            raise ValueError(
                "Qwen3ARStage.replay: segment requires tokens with framework-managed "
                "cu_seqlens (construct via TextSegment.pack)"
            )

        device = next(self.model.transformer.parameters()).device
        prompt_ids = conditions.prompt.input_ids.to(device)
        prompt_mask = conditions.prompt.attention_mask.to(device)
        batch_size = int(prompt_ids.shape[0])
        prompt_len = int(prompt_ids.shape[1])

        lengths = [int(n) for n in segment.lengths.tolist()]
        T_max = max(lengths) if lengths else 0
        pad_id = self.model.tokenizer.pad_token_id or 0

        response_tokens = torch.full((batch_size, T_max), pad_id, dtype=torch.long, device=device)
        response_mask = torch.zeros((batch_size, T_max), dtype=torch.long, device=device)
        cu = [int(c) for c in segment.cu_seqlens.tolist()]
        for b in range(batch_size):
            n = lengths[b]
            if n == 0:
                continue
            response_tokens[b, :n] = segment.tokens[cu[b] : cu[b] + n].to(device=device, dtype=torch.long)
            response_mask[b, :n] = 1

        real_prompt_lens = prompt_mask.long().sum(dim=-1)
        if int(real_prompt_lens.min().item()) < prompt_len:
            left_padded_ids = torch.full_like(prompt_ids, pad_id)
            left_padded_mask = torch.zeros_like(prompt_mask)
            for b in range(batch_size):
                n_real = int(real_prompt_lens[b].item())
                if n_real == 0:
                    continue
                left_padded_ids[b, prompt_len - n_real :] = prompt_ids[b, :n_real]
                left_padded_mask[b, prompt_len - n_real :] = 1
            prompt_ids = left_padded_ids
            prompt_mask = left_padded_mask

        # Trim left padding to each batch's true maximum prompt length.
        max_real_prompt = int(real_prompt_lens.max().item())
        if 0 < max_real_prompt < prompt_len:
            prompt_ids = prompt_ids[:, prompt_len - max_real_prompt :]
            prompt_mask = prompt_mask[:, prompt_len - max_real_prompt :]
            prompt_len = max_real_prompt

        if T_max > 0:
            full_ids = torch.cat([prompt_ids, response_tokens], dim=1)
            full_mask = torch.cat([prompt_mask, response_mask], dim=1)
        else:
            full_ids = prompt_ids
            full_mask = prompt_mask

        position_ids = (full_mask.long().cumsum(dim=-1) - 1).clamp(min=0)

        out = self.model.transformer(
            input_ids=full_ids,
            attention_mask=full_mask,
            position_ids=position_ids,
            response_tokens=response_tokens,
            prompt_len=prompt_len,
            temperature=temperature,
            return_values=return_values,
            autocast_dtype=(self.autocast_dtype if device.type == "cuda" else None),
        )

        finalized = _finalize_replay_output(
            out,
            segment=segment,
            return_values=return_values,
            logprob_dtype=self.logprob_dtype,
        )
        if isinstance(finalized, ReplayResult):
            per_token = finalized.log_probs
            values = finalized.values
        else:
            per_token = finalized
            values = None

        flat_log_probs: List[torch.Tensor] = []
        for b, n in enumerate(lengths):
            if n > 0:
                flat_log_probs.append(per_token[b, :n])
        log_probs = (
            torch.cat(flat_log_probs, dim=0)
            if flat_log_probs
            else torch.zeros(0, dtype=self.logprob_dtype, device=device)
        )
        if not return_values:
            return log_probs
        if values is None:
            raise ValueError("Qwen3ARStage.replay: return_values=True but critic returned no values")
        return ReplayResult(log_probs=log_probs, values=values)

    def _resolve_stop_ids(
        self,
        params: Optional[Qwen3ARParams],
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
    """Pack per-sample lists of tokens / log-probs into a varlen ``TextSegment``."""
    return TextSegment.pack(
        tokens=[torch.tensor(toks, dtype=torch.long, device=device) for toks in generated_tokens],
        log_probs=[torch.tensor(lps, dtype=torch.float32, device=device) for lps in per_token_logps],
    )


__all__ = ["Qwen3ARParams", "Qwen3ARStage", "Qwen3ARStep"]
