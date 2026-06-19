from __future__ import annotations

import functools
import inspect
import logging
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from unirl.models.types.ar import ARSamplingParams, ARStage, ARStep, left_pad_prompt
from unirl.types.segments import TextSegment

from .bundle import QwenVLBundle
from .conditions import QwenVLARConditions

logger = logging.getLogger(__name__)

# Qwen2.5-VL has NO flex_attention support (its mask path predates flex), so the
# only reachable sparse-block kernel here is FlashAttention (flash_attention_2/3/4,
# whichever flash-attn package is installed). flex is intentionally omitted — listing
# it would let the gate pass on a backend the model cannot actually run.
_SPARSE_PACKED_ATTN = ("flash_attention_2", "flash_attention_3", "flash_attention_4")


@functools.lru_cache(maxsize=None)
def _warn_packed_disabled(attn_impl: str) -> None:
    """One-time warning (per distinct backend) when packed replay is skipped.

    Fires when packing WOULD apply (B > 1) but the attention backend is not a
    sparse-block kernel, so replay uses the slower padded path instead.
    """
    logger.warning(
        "packed-varlen replay disabled: attn_implementation=%r is not a "
        "sparse-block kernel; using the padded replay path. Qwen2.5-VL has no "
        "flex_attention, so set attn_implementation to a FlashAttention backend "
        "('flash_attention_2'/'flash_attention_3', or 'flash_attention_4' for the "
        "pinned flash-attn-4) to enable packed replay.",
        attn_impl,
    )


def _packed_replay_supported(attn_impl: Optional[str]) -> bool:
    """Feature-detect the packed varlen replay prerequisites (review #43).

    1. A FlashAttention sparse-block backend (Qwen2.5-VL has no flex_attention);
       on plain sdpa packed attention is full O((sum L)^2) and can regress, so
       require a flash backend (checked first; warns once on fallback).
    2. transformers building a block-causal mask from restarting position_ids
       (masking_utils.find_packed_sequence_indices, transformers >= 4.53); on
       older versions the forward would silently attend ACROSS sequence
       boundaries (wrong logps, no error), so fall back to the dense path.
    """
    if attn_impl not in _SPARSE_PACKED_ATTN:
        _warn_packed_disabled(str(attn_impl))
        return False
    try:
        from transformers.masking_utils import find_packed_sequence_indices  # noqa: F401
    except Exception:
        return False
    return True


@dataclass
class QwenVLARParams:
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0
    stop_token_ids: List[int] = dc_field(default_factory=list)


class QwenVLARStep(ARStep):
    def __init__(self, *, temperature: float = 1.0, top_p: float = 1.0, top_k: int = 0) -> None:
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)

    def step(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if logits.dim() != 2:
            raise ValueError(f"QwenVLARStep.step: expected logits shape [B, vocab], got {tuple(logits.shape)}")

        if self.temperature <= 0.0:
            log_probs_full = F.log_softmax(logits.float(), dim=-1)
            token_id = log_probs_full.argmax(dim=-1)
            log_prob = log_probs_full.gather(-1, token_id.unsqueeze(-1)).squeeze(-1)
            return token_id, log_prob

        scaled = logits.float() / self.temperature
        # Behavior log-prob under the temperature-scaled distribution, matching
        # QwenVLARStage.replay's log_softmax(logits / T). MUST be computed from
        # `scaled` BEFORE the top-k/top-p masking below: replay re-applies the
        # temperature but NOT the truncation, so old_logp == replay new_logp on
        # the on-policy update -> ratio == 1 -> surrogate loss ~ 0. The prior
        # code stored the untempered log_softmax(logits), which only matched
        # replay at T == 1; at T < 1 it made ratio != 1 and trained on a
        # spurious ratio (e.g. T=0.7 drove reward 0.61 -> 0.27 in ~15 rollouts).
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


def _merge_pv(per_sample_pv: Optional[List[Optional[torch.Tensor]]]) -> Optional[torch.Tensor]:
    """Cat per-sample pixel_values into a single flat tensor for the model."""
    if per_sample_pv is None:
        return None
    parts = [pv for pv in per_sample_pv if pv is not None]
    return torch.cat(parts, dim=0) if parts else None


def _merge_igt(per_sample_igt: Optional[List[Optional[torch.Tensor]]]) -> Optional[torch.Tensor]:
    """Cat per-sample image_grid_thw into a single flat tensor for the model."""
    if per_sample_igt is None:
        return None
    parts = [igt for igt in per_sample_igt if igt is not None]
    return torch.cat(parts, dim=0) if parts else None


def _vision_rope_positions(
    transformer: Any,
    input_ids: torch.Tensor,
    *,
    image_grid_thw: Optional[torch.Tensor],
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Call Qwen2.5-VL ``get_rope_index`` across transformers versions → ``[3, bs, seq]``.

    transformers >= 5.x made ``mm_token_type_ids`` (text=0 / image=1 / video=2) a
    REQUIRED positional arg of ``get_rope_index``; <= 4.57 has no such parameter.
    Build it from the config token ids and pass it only when the installed
    signature accepts it, so both version lines work (mirrors transformers' own
    ``ProcessorMixin.create_mm_token_type_ids``).
    """
    get_rope_index = transformer.model.get_rope_index
    kwargs = {"image_grid_thw": image_grid_thw, "attention_mask": attention_mask}
    if "mm_token_type_ids" in inspect.signature(get_rope_index).parameters:
        cfg = transformer.config
        mm_token_type_ids = torch.zeros_like(input_ids)
        image_token_id = getattr(cfg, "image_token_id", None)
        video_token_id = getattr(cfg, "video_token_id", None)
        if image_token_id is not None:
            mm_token_type_ids[input_ids == image_token_id] = 1
        if video_token_id is not None:
            mm_token_type_ids[input_ids == video_token_id] = 2
        position_ids, _ = get_rope_index(input_ids, mm_token_type_ids, **kwargs)
    else:
        position_ids, _ = get_rope_index(input_ids, **kwargs)
    return position_ids


# Attention backends with a sparse packed kernel (skip cross-sequence blocks) →
# packed replay always wins. Qwen2.5-VL has NO flex support, so packed replay needs
# a FlashAttention backend (flash_attention_2/3/4, whichever flash-attn package is
# installed; the repo pins flash-attn-4 → 'flash_attention_4'). On any other backend
# (sdpa/eager) the gate falls back to the dense padded path (see packed_replay).
class QwenVLARStage(ARStage[QwenVLARConditions]):
    def __init__(self, *, model: QwenVLBundle) -> None:
        self.model = model

    def trainable_module(self) -> "torch.nn.Module":
        return self.model.transformer

    def autoregress(
        self,
        conditions: QwenVLARConditions,
        *,
        sampling_params: ARSamplingParams,
        params: Optional[QwenVLARParams] = None,
        **_kwargs: Any,
    ) -> TextSegment:
        if conditions.prompt is None or conditions.prompt.input_ids is None:
            raise ValueError("QwenVLARStage.autoregress: requires conditions.prompt.input_ids")
        if conditions.prompt.attention_mask is None:
            raise ValueError("QwenVLARStage.autoregress: requires conditions.prompt.attention_mask")

        transformer = self.model.transformer
        input_ids: torch.Tensor = conditions.prompt.input_ids
        attention_mask: torch.Tensor = conditions.prompt.attention_mask
        device = input_ids.device

        # QwenVLChatTemplateStage right-pads prompts to the in-batch max. The
        # decode loop reads ``logits[:, -1, :]`` and appends each new token at the
        # end, which is only correct when the last column is a row's last *real*
        # token — i.e. for an equal-length batch. Re-pad to LEFT so mixed-length
        # batches decode correctly too (no-op when already equal length, e.g. the
        # same-prompt-group recipe). The image placeholders shift with the real
        # prompt; ``get_rope_index`` still locates them by token id + the
        # left-padded ``attention_mask``.
        pad_id = self.model.tokenizer.pad_token_id or 0
        input_ids, attention_mask = left_pad_prompt(input_ids, attention_mask, pad_id)
        batch_size = int(input_ids.shape[0])

        # Reset stale rope_deltas from any prior forward/generate call
        transformer.model.rope_deltas = None

        stop_ids = self._resolve_stop_ids(params, sampling_params)
        step = QwenVLARStep(
            temperature=float(sampling_params.temperature),
            top_p=float(sampling_params.top_p),
            top_k=int(sampling_params.top_k),
        )
        max_new = int(sampling_params.max_new_tokens)

        # pixel_values / image_grid_thw: per-sample lists → merged tensors
        pv = _merge_pv(conditions.pixel_values)
        igt = _merge_igt(conditions.image_grid_thw)

        model_kwargs: Dict[str, Any] = {
            "attention_mask": attention_mask,
            "use_cache": True,
            "past_key_values": None,
            "cache_position": torch.arange(int(input_ids.shape[1]), device=device, dtype=torch.long),
        }

        if pv is not None:
            model_kwargs["pixel_values"] = pv
        if igt is not None:
            model_kwargs["image_grid_thw"] = igt

        cur_input_ids = input_ids

        generated_tokens: List[List[int]] = [[] for _ in range(batch_size)]
        per_token_logps: List[List[float]] = [[] for _ in range(batch_size)]
        finished = [False] * batch_size
        is_first_step = True

        for _ in range(max_new):
            prep_kwargs: Dict[str, Any] = {
                "past_key_values": model_kwargs.get("past_key_values"),
                "attention_mask": model_kwargs.get("attention_mask"),
                "cache_position": model_kwargs.get("cache_position"),
                "use_cache": True,
            }
            if is_first_step:
                if "pixel_values" in model_kwargs:
                    prep_kwargs["pixel_values"] = model_kwargs["pixel_values"]
                if "image_grid_thw" in model_kwargs:
                    prep_kwargs["image_grid_thw"] = model_kwargs["image_grid_thw"]
                prep_kwargs["is_first_iteration"] = True
            else:
                prep_kwargs["is_first_iteration"] = False

            model_inputs = transformer.prepare_inputs_for_generation(cur_input_ids, **prep_kwargs)

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
            # Synchronize finished status across all FSDP ranks.
            # If any rank still has unfinished samples, all ranks must
            # continue running forward passes (FSDP AllGather requires
            # every rank to participate).
            if all(finished):
                _local_done = torch.tensor([1], device=device)
            else:
                _local_done = torch.tensor([0], device=device)
            if dist.is_initialized():
                dist.all_reduce(_local_done, op=dist.ReduceOp.MIN)
            if _local_done.item() == 1:
                break

            cur_input_ids = torch.cat([cur_input_ids, token_id.unsqueeze(-1)], dim=1)
            model_kwargs = transformer._update_model_kwargs_for_generation(out, model_kwargs)
            model_kwargs["use_cache"] = True
            is_first_step = False

        return _pack_text_segment(generated_tokens, per_token_logps, device=device)

    def replay(
        self,
        conditions: QwenVLARConditions,
        *,
        segment: TextSegment,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Branch: prefer :meth:`packed_replay` (packed-varlen, B > 1), else
        :meth:`padding_replay` (dense padded default). Returns packed varlen
        ``[total_tokens]`` aligned with ``segment.log_probs``."""
        attn_impl = getattr(getattr(self.model.transformer, "config", None), "_attn_implementation", None)
        if _packed_replay_supported(attn_impl):
            packed = self.packed_replay(conditions, segment=segment, temperature=temperature)
            if packed is not None:
                return packed
        return self.padding_replay(conditions, segment=segment, temperature=temperature)

    def padding_replay(
        self,
        conditions: QwenVLARConditions,
        *,
        segment: TextSegment,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Dense padded ``[B, max_real + T_max]`` replay — the default / fallback path."""
        if conditions.prompt is None or conditions.prompt.input_ids is None:
            raise ValueError("QwenVLARStage.replay: conditions.prompt.input_ids is None")
        if conditions.prompt.attention_mask is None:
            raise ValueError("QwenVLARStage.replay: conditions.prompt.attention_mask is None")
        if segment.tokens is None or segment.cu_seqlens is None or segment.lengths is None:
            raise ValueError("QwenVLARStage.replay: segment requires tokens with cu_seqlens")

        # conditions.prompt (ids/mask) and pixel_values/image_grid_thw come back
        # from the SGLang rollout engine on CPU, while the trainable transformer
        # lives on this worker's CUDA device. Anchor on the model's device and
        # move the rollout-side tensors onto it so the embedding/forward index
        # ops don't hit a cpu-vs-cuda mismatch.
        device = next(self.model.transformer.parameters()).device
        prompt_ids = conditions.prompt.input_ids.to(device)
        prompt_mask = conditions.prompt.attention_mask.to(device)
        batch_size = int(prompt_ids.shape[0])

        # pixel_values / image_grid_thw: per-sample lists → merged tensors
        # The lists are already correctly sliced by Batched (CONCAT),
        # so each entry corresponds to the matching prompt row.
        pv = _merge_pv(conditions.pixel_values)
        igt = _merge_igt(conditions.image_grid_thw)
        if pv is not None:
            pv = pv.to(device)
        if igt is not None:
            igt = igt.to(device)

        # Strip right-padding introduced by TextTokenCondition.concat across
        # rollout workers.  During rollout each worker pads to its own batch
        # max; when tracks are concatenated for replay the global max adds
        # extra pad tokens that shift the logit extraction window and corrupt
        # position_ids for pad positions (text_pos=1 via masked_fill).
        real_lens = prompt_mask.sum(dim=1).long()  # [batch_size]
        max_real_len = int(real_lens.max().item())
        prompt_ids = prompt_ids[:, :max_real_len]
        prompt_mask = prompt_mask[:, :max_real_len]

        # Reset stale rope_deltas — critical for correct M-RoPE position IDs
        self.model.transformer.model.rope_deltas = None

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

        if T_max > 0:
            full_ids = torch.cat([prompt_ids, response_tokens], dim=1)
            full_mask = torch.cat([prompt_mask, response_mask], dim=1)
        else:
            full_ids = prompt_ids
            full_mask = prompt_mask

        forward_kwargs: Dict[str, Any] = {
            "input_ids": full_ids,
            "attention_mask": full_mask,
            "use_cache": False,
            "return_dict": True,
        }
        if pv is not None:
            forward_kwargs["pixel_values"] = pv
        if igt is not None:
            forward_kwargs["image_grid_thw"] = igt

        # Compute correct 4D position_ids for M-RoPE.
        # Rollout uses prepare_inputs_for_generation which produces [4, bs, seq]:
        #   row 0 = text positions (for causal mask), rows 1-3 = M-RoPE (temporal, height, width).
        # Direct forward with position_ids=None only produces [3, bs, seq] (no text_position_ids),
        # causing incorrect causal mask for multimodal inputs.
        # Fix: call get_rope_index ourselves and prepend text_positions.
        vision_pos = _vision_rope_positions(
            self.model.transformer,
            full_ids,
            image_grid_thw=igt,
            attention_mask=full_mask,
        )  # [3, bs, seq]
        text_pos = full_mask.long().cumsum(-1) - 1
        text_pos.masked_fill_(full_mask == 0, 1)
        forward_kwargs["position_ids"] = torch.cat([text_pos[None], vision_pos], dim=0)  # [4, bs, seq]

        out = self.model.transformer(**forward_kwargs)
        logits = out.logits

        if T_max == 0:
            return torch.zeros(0, dtype=torch.float32, device=device)

        # Per-sample logit extraction using real prompt lengths.
        # With right-padding, the logit at position (real_len_b - 1) correctly
        # predicts the first generated token (same context & position encoding
        # as rollout).  Subsequent generated-token logits are at contiguous
        # positions starting from max_real_len.  Positions real_len_b ..
        # max_real_len-1 are per-sample pad tokens with incorrect position
        # encoding, so their logits must be skipped.
        flat: List[torch.Tensor] = []
        for b in range(batch_size):
            n = lengths[b]
            if n == 0:
                continue
            real_len_b = int(real_lens[b].item())
            # First generated token: logit from last real prompt token
            first_logit = logits[b, real_len_b - 1 : real_len_b, :]  # [1, V]
            # Subsequent generated tokens: logits from generated-token positions
            rest_logits = logits[b, max_real_len : max_real_len + n - 1, :] if n > 1 else logits[b, :0, :]
            pred_logits_b = torch.cat([first_logit, rest_logits], dim=0)  # [n, V]
            # GRPO injects the rollout sampling temperature so replay's
            # log-softmax matches the sampling distribution (logits / T).
            log_probs_full = F.log_softmax(pred_logits_b.float() / float(temperature), dim=-1)
            per_token = log_probs_full.gather(-1, response_tokens[b, :n].unsqueeze(-1)).squeeze(-1)
            flat.append(per_token)
        if not flat:
            return torch.zeros(0, dtype=torch.float32, device=device)
        return torch.cat(flat, dim=0)

    def packed_replay(
        self,
        conditions: QwenVLARConditions,
        *,
        segment: TextSegment,
        temperature: float = 1.0,
    ) -> Optional[torch.Tensor]:
        """Packed-varlen replay for VL (M-RoPE) — verl remove_padding parity.

        Concatenate every sample's ``prompt + response`` into one zero-padded
        stream, with per-stream 4-D position ids ``[text_arange; get_rope_index
        (t,h,w)]``. The text row restarts at 0 per stream; under a FlashAttention
        backend transformers derives per-sequence ``cu_seqlens`` from that restarting
        row (varlen FA — no cross-sequence attention, verl remove_padding parity).
        sdpa/eager would instead build an explicit packed block-causal mask from the
        same row (also correct), but the gate routes those to the dense path for
        speed; Qwen2.5-VL has no flex_attention support, so a flash backend is the
        reachable packed path. Per-stream M-RoPE positions equal the standalone
        layout (validated bit-exact on transformers 4.57; the packed VL path needs a
        flash backend and re-validation on the 5.x stack — see PR notes / #54), so
        logp semantics match :meth:`padding_replay`.
        Returns ``None`` (→ padding fallback) when packing does not apply: single
        sample, or missing prompt / lengths / cu_seqlens.
        """
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
        pv = _merge_pv(conditions.pixel_values)
        igt = _merge_igt(conditions.image_grid_thw)
        if pv is not None:
            pv = pv.to(device)
        if igt is not None:
            igt = igt.to(device)

        flat_resp = segment.tokens.to(device=device, dtype=torch.long)
        lengths = [int(n) for n in segment.lengths.tolist()]
        igt_list = conditions.image_grid_thw  # per-sample list (a sample may have 0/≥1 images)
        self.model.transformer.model.rope_deltas = None  # avoid stale M-RoPE cache

        real_prompt_lens = prompt_mask.long().sum(dim=-1)  # [B] (right-padded layout)

        cu_p = [int(c) for c in segment.cu_seqlens.tolist()]
        streams: List[torch.Tensor] = []
        pos_parts: List[torch.Tensor] = []
        pred_parts: List[torch.Tensor] = []
        offset = 0
        for b in range(batch_size):
            n_p = int(real_prompt_lens[b].item())
            n_r = lengths[b]
            # The predict-index math below (offset + n_p - 1) assumes each stream has
            # >=1 real prompt token; n_p == 0 would gather the PRIOR stream's last
            # hidden state (silent cross-sequence logp corruption), so fail loud.
            assert n_p >= 1, "packed_replay: stream has 0 real prompt tokens"
            seq = torch.cat([prompt_ids[b, :n_p], flat_resp[cu_p[b] : cu_p[b] + n_r]])
            streams.append(seq)
            # Per-stream 4-D M-RoPE position [text; t; h; w]; text row restarts at
            # 0 per stream so transformers builds the packed block-causal mask
            # from it (sdpa). Per-stream get_rope_index == dense per-row (bit-exact).
            one = seq.unsqueeze(0)
            grid = igt_list[b] if (igt_list is not None and igt_list[b] is not None) else None
            if grid is not None:
                grid = grid.to(device)
            vision_pos = _vision_rope_positions(
                self.model.transformer, one, image_grid_thw=grid, attention_mask=torch.ones_like(one)
            )  # [3, 1, n]
            text_pos = torch.arange(seq.numel(), device=device).unsqueeze(0)  # [1, n]
            pos_parts.append(torch.cat([text_pos, vision_pos[:, 0, :]], dim=0))  # [4, n]
            if n_r > 0:
                pred_parts.append(torch.arange(offset + n_p - 1, offset + n_p - 1 + n_r, device=device))
            offset += int(seq.numel())
        packed_ids = torch.cat(streams).unsqueeze(0)  # [1, L]
        packed_pos = torch.cat(pos_parts, dim=1).unsqueeze(1)  # [4, L] -> [4, 1, L]
        predict_index = torch.cat(pred_parts) if pred_parts else torch.zeros(0, dtype=torch.long, device=device)

        if predict_index.numel() == 0:
            return torch.zeros(0, dtype=torch.float32, device=device)

        forward_kwargs: Dict[str, Any] = {
            "input_ids": packed_ids,
            "attention_mask": None,  # packed block-causal mask inferred from restarting text positions
            "position_ids": packed_pos,
            "use_cache": False,
            "return_dict": True,
            # Run the lm_head ONLY at the predict positions (transformers logits_to_keep
            # accepts an index tensor) so we never materialize the full [1, L, vocab]
            # logits — the packed analogue of Qwen3's chunked head. The returned logits
            # come back in predict_index order, which equals flat_resp (segment) order.
            "logits_to_keep": predict_index,
        }
        if pv is not None:
            forward_kwargs["pixel_values"] = pv
        if igt is not None:
            forward_kwargs["image_grid_thw"] = igt

        out = self.model.transformer(**forward_kwargs)
        pred_logits = out.logits[0].float()  # [T_total, V] — logits at the predict positions
        T = float(temperature) if float(temperature) > 0.0 else 1.0
        log_probs = F.log_softmax(pred_logits / T, dim=-1)
        per_token = log_probs.gather(-1, flat_resp.unsqueeze(-1)).squeeze(-1)  # [T_total]
        return per_token.to(dtype=torch.float32)

    def _resolve_stop_ids(
        self,
        params: Optional[QwenVLARParams],
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


__all__ = ["QwenVLARParams", "QwenVLARStage", "QwenVLARStep"]
