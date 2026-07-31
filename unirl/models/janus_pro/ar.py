from __future__ import annotations

import logging
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from unirl.models.types.ar import ARSamplingParams, ARStage, ARStep
from unirl.types.conditions import TextTokenCondition
from unirl.types.segments import TextSegment
from unirl.utils.dtypes import parse_torch_dtype

from .bundle import JanusProBundle
from .conditions import JanusProARConditions

logger = logging.getLogger(__name__)


@dataclass
class JanusProARParams:
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0
    stop_token_ids: List[int] = dc_field(default_factory=list)


class JanusProARStep(ARStep):
    def __init__(self, *, temperature: float = 1.0, top_p: float = 1.0, top_k: int = 0) -> None:
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)

    def step(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if logits.dim() != 2:
            raise ValueError(f"JanusProARStep.step: expected logits shape [B, vocab], got {tuple(logits.shape)}")

        if self.temperature <= 0.0:
            log_probs_full = F.log_softmax(logits.float(), dim=-1)
            token_id = log_probs_full.argmax(dim=-1)
            log_prob = log_probs_full.gather(-1, token_id.unsqueeze(-1)).squeeze(-1)
            return token_id, log_prob

        scaled = logits.float() / self.temperature
        # Match replay: store behavior log-prob under the temperature-scaled
        # full softmax before top-k/top-p truncation.
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
            sorted_vals = sorted_vals.masked_fill(cutoff > 0, float("-inf"))
            scaled = torch.full_like(scaled, float("-inf")).scatter(-1, sorted_idx, sorted_vals)

        probs = F.softmax(scaled, dim=-1)
        token_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
        log_prob = log_probs_full.gather(-1, token_id.unsqueeze(-1)).squeeze(-1)
        return token_id, log_prob


def _position_ids_from_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    return position_ids


def _language_body(transformer: torch.nn.Module) -> torch.nn.Module:
    """The bare decoder under a ``LlamaForCausalLM``-style head.

    Both AR stages read ``last_hidden_state`` and apply their own head
    (``lm_head`` for text, ``gen_head`` for image tokens), so neither wants the
    wrapper's fused full-sequence logits.
    """
    body = getattr(transformer, "model", None)
    if body is None:
        body = getattr(getattr(transformer, "module", None), "model", None)
    if body is None:
        body = getattr(getattr(transformer, "_fsdp_wrapped_module", None), "model", None)
    if body is None:
        raise AttributeError(
            "Janus-Pro AR requires a LlamaForCausalLM-style transformer exposing `.model`; "
            f"got {type(transformer).__name__}."
        )
    return body


def _output_head(transformer: torch.nn.Module) -> torch.nn.Module:
    head = transformer.get_output_embeddings()
    if head is None:
        raise AttributeError(
            "Janus-Pro AR replay requires an output head; "
            f"{type(transformer).__name__}.get_output_embeddings() returned None."
        )
    return head


def _left_repack_prompt(
    prompt: TextTokenCondition,
    images_seq_mask: torch.Tensor,
    *,
    pad_id: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if prompt.input_ids is None or prompt.attention_mask is None:
        raise ValueError("Janus-Pro AR requires prompt.input_ids and prompt.attention_mask.")

    input_ids = prompt.input_ids.to(device=device, dtype=torch.long)
    attention_mask = prompt.attention_mask.to(device=device, dtype=torch.long)
    images_seq_mask = images_seq_mask.to(device=device, dtype=torch.bool)

    lengths = attention_mask.sum(dim=1).long()
    if int(lengths.min().item()) <= 0:
        raise ValueError("Janus-Pro AR received an empty prompt row.")
    max_len = int(lengths.max().item())
    batch = int(input_ids.shape[0])

    repacked_ids = torch.full((batch, max_len), int(pad_id), dtype=torch.long, device=device)
    repacked_mask = torch.zeros((batch, max_len), dtype=torch.long, device=device)
    repacked_img_mask = torch.zeros((batch, max_len), dtype=torch.bool, device=device)

    real_mask = attention_mask.bool()
    for b in range(batch):
        real_ids = input_ids[b][real_mask[b]]
        real_img = images_seq_mask[b][real_mask[b]]
        n = int(real_ids.numel())
        repacked_ids[b, max_len - n :] = real_ids
        repacked_mask[b, max_len - n :] = 1
        repacked_img_mask[b, max_len - n :] = real_img

    return repacked_ids, repacked_mask, repacked_img_mask


class JanusProARStage(ARStage[JanusProARConditions]):
    def __init__(
        self,
        *,
        model: JanusProBundle,
        autocast_precision: str = "bf16",
        logprob_precision: str = "fp32",
    ) -> None:
        self.model = model
        self.autocast_dtype = parse_torch_dtype(autocast_precision, field_name="JanusProARStage.autocast_precision")
        self.logprob_dtype = parse_torch_dtype(logprob_precision, field_name="JanusProARStage.logprob_precision")

    def trainable_module(self) -> torch.nn.Module:
        return self.model.transformer

    def _autocast_ctx(self, device: torch.device):
        if device.type == "cuda" and self.autocast_dtype in (torch.float16, torch.bfloat16):
            return torch.autocast("cuda", self.autocast_dtype)
        from contextlib import nullcontext

        return nullcontext()

    def _prepare_prompt_embeds(
        self,
        conditions: JanusProARConditions,
        *,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if conditions.prompt is None:
            raise ValueError("JanusProARStage: conditions.prompt is None")
        if conditions.pixel_values is None or conditions.images_seq_mask is None or conditions.images_emb_mask is None:
            raise ValueError("JanusProARStage requires pixel_values, images_seq_mask, and images_emb_mask.")

        input_ids, attention_mask, images_seq_mask = _left_repack_prompt(
            conditions.prompt,
            conditions.images_seq_mask,
            pad_id=self.model.pad_token_id,
            device=device,
        )
        pixel_values = conditions.pixel_values.to(device=device, dtype=self.model.dtype)
        images_emb_mask = conditions.images_emb_mask.to(device=device, dtype=torch.bool)

        inputs_embeds = self.model.model.prepare_inputs_embeds(
            input_ids=input_ids.clone(),
            pixel_values=pixel_values,
            images_seq_mask=images_seq_mask,
            images_emb_mask=images_emb_mask,
        )
        return inputs_embeds, attention_mask

    def autoregress(
        self,
        conditions: JanusProARConditions,
        *,
        sampling_params: ARSamplingParams,
        params: Optional[JanusProARParams] = None,
        **_kwargs,
    ) -> TextSegment:
        device = next(self.model.transformer.parameters()).device
        step = JanusProARStep(
            temperature=float(sampling_params.temperature),
            top_p=float(sampling_params.top_p),
            top_k=int(sampling_params.top_k),
        )
        stop_ids = self._resolve_stop_ids(params, sampling_params)
        max_new = int(sampling_params.max_new_tokens)

        generated_tokens: List[List[int]] = []
        per_token_logps: List[List[float]] = []

        self.model.transformer.eval()
        with torch.no_grad(), self._autocast_ctx(device):
            inputs_embeds, attention_mask = self._prepare_prompt_embeds(conditions, device=device)
            batch_size = int(inputs_embeds.shape[0])
            generated_tokens = [[] for _ in range(batch_size)]
            per_token_logps = [[] for _ in range(batch_size)]
            finished = [False] * batch_size
            past_key_values = None
            next_input_ids = None
            cur_attention_mask = attention_mask

            for i in range(max_new):
                if i == 0:
                    position_ids = _position_ids_from_attention_mask(cur_attention_mask)
                    out = self.model.transformer(
                        inputs_embeds=inputs_embeds,
                        attention_mask=cur_attention_mask,
                        position_ids=position_ids,
                        use_cache=True,
                        return_dict=True,
                    )
                else:
                    position_ids = _position_ids_from_attention_mask(cur_attention_mask)[:, -1:]
                    out = self.model.transformer(
                        input_ids=next_input_ids.unsqueeze(-1),
                        attention_mask=cur_attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                    )
                past_key_values = out.past_key_values
                token_id, log_prob = step.step(out.logits[:, -1, :])
                for b in range(batch_size):
                    if finished[b]:
                        continue
                    tid = int(token_id[b].item())
                    generated_tokens[b].append(tid)
                    per_token_logps[b].append(float(log_prob[b].item()))
                    if tid in stop_ids:
                        finished[b] = True

                local_done = torch.tensor([1 if all(finished) else 0], device=device)
                if dist.is_initialized():
                    dist.all_reduce(local_done, op=dist.ReduceOp.MIN)
                if int(local_done.item()) == 1:
                    break

                next_input_ids = token_id.to(device=device, dtype=torch.long)
                cur_attention_mask = torch.cat(
                    [cur_attention_mask, torch.ones((batch_size, 1), dtype=cur_attention_mask.dtype, device=device)],
                    dim=1,
                )

        # Cached one-token decode and full-sequence teacher forcing use different
        # bf16 attention geometries, so these decode-time log-probs can put the
        # first on-policy PPO ratio outside a narrow clip range. Recipes that
        # care set `algorithm.old_logp_source: replay`, which re-anchors them
        # train-side at the exact micro geometry training replays at.
        return TextSegment.pack(
            tokens=[torch.tensor(toks, dtype=torch.long, device=device) for toks in generated_tokens],
            log_probs=[torch.tensor(lps, dtype=torch.float32, device=device) for lps in per_token_logps],
        )

    def replay(
        self,
        conditions: JanusProARConditions,
        *,
        segment: TextSegment,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        if segment.tokens is None or segment.cu_seqlens is None or segment.lengths is None:
            raise ValueError("JanusProARStage.replay: segment requires tokens with cu_seqlens")

        device = next(self.model.transformer.parameters()).device
        inputs_embeds, attention_mask = self._prepare_prompt_embeds(conditions, device=device)
        batch_size = int(inputs_embeds.shape[0])
        lengths = [int(n) for n in segment.lengths.tolist()]
        if batch_size != len(lengths):
            raise ValueError(f"JanusProARStage.replay: batch={batch_size} != segment samples={len(lengths)}")

        t_max = max(lengths) if lengths else 0
        if t_max == 0:
            return torch.zeros(0, dtype=self.logprob_dtype, device=device)

        pad_id = self.model.pad_token_id
        response_tokens = torch.full((batch_size, t_max), pad_id, dtype=torch.long, device=device)
        response_mask = torch.zeros((batch_size, t_max), dtype=torch.long, device=device)
        cu = [int(c) for c in segment.cu_seqlens.tolist()]
        for b, n in enumerate(lengths):
            if n == 0:
                continue
            response_tokens[b, :n] = segment.tokens[cu[b] : cu[b] + n].to(device=device, dtype=torch.long)
            response_mask[b, :n] = 1

        response_embeds = self.model.transformer.get_input_embeddings()(response_tokens)
        # Feed the prompt plus tokens [0, ..., T-2]: hidden position
        # ``prompt_len - 1 + k`` predicts response token ``k``, so the last
        # response token is a label only and never an input.
        full_embeds = torch.cat([inputs_embeds, response_embeds[:, :-1]], dim=1)
        full_mask = torch.cat([attention_mask, response_mask[:, :-1]], dim=1)
        position_ids = _position_ids_from_attention_mask(full_mask)

        body = _language_body(self.model.transformer)
        lm_head = _output_head(self.model.transformer)
        with self._autocast_ctx(device):
            out = body(
                inputs_embeds=full_embeds,
                attention_mask=full_mask,
                position_ids=position_ids,
                use_cache=False,
            )
            hidden = out.last_hidden_state

        prompt_len = int(inputs_embeds.shape[1])
        if int(hidden.shape[1]) != prompt_len + t_max - 1:
            raise RuntimeError(
                "JanusProARStage.replay: unexpected teacher-forced length "
                f"{hidden.shape[1]}, expected {prompt_len + t_max - 1}."
            )
        temp = float(temperature) if float(temperature) > 0.0 else 1.0
        flat: List[torch.Tensor] = []
        for b, n in enumerate(lengths):
            if n == 0:
                continue
            # Run lm_head only at this row's predict positions. Janus-Pro's
            # vocab is 102400, so a fused full-sequence forward would keep a
            # [prompt_len + T, 102400] logits tensor alive in the autograd
            # graph — ~430 MB per 2k-token sequence in bf16 — to use T rows of
            # it. Mirrors the hidden-then-head split in `image_ar.replay`.
            with self._autocast_ctx(device):
                pred_logits = lm_head(hidden[b, prompt_len - 1 : prompt_len - 1 + n, :])
            log_probs_full = F.log_softmax(pred_logits.float() / temp, dim=-1)
            flat.append(log_probs_full.gather(-1, response_tokens[b, :n].unsqueeze(-1)).squeeze(-1))

        if not flat:
            return torch.zeros(0, dtype=self.logprob_dtype, device=device)
        return torch.cat(flat, dim=0).to(dtype=self.logprob_dtype)

    def _resolve_stop_ids(
        self,
        params: Optional[JanusProARParams],
        sampling_params: ARSamplingParams,
    ) -> List[int]:
        ids: List[int] = []
        if params is not None and params.stop_token_ids:
            ids.extend(int(t) for t in params.stop_token_ids)
        if sampling_params.stop_token_id is not None:
            ids.append(int(sampling_params.stop_token_id))
        eos = getattr(self.model.tokenizer, "eos_token_id", None)
        if eos is not None:
            if isinstance(eos, (list, tuple)):
                ids.extend(int(t) for t in eos)
            else:
                ids.append(int(eos))

        seen = set()
        out = []
        for token_id in ids:
            if token_id not in seen:
                seen.add(token_id)
                out.append(token_id)
        return out


__all__ = ["JanusProARParams", "JanusProARStage", "JanusProARStep"]
