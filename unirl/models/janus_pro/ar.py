from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import List, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from unirl.models.types.ar import ARSamplingParams, ARStage, ARStep, left_pad_prompt
from unirl.types.conditions import TextTokenCondition
from unirl.types.segments import TextSegment
from unirl.utils.dtypes import parse_torch_dtype

from .bundle import JanusProBundle
from .conditions import JanusProARConditions


class JanusProARStep(ARStep):
    def __init__(self, *, temperature: float = 1.0, top_p: float = 1.0, top_k: int = 0) -> None:
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

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

        candidate_ids = None
        if self.top_k > 0 and self.top_k < scaled.shape[-1]:
            scaled, candidate_ids = torch.topk(scaled, self.top_k, dim=-1)

        if self.top_p < 1.0:
            scaled, order = torch.sort(scaled, dim=-1, descending=True)
            candidate_ids = order if candidate_ids is None else candidate_ids.gather(-1, order)
            cutoff = torch.softmax(scaled, dim=-1).cumsum(dim=-1) > self.top_p
            cutoff = torch.cat([torch.zeros_like(cutoff[..., :1]), cutoff[..., :-1]], dim=-1)
            scaled = scaled.masked_fill(cutoff, float("-inf"))

        probs = F.softmax(scaled, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1)
        token_id = sampled.squeeze(-1) if candidate_ids is None else candidate_ids.gather(-1, sampled).squeeze(-1)
        log_prob = log_probs_full.gather(-1, token_id.unsqueeze(-1)).squeeze(-1)
        return token_id, log_prob


def _position_ids_from_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    return position_ids


def _decode_attention_state(
    attention_mask: torch.Tensor,
    max_new_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    prompt_length = attention_mask.shape[1]
    full_mask = attention_mask.new_ones((attention_mask.shape[0], prompt_length + max_new_tokens))
    full_mask[:, :prompt_length] = attention_mask
    return full_mask, _position_ids_from_attention_mask(attention_mask), attention_mask.sum(dim=-1, keepdim=True)


def _autocast_ctx(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        return torch.autocast("cuda", dtype)
    return nullcontext()


@contextmanager
def _keep_fsdp_params_unsharded(model: torch.nn.Module, *, enabled: bool):
    previous = getattr(model, "_unirl_fsdp_reshard_after_forward", None)
    if not enabled or not previous:
        yield
        return

    from torch.distributed.fsdp import FSDPModule

    modules = [module for module in model.modules() if isinstance(module, FSDPModule)]
    changed = []
    try:
        for module in modules:
            module.set_reshard_after_forward(False, recurse=False)
            changed.append(module)
        yield
    finally:
        try:
            for module in reversed(changed):
                module.reshard()
        finally:
            for module in changed:
                module.set_reshard_after_forward(previous, recurse=False)


def _replay_temperature(temperature: float) -> float:
    return temperature if temperature > 0.0 else 1.0


def _left_repack_token_condition(
    prompt: TextTokenCondition,
    *,
    pad_id: int,
    device: torch.device,
    where: str = "Janus-Pro AR",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Move a prompt onto ``device`` and repack it with real tokens right-aligned."""
    if prompt.input_ids is None or prompt.attention_mask is None:
        raise ValueError(f"{where} requires prompt.input_ids and prompt.attention_mask.")

    input_ids = prompt.input_ids.to(device=device, dtype=torch.long)
    attention_mask = prompt.attention_mask.to(device=device, dtype=torch.long)
    if input_ids.shape != attention_mask.shape:
        raise ValueError(
            f"{where}: prompt.input_ids and prompt.attention_mask must have matching shapes, "
            f"got {tuple(input_ids.shape)} and {tuple(attention_mask.shape)}."
        )
    if attention_mask.sum(dim=1).min().item() <= 0:
        raise ValueError(f"{where} received an empty prompt row.")

    return left_pad_prompt(input_ids, attention_mask, pad_id)


def _left_repack_prompt(
    prompt: TextTokenCondition,
    images_seq_mask: torch.Tensor,
    *,
    pad_id: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-align prompt tokens and their image-placeholder mask together."""
    repacked_ids, repacked_mask = _left_repack_token_condition(prompt, pad_id=pad_id, device=device)

    attention_mask = prompt.attention_mask.to(device=device, dtype=torch.long)
    images_seq_mask = images_seq_mask.to(device=device, dtype=torch.bool)
    if images_seq_mask.shape != attention_mask.shape:
        raise ValueError(
            "Janus-Pro AR requires images_seq_mask to match the prompt shape; "
            f"got {tuple(images_seq_mask.shape)} and {tuple(attention_mask.shape)}."
        )
    max_len = repacked_ids.shape[1]
    repacked_img_mask = torch.zeros_like(repacked_ids, dtype=torch.bool)
    real_mask = attention_mask.bool()
    for b in range(images_seq_mask.shape[0]):
        real_img = images_seq_mask[b][real_mask[b]]
        n = real_img.numel()
        repacked_img_mask[b, max_len - n :] = real_img

    return repacked_ids, repacked_mask, repacked_img_mask


class JanusProARStage(ARStage[JanusProARConditions]):
    def __init__(
        self,
        *,
        model: JanusProBundle,
        autocast_precision: str,
        logprob_precision: str,
    ) -> None:
        self.model = model
        self.autocast_dtype = parse_torch_dtype(autocast_precision, field_name="JanusProARStage.autocast_precision")
        self.logprob_dtype = parse_torch_dtype(logprob_precision, field_name="JanusProARStage.logprob_precision")

    def trainable_module(self) -> torch.nn.Module:
        return self.model.transformer

    def _prepare_prompt_embeds(
        self,
        conditions: JanusProARConditions,
        *,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        input_ids, attention_mask, images_seq_mask = _left_repack_prompt(
            conditions.prompt,
            conditions.images_seq_mask,
            pad_id=self.model.pad_token_id,
            device=device,
        )
        images_emb_mask = conditions.images_emb_mask.to(device=device, dtype=torch.bool)
        pixel_values = conditions.pixel_values
        image_embeds = conditions.image_embeds
        if image_embeds is not None:
            image_embeds = image_embeds.to(device=device, dtype=self.model.dtype)
            pixel_values = None
        elif self.model.cache_vision_embeddings:
            pixel_values = pixel_values.to(device=device, dtype=self.model.dtype)
            with torch.no_grad():
                image_embeds = self.model.model.prepare_image_embeds(pixel_values).detach()
            conditions.image_embeds = image_embeds
            conditions.pixel_values = None
            pixel_values = None
        else:
            pixel_values = pixel_values.to(device=device, dtype=self.model.dtype)

        inputs_embeds = self.model.model.prepare_inputs_embeds(
            input_ids=input_ids.clone(),
            pixel_values=pixel_values,
            images_seq_mask=images_seq_mask,
            images_emb_mask=images_emb_mask,
            images_embeds=image_embeds,
        )
        return inputs_embeds, attention_mask

    def autoregress(
        self,
        conditions: JanusProARConditions,
        *,
        sampling_params: ARSamplingParams,
        **_kwargs,
    ) -> TextSegment:
        device = next(self.model.transformer.parameters()).device
        step = JanusProARStep(
            temperature=sampling_params.temperature,
            top_p=sampling_params.top_p,
            top_k=sampling_params.top_k,
        )
        stop_ids = self._resolve_stop_ids(sampling_params)
        max_new = sampling_params.max_new_tokens

        with (
            torch.no_grad(),
            _autocast_ctx(device, self.autocast_dtype),
            _keep_fsdp_params_unsharded(
                self.model.transformer,
                enabled=self.model.rollout_keep_params_unsharded,
            ),
        ):
            inputs_embeds, attention_mask = self._prepare_prompt_embeds(conditions, device=device)
            batch_size = inputs_embeds.shape[0]
            # Accumulate on device: reading each token back with .item() inside
            # the loop costs 2*B host syncs per step on top of the one the
            # all-ranks-done reduction already pays.
            generated_tokens = torch.zeros((batch_size, max_new), dtype=torch.long, device=device)
            generated_logps = torch.zeros((batch_size, max_new), dtype=self.logprob_dtype, device=device)
            lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            stop_ids_t = torch.tensor(stop_ids, dtype=torch.long, device=device) if stop_ids else None
            past_key_values = None
            next_input_ids = None
            full_attention_mask, prompt_position_ids, next_position_ids = _decode_attention_state(
                attention_mask,
                max_new,
            )
            prompt_length = attention_mask.shape[1]

            for i in range(max_new):
                if i == 0:
                    out = self.model.transformer(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        position_ids=prompt_position_ids,
                        use_cache=True,
                        return_dict=True,
                        logits_to_keep=1,
                    )
                else:
                    out = self.model.transformer(
                        input_ids=next_input_ids.unsqueeze(-1),
                        attention_mask=full_attention_mask[:, : prompt_length + i],
                        position_ids=next_position_ids + i - 1,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                        logits_to_keep=1,
                    )
                past_key_values = out.past_key_values
                token_id, log_prob = step.step(out.logits[:, -1, :])

                # A row that already emitted its stop token keeps decoding (FSDP
                # needs every rank in every collective) but stops recording.
                active = ~finished
                log_prob = log_prob.to(self.logprob_dtype)
                generated_tokens[:, i] = torch.where(active, token_id, torch.zeros_like(token_id))
                generated_logps[:, i] = torch.where(active, log_prob, torch.zeros_like(log_prob))
                lengths += active.long()
                if stop_ids_t is not None:
                    finished |= active & (token_id.unsqueeze(-1) == stop_ids_t).any(dim=-1)

                # All ranks must agree before breaking: FSDP all-gathers on every
                # forward, so one rank leaving early would hang the others.
                local_done = finished.all().to(dtype=torch.long).reshape(1)
                if dist.is_initialized():
                    dist.all_reduce(local_done, op=dist.ReduceOp.MIN)
                if local_done.item() == 1:
                    break

                next_input_ids = token_id

        # One host sync for the whole loop, instead of one per token per row.
        lens = lengths.tolist()
        # Cached one-token decode and full-sequence teacher forcing use different
        # bf16 attention geometries, so these decode-time log-probs can put the
        # first on-policy PPO ratio outside a narrow clip range. Recipes that
        # care set `algorithm.old_logp_source: replay`, which re-anchors them
        # train-side at the exact micro geometry training replays at.
        return TextSegment.pack(
            tokens=[generated_tokens[b, :n] for b, n in enumerate(lens)],
            log_probs=[generated_logps[b, :n] for b, n in enumerate(lens)],
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
        with _autocast_ctx(device, self.autocast_dtype):
            inputs_embeds, attention_mask = self._prepare_prompt_embeds(conditions, device=device)
        batch_size = inputs_embeds.shape[0]
        lengths = segment.lengths.tolist()
        if batch_size != len(lengths):
            raise ValueError(f"JanusProARStage.replay: batch={batch_size} != segment samples={len(lengths)}")

        t_max = max(lengths) if lengths else 0
        if t_max == 0:
            return torch.zeros(0, dtype=self.logprob_dtype, device=device)

        pad_id = self.model.pad_token_id
        response_tokens = torch.full((batch_size, t_max), pad_id, dtype=torch.long, device=device)
        response_mask = torch.zeros((batch_size, t_max), dtype=torch.long, device=device)
        cu = segment.cu_seqlens.tolist()
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

        lm_head = self.model.transformer.get_output_embeddings()
        with _autocast_ctx(device, self.autocast_dtype):
            out = self.model.transformer.model(
                inputs_embeds=full_embeds,
                attention_mask=full_mask,
                position_ids=position_ids,
                use_cache=False,
            )
            hidden = out.last_hidden_state

        prompt_len = inputs_embeds.shape[1]
        if hidden.shape[1] != prompt_len + t_max - 1:
            raise RuntimeError(
                "JanusProARStage.replay: unexpected teacher-forced length "
                f"{hidden.shape[1]}, expected {prompt_len + t_max - 1}."
            )
        temp = _replay_temperature(temperature)
        flat: List[torch.Tensor] = []
        for b, n in enumerate(lengths):
            if n == 0:
                continue
            # Run lm_head only at this row's predict positions. Janus-Pro's
            # vocab is 102400, so a fused full-sequence forward would keep a
            # [prompt_len + T, 102400] logits tensor alive in the autograd
            # graph — ~430 MB per 2k-token sequence in bf16 — to use T rows of
            # it. Mirrors the hidden-then-head split in `image_ar.replay`.
            with _autocast_ctx(device, self.autocast_dtype):
                pred_logits = lm_head(hidden[b, prompt_len - 1 : prompt_len - 1 + n, :])
            log_probs_full = F.log_softmax(pred_logits.float() / temp, dim=-1)
            flat.append(log_probs_full.gather(-1, response_tokens[b, :n].unsqueeze(-1)).squeeze(-1))

        return torch.cat(flat, dim=0).to(dtype=self.logprob_dtype)

    def _resolve_stop_ids(
        self,
        sampling_params: ARSamplingParams,
    ) -> List[int]:
        ids: List[int] = []
        if sampling_params.stop_token_id is not None:
            ids.append(sampling_params.stop_token_id)
        eos = self.model.tokenizer.eos_token_id
        if eos is not None:
            if isinstance(eos, (list, tuple)):
                ids.extend(int(t) for t in eos)
            else:
                ids.append(int(eos))

        return list(dict.fromkeys(ids))


__all__ = ["JanusProARStage", "JanusProARStep"]
