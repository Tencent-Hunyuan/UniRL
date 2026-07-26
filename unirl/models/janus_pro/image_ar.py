from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from unirl.models.types.ar import ARSamplingParams, ARStage
from unirl.types.conditions import TextTokenCondition
from unirl.types.primitives import Images
from unirl.types.segments import TextSegment
from unirl.utils.dtypes import parse_torch_dtype

from .ar import JanusProARStep, _position_ids_from_attention_mask
from .bundle import JanusProBundle
from .conditions import JanusProImageARConditions


@dataclass
class JanusProImageARSamplingParams(ARSamplingParams):
    """Sampling parameters for Janus-Pro autoregressive image tokens."""

    temperature: float = 1.0
    max_new_tokens: int = 576
    top_p: float = 1.0
    top_k: int = 0
    cfg_weight: float = 5.0
    img_size: int = 384
    width: Optional[int] = None
    height: Optional[int] = None
    patch_size: int = 16


def _left_repack_token_condition(
    prompt: TextTokenCondition,
    *,
    pad_id: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if prompt.input_ids is None or prompt.attention_mask is None:
        raise ValueError("JanusProImageARStage requires prompt.input_ids and prompt.attention_mask.")

    input_ids = prompt.input_ids.to(device=device, dtype=torch.long)
    attention_mask = prompt.attention_mask.to(device=device, dtype=torch.long)
    if input_ids.shape != attention_mask.shape:
        raise ValueError(
            "JanusProImageARStage: prompt.input_ids and prompt.attention_mask must have matching shapes, "
            f"got {tuple(input_ids.shape)} and {tuple(attention_mask.shape)}."
        )

    lengths = attention_mask.sum(dim=1).long()
    if int(lengths.min().item()) <= 0:
        raise ValueError("JanusProImageARStage received an empty prompt row.")
    max_len = int(lengths.max().item())
    batch = int(input_ids.shape[0])

    repacked_ids = torch.full((batch, max_len), int(pad_id), dtype=torch.long, device=device)
    repacked_mask = torch.zeros((batch, max_len), dtype=torch.long, device=device)
    real_mask = attention_mask.bool()
    for b in range(batch):
        real_ids = input_ids[b][real_mask[b]]
        n = int(real_ids.numel())
        repacked_ids[b, max_len - n :] = real_ids
        repacked_mask[b, max_len - n :] = 1
    return repacked_ids, repacked_mask


def _resolve_image_grid(params: ARSamplingParams) -> Tuple[int, int, int, int]:
    width = getattr(params, "width", None)
    height = getattr(params, "height", None)
    img_size = int(getattr(params, "img_size", 384))
    width = img_size if width is None else int(width)
    height = img_size if height is None else int(height)
    patch_size = int(getattr(params, "patch_size", 16))
    if width <= 0 or height <= 0 or patch_size <= 0:
        raise ValueError(
            "JanusProImageARSamplingParams requires positive width, height, and patch_size; "
            f"got width={width}, height={height}, patch_size={patch_size}."
        )
    if width % patch_size != 0 or height % patch_size != 0:
        raise ValueError(
            "Janus-Pro image size must be divisible by patch_size; "
            f"got width={width}, height={height}, patch_size={patch_size}."
        )
    grid_w = width // patch_size
    grid_h = height // patch_size
    token_count = grid_w * grid_h
    requested = int(params.max_new_tokens)
    if requested != token_count:
        raise ValueError(
            "Janus-Pro image token count must match the decode grid: "
            f"max_new_tokens={requested}, expected {token_count} for {width}x{height} / patch_size={patch_size}."
        )
    return width, height, patch_size, token_count


class JanusProImageARStage(ARStage[JanusProImageARConditions]):
    def __init__(
        self,
        *,
        model: JanusProBundle,
        autocast_precision: str = "bf16",
        logprob_precision: str = "fp32",
    ) -> None:
        self.model = model
        self.autocast_dtype = parse_torch_dtype(
            autocast_precision,
            field_name="JanusProImageARStage.autocast_precision",
        )
        self.logprob_dtype = parse_torch_dtype(
            logprob_precision,
            field_name="JanusProImageARStage.logprob_precision",
        )

    def trainable_module(self) -> torch.nn.Module:
        return self.model.transformer

    def _device(self) -> torch.device:
        return next(self.model.transformer.parameters()).device

    def _language_body(self) -> torch.nn.Module:
        body = getattr(self.model.transformer, "model", None)
        if body is None:
            body = getattr(getattr(self.model.transformer, "module", None), "model", None)
        if body is None:
            body = getattr(getattr(self.model.transformer, "_fsdp_wrapped_module", None), "model", None)
        if body is None:
            raise AttributeError("JanusProImageARStage requires a LlamaForCausalLM-style transformer.model.")
        return body

    def _autocast_ctx(self, device: torch.device):
        if device.type == "cuda" and self.autocast_dtype in (torch.float16, torch.bfloat16):
            return torch.autocast("cuda", self.autocast_dtype)
        return nullcontext()

    def _prepare_paired_prompt_embeds(
        self,
        conditions: JanusProImageARConditions,
        *,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if conditions.prompt is None or conditions.cfg_prompt is None:
            raise ValueError("JanusProImageARStage requires prompt and cfg_prompt conditions.")

        prompt_ids, prompt_mask = _left_repack_token_condition(
            conditions.prompt,
            pad_id=self.model.pad_token_id,
            device=device,
        )
        cfg_ids, cfg_mask = _left_repack_token_condition(
            conditions.cfg_prompt,
            pad_id=self.model.pad_token_id,
            device=device,
        )
        if prompt_ids.shape != cfg_ids.shape:
            raise ValueError(
                "JanusProImageARStage requires prompt and cfg_prompt to repack to the same shape; "
                f"got {tuple(prompt_ids.shape)} and {tuple(cfg_ids.shape)}."
            )

        paired_ids = torch.stack([prompt_ids, cfg_ids], dim=1).reshape(-1, prompt_ids.shape[1])
        paired_mask = torch.stack([prompt_mask, cfg_mask], dim=1).reshape(-1, prompt_mask.shape[1])
        inputs_embeds = self.model.transformer.get_input_embeddings()(paired_ids)
        return inputs_embeds, paired_mask

    @staticmethod
    def _cfg_logits(logits: torch.Tensor, cfg_weight: float) -> torch.Tensor:
        cond = logits[0::2]
        uncond = logits[1::2]
        return uncond + float(cfg_weight) * (cond - uncond)

    def autoregress(
        self,
        conditions: JanusProImageARConditions,
        *,
        sampling_params: ARSamplingParams,
        **_kwargs,
    ) -> TextSegment:
        _width, _height, _patch_size, token_count = _resolve_image_grid(sampling_params)
        device = self._device()
        step = JanusProARStep(
            temperature=float(sampling_params.temperature),
            top_p=float(sampling_params.top_p),
            top_k=int(sampling_params.top_k),
        )
        cfg_weight = float(getattr(sampling_params, "cfg_weight", conditions.cfg_weight))

        self.model.transformer.eval()
        self.model.model.eval()

        with torch.no_grad(), self._autocast_ctx(device):
            inputs_embeds, attention_mask = self._prepare_paired_prompt_embeds(conditions, device=device)
            paired_batch = int(inputs_embeds.shape[0])
            batch_size = paired_batch // 2
            body = self._language_body()
            past_key_values = None
            cur_attention_mask = attention_mask
            generated_tokens = torch.empty((batch_size, token_count), dtype=torch.long, device=device)
            generated_logps = torch.empty((batch_size, token_count), dtype=torch.float32, device=device)

            for i in range(token_count):
                if i == 0:
                    position_ids = _position_ids_from_attention_mask(cur_attention_mask)
                    out = body(
                        inputs_embeds=inputs_embeds,
                        attention_mask=cur_attention_mask,
                        position_ids=position_ids,
                        use_cache=True,
                    )
                else:
                    position_ids = _position_ids_from_attention_mask(cur_attention_mask)[:, -1:]
                    out = body(
                        inputs_embeds=inputs_embeds,
                        attention_mask=cur_attention_mask,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                past_key_values = out.past_key_values
                logits = self.model.model.gen_head(out.last_hidden_state[:, -1, :])
                token_id, log_prob = step.step(self._cfg_logits(logits, cfg_weight))

                generated_tokens[:, i] = token_id
                generated_logps[:, i] = log_prob

                paired_token = torch.stack([token_id, token_id], dim=1).reshape(-1)
                inputs_embeds = self.model.model.prepare_gen_img_embeds(paired_token).unsqueeze(1)
                cur_attention_mask = torch.cat(
                    [
                        cur_attention_mask,
                        torch.ones((paired_batch, 1), dtype=cur_attention_mask.dtype, device=device),
                    ],
                    dim=1,
                )

        segment = TextSegment.pack(
            tokens=[generated_tokens[i] for i in range(batch_size)],
            log_probs=[generated_logps[i] for i in range(batch_size)],
        )
        # Cached one-token decode and full-sequence teacher forcing are
        # mathematically equivalent, but bf16 attention kernels use different
        # numerical geometries. CFG amplifies that gap enough to move the
        # nominally on-policy PPO ratio far outside its clip range. Freeze the
        # old-policy anchor with the exact replay geometry used by training.
        # This extra forward is graph-free and replaces 576 cached forwards in
        # the old replay implementation.
        with torch.no_grad():
            segment.log_probs = self.replay(
                conditions,
                segment=segment,
                temperature=float(sampling_params.temperature),
            ).detach()
        return segment

    def replay(
        self,
        conditions: JanusProImageARConditions,
        *,
        segment: TextSegment,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        if segment.tokens is None or segment.cu_seqlens is None or segment.lengths is None:
            raise ValueError("JanusProImageARStage.replay: segment requires tokens with cu_seqlens")

        device = self._device()
        inputs_embeds, attention_mask = self._prepare_paired_prompt_embeds(conditions, device=device)
        paired_batch = int(inputs_embeds.shape[0])
        batch_size = paired_batch // 2
        lengths = [int(n) for n in segment.lengths.tolist()]
        if batch_size != len(lengths):
            raise ValueError(f"JanusProImageARStage.replay: batch={batch_size} != segment samples={len(lengths)}")

        t_max = max(lengths) if lengths else 0
        if t_max == 0:
            return torch.zeros(0, dtype=self.logprob_dtype, device=device)

        response_tokens = torch.zeros((batch_size, t_max), dtype=torch.long, device=device)
        response_mask = torch.zeros((batch_size, t_max), dtype=torch.long, device=device)
        cu = [int(c) for c in segment.cu_seqlens.tolist()]
        for b, n in enumerate(lengths):
            if n == 0:
                continue
            response_tokens[b, :n] = segment.tokens[cu[b] : cu[b] + n].to(device=device, dtype=torch.long)
            response_mask[b, :n] = 1

        paired_response_mask = torch.stack([response_mask, response_mask], dim=1).reshape(paired_batch, t_max)
        with self._autocast_ctx(device):
            body = self._language_body()
            paired_response_tokens = torch.stack([response_tokens, response_tokens], dim=1).reshape(
                paired_batch,
                t_max,
            )
            response_embeds = self.model.model.prepare_gen_img_embeds(paired_response_tokens.reshape(-1)).reshape(
                paired_batch, t_max, -1
            )

            # Teacher forcing does not need a token-by-token KV-cache loop: every
            # generated image token is already known. Feed the prompt plus tokens
            # [0, ..., T-2] once, then use hidden positions [prompt_last, ...]
            # to predict tokens [0, ..., T-1]. This is mathematically identical
            # to cached replay while avoiding 576 graph-retaining forwards.
            full_inputs_embeds = torch.cat([inputs_embeds, response_embeds[:, :-1]], dim=1)
            full_attention_mask = torch.cat([attention_mask, paired_response_mask[:, :-1]], dim=1)
            out = body(
                inputs_embeds=full_inputs_embeds,
                attention_mask=full_attention_mask,
                position_ids=_position_ids_from_attention_mask(full_attention_mask),
                use_cache=False,
            )

            temp = float(temperature) if float(temperature) > 0.0 else 1.0
            prompt_len = int(inputs_embeds.shape[1])
            prediction_hidden = out.last_hidden_state[:, prompt_len - 1 : prompt_len - 1 + t_max, :]
            if int(prediction_hidden.shape[1]) != t_max:
                raise RuntimeError(
                    "JanusProImageARStage.replay produced too few teacher-forced positions: "
                    f"got {prediction_hidden.shape[1]}, expected {t_max}."
                )
            logits = self.model.model.gen_head(prediction_hidden)
            cfg_logits = self._cfg_logits(logits, float(conditions.cfg_weight))
            log_probs_full = F.log_softmax(cfg_logits.float() / temp, dim=-1)
            per_token_logps = log_probs_full.gather(-1, response_tokens.unsqueeze(-1)).squeeze(-1)

        flat: List[torch.Tensor] = []
        for b, n in enumerate(lengths):
            if n == 0:
                continue
            flat.append(per_token_logps[b, :n])
        if not flat:
            return torch.zeros(0, dtype=self.logprob_dtype, device=device)
        return torch.cat(flat, dim=0).to(dtype=self.logprob_dtype)

    def decode(
        self,
        segment: TextSegment,
        *,
        sampling_params: ARSamplingParams,
    ) -> Images:
        if segment.tokens is None or segment.cu_seqlens is None or segment.lengths is None:
            raise ValueError("JanusProImageARStage.decode: segment requires tokens with cu_seqlens")

        width, height, patch_size, token_count = _resolve_image_grid(sampling_params)
        device = self._device()
        lengths = [int(n) for n in segment.lengths.tolist()]
        if any(n != token_count for n in lengths):
            raise ValueError(
                "JanusProImageARStage.decode expects fixed-length image token sequences; "
                f"got lengths={lengths}, expected={token_count}."
            )

        batch_size = len(lengths)
        tokens = torch.empty((batch_size, token_count), dtype=torch.long, device=device)
        cu = [int(c) for c in segment.cu_seqlens.tolist()]
        for b, n in enumerate(lengths):
            tokens[b, :n] = segment.tokens[cu[b] : cu[b] + n].to(device=device, dtype=torch.long)

        grid_h = height // patch_size
        grid_w = width // patch_size
        with torch.no_grad(), self._autocast_ctx(device):
            decoded = self.model.model.gen_vision_model.decode_code(
                tokens.to(dtype=torch.int),
                shape=[batch_size, 8, grid_h, grid_w],
            )
        pixels = ((decoded.float() + 1.0) / 2.0).clamp(0.0, 1.0)
        return Images(pixels=pixels)


__all__ = [
    "JanusProImageARSamplingParams",
    "JanusProImageARStage",
    "_left_repack_token_condition",
]
