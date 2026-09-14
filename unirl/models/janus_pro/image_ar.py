from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, List, Optional, Tuple

import torch
import torch.nn.functional as F

from unirl.models.types.ar import ARSamplingParams, ARStage
from unirl.types.primitives import Images
from unirl.types.sampling import _as_int
from unirl.types.segments import TextSegment
from unirl.utils.dtypes import parse_torch_dtype

from .ar import (
    JanusProARStep,
    _autocast_ctx,
    _decode_attention_state,
    _keep_fsdp_params_unsharded,
    _left_repack_token_condition,
    _position_ids_from_attention_mask,
    _replay_temperature,
)
from .bundle import JanusProBundle
from .conditions import JanusProImageARConditions, _finite_cfg_weight

_VQ_SPATIAL_FACTOR = 16


@dataclass
class JanusProImageARSamplingParams(ARSamplingParams):
    """Sampling parameters for Janus-Pro autoregressive image tokens."""

    emits_fixed_length: ClassVar[bool] = True

    temperature: float = 1.0
    max_new_tokens: int = 576
    top_p: float = 1.0
    top_k: int = 0
    cfg_weight: float = 5.0
    img_size: int = 384
    width: Optional[int] = None
    height: Optional[int] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        self.cfg_weight = _finite_cfg_weight(
            self.cfg_weight,
            where="JanusProImageARSamplingParams.cfg_weight",
        )
        self.img_size = _as_int(self.img_size, name="JanusProImageARSamplingParams.img_size")
        self.width = _as_int(self.width, name="JanusProImageARSamplingParams.width", optional=True)
        self.height = _as_int(self.height, name="JanusProImageARSamplingParams.height", optional=True)


def _resolve_image_grid(params: ARSamplingParams) -> Tuple[int, int, int]:
    if not isinstance(params, JanusProImageARSamplingParams):
        raise TypeError(
            f"Janus-Pro image generation requires JanusProImageARSamplingParams; got {type(params).__name__}."
        )
    width = params.width
    height = params.height
    width = params.img_size if width is None else width
    height = params.img_size if height is None else height
    if width <= 0 or height <= 0:
        raise ValueError(f"Janus-Pro image dimensions must be positive; got width={width}, height={height}.")
    if width % _VQ_SPATIAL_FACTOR != 0 or height % _VQ_SPATIAL_FACTOR != 0:
        raise ValueError(
            f"Janus-Pro image dimensions must be divisible by {_VQ_SPATIAL_FACTOR}; got width={width}, height={height}."
        )
    grid_w = width // _VQ_SPATIAL_FACTOR
    grid_h = height // _VQ_SPATIAL_FACTOR
    token_count = grid_w * grid_h
    if params.max_new_tokens != token_count:
        raise ValueError(
            "Janus-Pro image token count must match the decode grid: "
            f"max_new_tokens={params.max_new_tokens}, expected {token_count} for {width}x{height}."
        )
    return width, height, token_count


class JanusProImageARStage(ARStage[JanusProImageARConditions]):
    def __init__(
        self,
        *,
        model: JanusProBundle,
        autocast_precision: str,
        logprob_precision: str,
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

    def _prepare_paired_prompt_embeds(
        self,
        conditions: JanusProImageARConditions,
        *,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prompt_ids, prompt_mask = _left_repack_token_condition(
            conditions.prompt,
            pad_id=self.model.pad_token_id,
            device=device,
            where="JanusProImageARStage",
        )
        cfg_ids, cfg_mask = _left_repack_token_condition(
            conditions.cfg_prompt,
            pad_id=self.model.pad_token_id,
            device=device,
            where="JanusProImageARStage.cfg_prompt",
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
        return uncond + cfg_weight * (cond - uncond)

    def autoregress(
        self,
        conditions: JanusProImageARConditions,
        *,
        sampling_params: ARSamplingParams,
        **_kwargs,
    ) -> TextSegment:
        _width, _height, token_count = _resolve_image_grid(sampling_params)
        device = self._device()
        step = JanusProARStep(
            temperature=sampling_params.temperature,
            top_p=sampling_params.top_p,
            top_k=sampling_params.top_k,
        )
        # `conditions.cfg_weight` is the single source of truth: replay runs from
        # GRPO with no sampling_params, so anchoring the rollout on a different
        # value would silently bias every importance ratio. The pipeline copies
        # sampling_params.cfg_weight into the conditions, so a mismatch here
        # means the two were wired from different places.
        cfg_weight = conditions.cfg_weight
        if sampling_params.cfg_weight != cfg_weight:
            raise ValueError(
                "JanusProImageARStage: sampling_params.cfg_weight="
                f"{sampling_params.cfg_weight} disagrees with conditions.cfg_weight={cfg_weight}; "
                "replay can only see the conditions value, so the PPO ratio would be biased."
            )

        with (
            torch.no_grad(),
            _autocast_ctx(device, self.autocast_dtype),
            _keep_fsdp_params_unsharded(
                self.model.transformer,
                enabled=self.model.rollout_keep_params_unsharded,
            ),
        ):
            inputs_embeds, attention_mask = self._prepare_paired_prompt_embeds(conditions, device=device)
            batch_size = inputs_embeds.shape[0] // 2
            body = self.model.transformer.model
            past_key_values = None
            generated_tokens = torch.empty((batch_size, token_count), dtype=torch.long, device=device)
            generated_logps = torch.empty((batch_size, token_count), dtype=self.logprob_dtype, device=device)
            full_attention_mask, prompt_position_ids, next_position_ids = _decode_attention_state(
                attention_mask,
                token_count,
            )
            prompt_length = attention_mask.shape[1]

            for i in range(token_count):
                if i == 0:
                    out = body(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask,
                        position_ids=prompt_position_ids,
                        use_cache=True,
                    )
                else:
                    out = body(
                        inputs_embeds=inputs_embeds,
                        attention_mask=full_attention_mask[:, : prompt_length + i],
                        position_ids=next_position_ids + i - 1,
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

        # Cached one-token decode and full-sequence teacher forcing are
        # mathematically equivalent, but bf16 attention kernels use different
        # numerical geometries and CFG amplifies that gap enough to move the
        # nominally on-policy PPO ratio far outside its clip range. The T2I
        # recipe therefore sets `algorithm.old_logp_source: replay` so the
        # anchor is frozen train-side at the exact micro geometry training
        # replays at, rather than at whatever shape rollout happened to use.
        return TextSegment.pack(
            tokens=[generated_tokens[i] for i in range(batch_size)],
            log_probs=[generated_logps[i] for i in range(batch_size)],
        )

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
        paired_batch = inputs_embeds.shape[0]
        batch_size = paired_batch // 2
        lengths = segment.lengths.tolist()
        if batch_size != len(lengths):
            raise ValueError(f"JanusProImageARStage.replay: batch={batch_size} != segment samples={len(lengths)}")

        t_max = max(lengths) if lengths else 0
        if t_max == 0:
            return torch.zeros(0, dtype=self.logprob_dtype, device=device)

        response_tokens = torch.zeros((batch_size, t_max), dtype=torch.long, device=device)
        response_mask = torch.zeros((batch_size, t_max), dtype=torch.long, device=device)
        cu = segment.cu_seqlens.tolist()
        for b, n in enumerate(lengths):
            if n == 0:
                continue
            response_tokens[b, :n] = segment.tokens[cu[b] : cu[b] + n].to(device=device, dtype=torch.long)
            response_mask[b, :n] = 1

        paired_response_mask = torch.stack([response_mask, response_mask], dim=1).reshape(paired_batch, t_max)
        with _autocast_ctx(device, self.autocast_dtype):
            body = self.model.transformer.model
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

            temp = _replay_temperature(temperature)
            prompt_len = inputs_embeds.shape[1]
            prediction_hidden = out.last_hidden_state[:, prompt_len - 1 : prompt_len - 1 + t_max, :]
            if prediction_hidden.shape[1] != t_max:
                raise RuntimeError(
                    "JanusProImageARStage.replay produced too few teacher-forced positions: "
                    f"got {prediction_hidden.shape[1]}, expected {t_max}."
                )
            logits = self.model.model.gen_head(prediction_hidden)
            cfg_logits = self._cfg_logits(logits, conditions.cfg_weight)
            log_probs_full = F.log_softmax(cfg_logits.float() / temp, dim=-1)
            per_token_logps = log_probs_full.gather(-1, response_tokens.unsqueeze(-1)).squeeze(-1)

        flat: List[torch.Tensor] = []
        for b, n in enumerate(lengths):
            if n == 0:
                continue
            flat.append(per_token_logps[b, :n])
        return torch.cat(flat, dim=0).to(dtype=self.logprob_dtype)

    def decode(
        self,
        segment: TextSegment,
        *,
        sampling_params: ARSamplingParams,
    ) -> Images:
        if segment.tokens is None or segment.cu_seqlens is None or segment.lengths is None:
            raise ValueError("JanusProImageARStage.decode: segment requires tokens with cu_seqlens")

        width, height, token_count = _resolve_image_grid(sampling_params)
        device = self._device()
        lengths = segment.lengths.tolist()
        if any(n != token_count for n in lengths):
            raise ValueError(
                "JanusProImageARStage.decode expects fixed-length image token sequences; "
                f"got lengths={lengths}, expected={token_count}."
            )

        batch_size = len(lengths)
        tokens = torch.empty((batch_size, token_count), dtype=torch.long, device=device)
        cu = segment.cu_seqlens.tolist()
        for b, n in enumerate(lengths):
            tokens[b, :n] = segment.tokens[cu[b] : cu[b] + n].to(device=device, dtype=torch.long)

        grid_h = height // _VQ_SPATIAL_FACTOR
        grid_w = width // _VQ_SPATIAL_FACTOR
        with torch.no_grad(), _autocast_ctx(device, self.autocast_dtype):
            decoded = self.model.model.gen_vision_model.decode_code(
                tokens.to(dtype=torch.int),
                shape=[batch_size, 8, grid_h, grid_w],
            )
        pixels = ((decoded.float() + 1.0) / 2.0).clamp(0.0, 1.0)
        return Images.from_dense(pixels)


__all__ = [
    "JanusProImageARSamplingParams",
    "JanusProImageARStage",
]
