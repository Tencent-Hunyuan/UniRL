"""Autoregression and replay for the Qwen3-Omni thinker."""

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass
from dataclasses import field as dc_field
from types import MethodType
from typing import Any, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from unirl.models.types.ar import ARSamplingParams, ARStage, ARStep, left_pad_prompt
from unirl.types.segments import TextSegment
from unirl.utils.dtypes import parse_torch_dtype

from .bundle import Qwen3OmniBundle
from .conditions import Qwen3OmniARConditions

logger = logging.getLogger(__name__)


def _fuse_mm_embeds(
    transformer: Any,
    full_ids: torch.Tensor,
    pixel_values: Optional[torch.Tensor] = None,
    image_grid_thw: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
    input_features: Optional[torch.Tensor] = None,
    feature_attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]], Optional[torch.Tensor]]:
    """Prepare audio, video, and DeepStack inputs inside FSDP."""
    inputs_embeds = transformer.get_input_embeddings()(full_ids)
    if input_features is not None:
        try:
            audio_outputs = transformer.get_audio_features(
                input_features,
                feature_attention_mask=feature_attention_mask,
                return_dict=True,
            )
        except TypeError as exc:
            if "unexpected keyword argument 'return_dict'" not in str(exc):
                raise
            audio_outputs = transformer.get_audio_features(
                input_features,
                feature_attention_mask=feature_attention_mask,
            )
        if hasattr(audio_outputs, "last_hidden_state"):
            audio_features = audio_outputs.last_hidden_state
        elif isinstance(audio_outputs, tuple):
            audio_features = audio_outputs[0]
        else:
            audio_features = audio_outputs
        audio_features = audio_features.to(inputs_embeds.device, inputs_embeds.dtype)
        _, _, audio_mask = transformer.get_placeholder_mask(full_ids, inputs_embeds=inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(audio_mask, audio_features)

    image_multiscale: Optional[List[torch.Tensor]] = None
    image_mask: Optional[torch.Tensor] = None
    image_legacy = False
    video_multiscale: Optional[List[torch.Tensor]] = None
    video_mask: Optional[torch.Tensor] = None
    video_legacy = False
    visual_mask: Optional[torch.Tensor] = None

    def _unpack_visual(outputs: Any) -> Tuple[torch.Tensor, List[torch.Tensor], bool]:
        if hasattr(outputs, "pooler_output"):
            return outputs.pooler_output, list(outputs.deepstack_features), False
        embeds, multiscale = outputs
        return embeds, list(multiscale), True

    if pixel_values is not None:
        try:
            image_outputs = transformer.get_image_features(
                pixel_values,
                image_grid_thw,
                return_dict=True,
            )
        except TypeError as exc:
            if "unexpected keyword argument 'return_dict'" not in str(exc):
                raise
            image_outputs = transformer.get_image_features(pixel_values, image_grid_thw)
        image_embeds, image_multiscale, image_legacy = _unpack_visual(image_outputs)
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _, _ = transformer.get_placeholder_mask(
            full_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        try:
            video_outputs = transformer.get_video_features(
                pixel_values_videos,
                video_grid_thw,
                return_dict=True,
            )
        except TypeError as exc:
            if "unexpected keyword argument 'return_dict'" not in str(exc):
                raise
            video_outputs = transformer.get_video_features(pixel_values_videos, video_grid_thw)
        video_embeds, video_multiscale, video_legacy = _unpack_visual(video_outputs)
        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        _, video_mask, _ = transformer.get_placeholder_mask(
            full_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    masks = [mask[..., 0] for mask in (image_mask, video_mask) if mask is not None]
    if masks:
        visual_mask = torch.stack(masks).any(dim=0)
    visual_multiscale = _merge_visual_multiscale(
        image_multiscale,
        video_multiscale,
        image_mask[..., 0] if image_mask is not None else None,
        video_mask[..., 0] if video_mask is not None else None,
    )
    visual_pos_masks = None
    if visual_mask is not None:
        legacy_visual_outputs = image_legacy or video_legacy
        visual_pos_masks = visual_mask.unsqueeze(-1).expand_as(inputs_embeds) if legacy_visual_outputs else visual_mask
    return inputs_embeds, visual_multiscale, visual_pos_masks


def _merge_visual_multiscale(
    image_features: Optional[List[torch.Tensor]],
    video_features: Optional[List[torch.Tensor]],
    image_mask: Optional[torch.Tensor],
    video_mask: Optional[torch.Tensor],
) -> Optional[List[torch.Tensor]]:
    """Merge image/video DeepStack features exactly in placeholder order."""
    if image_features is None:
        return video_features
    if video_features is None:
        return image_features
    if image_mask is None or video_mask is None:
        raise ValueError("Qwen3-Omni visual features require matching placeholder masks.")
    if len(image_features) != len(video_features):
        raise ValueError("Qwen3-Omni image/video DeepStack feature counts do not match.")

    visual_mask = image_mask | video_mask
    image_mask_joint = image_mask[visual_mask]
    video_mask_joint = video_mask[visual_mask]
    merged: List[torch.Tensor] = []
    for image_level, video_level in zip(image_features, video_features):
        joint = image_level.new_zeros((int(visual_mask.sum().item()), image_level.shape[-1]))
        joint[image_mask_joint, :] = image_level
        joint[video_mask_joint, :] = video_level
        merged.append(joint)
    return merged


def _replay_aware_forward(
    self: Any,
    *,
    response_tokens: Optional[torch.Tensor] = None,
    prompt_len: Optional[int] = None,
    temperature: float = 1.0,
    autocast_dtype: Optional[torch.dtype] = None,
    **kw: Any,
) -> Any:
    """Delegate decode or compute chunked replay log-probs inside FSDP."""
    if response_tokens is None:
        for klass in type(self).__mro__:
            f = klass.__dict__.get("forward")
            if f is not None and f is not _replay_aware_forward:
                return f(self, **kw)
        raise RuntimeError("_replay_aware_forward: no class-level forward found in the MRO")

    if torch.cuda.is_available():
        torch.backends.cuda.enable_cudnn_sdp(False)

    # FSDP must unshard embeddings and the media towers before fusion.
    pixel_values = kw.pop("pixel_values", None)
    pixel_values_videos = kw.pop("pixel_values_videos", None)
    input_features = kw.pop("input_features", None)
    feature_attention_mask = kw.pop("feature_attention_mask", None)
    if pixel_values is not None or pixel_values_videos is not None or input_features is not None:
        image_grid_thw = kw.pop("image_grid_thw", None)
        video_grid_thw = kw.pop("video_grid_thw", None)
        fuse_full_ids = kw.pop("fuse_full_ids")
        inputs_embeds, deepstack_embeds, visual_pos_masks = _fuse_mm_embeds(
            self,
            fuse_full_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
        )
        kw["inputs_embeds"] = inputs_embeds
        if deepstack_embeds is not None:
            kw["deepstack_visual_embeds"] = deepstack_embeds
            kw["visual_pos_masks"] = visual_pos_masks

    autocast_ctx = (
        torch.autocast("cuda", autocast_dtype) if autocast_dtype in (torch.float16, torch.bfloat16) else nullcontext()
    )
    with autocast_ctx:
        hidden = self.model(**kw, use_cache=False, return_dict=True).last_hidden_state

    T = float(temperature) if float(temperature) > 0.0 else 1.0
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
        return resp_hidden.new_zeros((bsz, 0), dtype=torch.float32)
    return torch.cat(parts, dim=1)


@dataclass
class Qwen3OmniARParams:
    """Per-request AR-mode knobs. ``stop_token_ids`` is unioned with EOS in-stage."""

    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0
    stop_token_ids: List[int] = dc_field(default_factory=list)


class Qwen3OmniARStep(ARStep):
    """Sample tokens while recording pre-truncation behavior log-probabilities."""

    def __init__(self, *, temperature: float = 1.0, top_p: float = 1.0, top_k: int = 0) -> None:
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)

    def step(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if logits.dim() != 2:
            raise ValueError(f"Qwen3OmniARStep.step: expected logits [B, vocab], got {tuple(logits.shape)}")

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


def _merge_video(per_sample: Optional[List[Optional[torch.Tensor]]]) -> Optional[torch.Tensor]:
    """Cat per-sample video tensors into one flat tensor for the thinker forward."""
    if per_sample is None:
        return None
    parts = [t for t in per_sample if t is not None]
    return torch.cat(parts, dim=0) if parts else None


def _merge_audio(
    input_features: Optional[List[Optional[torch.Tensor]]],
    feature_attention_mask: Optional[List[Optional[torch.Tensor]]],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Pad and concatenate per-sample Whisper features and masks."""
    if input_features is None or feature_attention_mask is None:
        return None, None
    if len(input_features) != len(feature_attention_mask):
        raise ValueError(
            "Qwen3-Omni audio feature/mask lists must share one batch size: "
            f"input_features={len(input_features)}, "
            f"feature_attention_mask={len(feature_attention_mask)}."
        )
    pairs = [(f, m) for f, m in zip(input_features, feature_attention_mask) if f is not None and m is not None]
    if not pairs:
        return None, None
    max_t = max(int(f.shape[-1]) for f, _ in pairs)
    features, masks = [], []
    for feature, mask in pairs:
        pad = max_t - int(feature.shape[-1])
        if pad:
            feature = F.pad(feature, (0, pad))
            mask = F.pad(mask, (0, pad))
        features.append(feature)
        masks.append(mask)
    return torch.cat(features, dim=0), torch.cat(masks, dim=0)


def _validate_audio_in_video_modes(conditions: Qwen3OmniARConditions, batch_size: int) -> None:
    """Reject video-only/audio-in-video mixing unsupported by Qwen3-Omni generation."""
    flags = conditions.use_audio_in_video
    video_grids = conditions.video_grid_thw
    if flags is None or video_grids is None:
        return
    if len(flags) != batch_size or len(video_grids) != batch_size:
        raise ValueError(
            "Qwen3-Omni per-row media conditions must match the text batch size: "
            f"batch={batch_size}, use_audio_in_video={len(flags)}, video_grid_thw={len(video_grids)}."
        )

    video_modes = {bool(flags[row]) for row, grid in enumerate(video_grids) if grid is not None}
    if len(video_modes) > 1:
        raise ValueError(
            "Qwen3-Omni does not currently support mixing video-only and audio-in-video rows "
            "in one batch. Configure the training run uniformly with use_audio_in_video=false "
            "or use_audio_in_video=true."
        )

    # TODO(qwen3-omni): Add end-to-end support for training video-only and
    # audio-in-video rows together. This requires a separate design and parity
    # validation across generation, model forward, and replay.


def _per_sample_rope(
    transformer: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    conditions: Qwen3OmniARConditions,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build per-row TMRoPE positions for one supported, uniform video mode."""
    batch_size = int(input_ids.shape[0])
    _validate_audio_in_video_modes(conditions, batch_size)
    device = input_ids.device
    image_grids = conditions.image_grid_thw or [None] * batch_size
    video_grids = conditions.video_grid_thw or [None] * batch_size
    seconds = conditions.video_second_per_grid or [None] * batch_size
    features = conditions.input_features or [None] * batch_size
    feature_masks = conditions.feature_attention_mask or [None] * batch_size
    audio_in_video = conditions.use_audio_in_video or [False] * batch_size
    position_parts: List[torch.Tensor] = []
    delta_parts: List[torch.Tensor] = []
    for row in range(batch_size):
        feature = features[row]
        feature_mask = feature_masks[row]
        has_audio = feature is not None and feature_mask is not None
        audio_seqlens = feature_mask.to(device=device).sum(-1) if has_audio else None
        position_ids, rope_delta = transformer.get_rope_index(
            input_ids[row : row + 1],
            image_grid_thw=image_grids[row].to(device=device) if image_grids[row] is not None else None,
            video_grid_thw=video_grids[row].to(device=device) if video_grids[row] is not None else None,
            attention_mask=attention_mask[row : row + 1],
            use_audio_in_video=bool(audio_in_video[row]),
            audio_seqlens=audio_seqlens,
            second_per_grids=seconds[row].to(device=device) if seconds[row] is not None else None,
        )
        position_parts.append(position_ids)
        delta_parts.append(torch.as_tensor(rope_delta, device=device).reshape(1, -1))
    position_ids = torch.cat(position_parts, dim=1).long()
    rope_deltas = torch.cat(delta_parts, dim=0)
    rope_deltas = rope_deltas - (1 - attention_mask.long()).sum(dim=-1, keepdim=True)
    return position_ids, rope_deltas


class Qwen3OmniARStage(ARStage[Qwen3OmniARConditions]):
    """Rollout-level AR stage for the Qwen3-Omni thinker."""

    def __init__(
        self,
        *,
        model: Qwen3OmniBundle,
        autocast_precision: str = "bf16",
        logprob_precision: str = "fp32",
    ) -> None:
        self.model = model
        self.autocast_dtype = parse_torch_dtype(autocast_precision, field_name="Qwen3OmniARStage.autocast_precision")
        self.logprob_dtype = parse_torch_dtype(logprob_precision, field_name="Qwen3OmniARStage.logprob_precision")
        transformer = model.transformer
        if getattr(transformer.forward, "__func__", None) is not _replay_aware_forward:
            transformer.forward = MethodType(_replay_aware_forward, transformer)

    def trainable_module(self) -> "torch.nn.Module":
        """The thinker CausalLM module — the FSDP/LoRA wrap target."""
        return self.model.transformer

    def autoregress(
        self,
        conditions: Qwen3OmniARConditions,
        *,
        sampling_params: ARSamplingParams,
        params: Optional[Qwen3OmniARParams] = None,
        **_kwargs: Any,
    ) -> TextSegment:
        """Generate a packed segment, including TMRoPE video inputs when present."""
        if conditions.prompt is None or conditions.prompt.input_ids is None:
            raise ValueError("Qwen3OmniARStage.autoregress: requires conditions.prompt.input_ids")
        if conditions.prompt.attention_mask is None:
            raise ValueError("Qwen3OmniARStage.autoregress: requires conditions.prompt.attention_mask")

        transformer = self.model.transformer
        input_ids: torch.Tensor = conditions.prompt.input_ids
        attention_mask: torch.Tensor = conditions.prompt.attention_mask
        device = input_ids.device

        pad_id = self.model.tokenizer.pad_token_id or 0
        input_ids, attention_mask = left_pad_prompt(input_ids, attention_mask, pad_id)
        batch_size = int(input_ids.shape[0])

        stop_ids = self._resolve_stop_ids(params, sampling_params)
        step = Qwen3OmniARStep(
            temperature=float(sampling_params.temperature),
            top_p=float(sampling_params.top_p),
            top_k=int(sampling_params.top_k),
        )
        max_new = int(sampling_params.max_new_tokens)

        if hasattr(transformer, "model") and hasattr(transformer.model, "rope_deltas"):
            transformer.model.rope_deltas = None

        model_kwargs: dict = {
            "attention_mask": attention_mask,
            "use_cache": True,
            "past_key_values": None,
            "cache_position": torch.arange(int(input_ids.shape[1]), device=device, dtype=torch.long),
        }
        pv = _merge_video(conditions.pixel_values)
        igt = _merge_video(conditions.image_grid_thw)
        pvv = _merge_video(conditions.pixel_values_videos)
        vgt = _merge_video(conditions.video_grid_thw)
        vspg = _merge_video(conditions.video_second_per_grid)
        ivf, fam = _merge_audio(conditions.input_features, conditions.feature_attention_mask)
        if pv is not None:
            model_kwargs["pixel_values"] = pv
        if igt is not None:
            model_kwargs["image_grid_thw"] = igt
        if pvv is not None:
            model_kwargs["pixel_values_videos"] = pvv
        if vgt is not None:
            model_kwargs["video_grid_thw"] = vgt
        if vspg is not None:
            model_kwargs["video_second_per_grid"] = vspg
        if ivf is not None:
            model_kwargs["input_features"] = ivf
            model_kwargs["feature_attention_mask"] = fam
            model_kwargs["use_audio_in_video"] = any(conditions.use_audio_in_video or [])
        if pv is not None or pvv is not None or ivf is not None:
            position_ids, rope_deltas = _per_sample_rope(transformer, input_ids, attention_mask, conditions)
            model_kwargs["position_ids"] = position_ids
            if hasattr(transformer, "model") and hasattr(transformer.model, "rope_deltas"):
                transformer.model.rope_deltas = rope_deltas

        cur_input_ids = input_ids
        generated_tokens: List[List[int]] = [[] for _ in range(batch_size)]
        per_token_logps: List[List[float]] = [[] for _ in range(batch_size)]
        finished = [False] * batch_size
        is_first_step = True

        for _ in range(max_new):
            prep_kwargs: dict = {
                "past_key_values": model_kwargs.get("past_key_values"),
                "attention_mask": model_kwargs.get("attention_mask"),
                "cache_position": model_kwargs.get("cache_position"),
                "use_cache": True,
            }
            if is_first_step:
                if "position_ids" in model_kwargs:
                    prep_kwargs["position_ids"] = model_kwargs["position_ids"]
                if "pixel_values" in model_kwargs:
                    prep_kwargs["pixel_values"] = model_kwargs["pixel_values"]
                if "image_grid_thw" in model_kwargs:
                    prep_kwargs["image_grid_thw"] = model_kwargs["image_grid_thw"]
                if "pixel_values_videos" in model_kwargs:
                    prep_kwargs["pixel_values_videos"] = model_kwargs["pixel_values_videos"]
                if "video_grid_thw" in model_kwargs:
                    prep_kwargs["video_grid_thw"] = model_kwargs["video_grid_thw"]
                if "video_second_per_grid" in model_kwargs:
                    prep_kwargs["video_second_per_grid"] = model_kwargs["video_second_per_grid"]
                if "input_features" in model_kwargs:
                    prep_kwargs["input_features"] = model_kwargs["input_features"]
                    prep_kwargs["feature_attention_mask"] = model_kwargs["feature_attention_mask"]
                    prep_kwargs["use_audio_in_video"] = model_kwargs["use_audio_in_video"]

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
            is_first_step = False

        return _pack_text_segment(generated_tokens, per_token_logps, device=device)

    def replay(
        self,
        conditions: Qwen3OmniARConditions,
        *,
        segment: TextSegment,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Return packed teacher-forced log-probs aligned with ``segment``."""
        if conditions.prompt is None or conditions.prompt.input_ids is None:
            raise ValueError("Qwen3OmniARStage.replay: conditions.prompt.input_ids is None")
        if conditions.prompt.attention_mask is None:
            raise ValueError("Qwen3OmniARStage.replay: conditions.prompt.attention_mask is None")
        if segment.tokens is None or segment.cu_seqlens is None or segment.lengths is None:
            raise ValueError("Qwen3OmniARStage.replay: segment requires tokens with cu_seqlens (TextSegment.pack)")

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
            left_ids = torch.full_like(prompt_ids, pad_id)
            left_mask = torch.zeros_like(prompt_mask)
            for b in range(batch_size):
                n_real = int(real_prompt_lens[b].item())
                if n_real == 0:
                    continue
                left_ids[b, prompt_len - n_real :] = prompt_ids[b, :n_real]
                left_mask[b, prompt_len - n_real :] = 1
            prompt_ids = left_ids
            prompt_mask = left_mask

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

        transformer = self.model.transformer
        if hasattr(transformer, "model") and hasattr(transformer.model, "rope_deltas"):
            transformer.model.rope_deltas = None

        pv = _merge_video(conditions.pixel_values)
        igt = _merge_video(conditions.image_grid_thw)
        pvv = _merge_video(conditions.pixel_values_videos)
        vgt = _merge_video(conditions.video_grid_thw)
        ivf, fam = _merge_audio(conditions.input_features, conditions.feature_attention_mask)

        forward_kwargs: dict = {
            "response_tokens": response_tokens,
            "prompt_len": prompt_len,
            "temperature": temperature,
            "autocast_dtype": (self.autocast_dtype if device.type == "cuda" else None),
        }

        if pv is None and pvv is None and ivf is None:
            forward_kwargs["input_ids"] = full_ids
            forward_kwargs["attention_mask"] = full_mask
            forward_kwargs["position_ids"] = (full_mask.long().cumsum(dim=-1) - 1).clamp(min=0)
        else:
            position_ids, _ = _per_sample_rope(transformer, full_ids, full_mask, conditions)
            if pv is not None:
                forward_kwargs["pixel_values"] = pv.to(device=device, dtype=self.model.dtype)
                forward_kwargs["image_grid_thw"] = igt.to(device=device)
            if pvv is not None:
                forward_kwargs["pixel_values_videos"] = pvv.to(device=device, dtype=self.model.dtype)
                forward_kwargs["video_grid_thw"] = vgt.to(device=device)
            if ivf is not None:
                ivf = ivf.to(device=device, dtype=self.model.dtype)
                fam = fam.to(device=device)
                forward_kwargs["input_features"] = ivf
                forward_kwargs["feature_attention_mask"] = fam
            forward_kwargs["fuse_full_ids"] = full_ids
            forward_kwargs["attention_mask"] = full_mask
            forward_kwargs["position_ids"] = position_ids
        per_token = transformer(**forward_kwargs)

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
        params: Optional[Qwen3OmniARParams],
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


__all__ = ["Qwen3OmniARParams", "Qwen3OmniARStage", "Qwen3OmniARStep"]
