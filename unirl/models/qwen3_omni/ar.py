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


def _fuse_video_embeds(
    transformer: Any,
    full_ids: torch.Tensor,
    pixel_values_videos: torch.Tensor,
    video_grid_thw: torch.Tensor,
) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
    """Prepare video and DeepStack inputs inside the root FSDP forward."""
    inputs_embeds = transformer.get_input_embeddings()(full_ids)
    video_outputs = transformer.get_video_features(
        pixel_values_videos,
        video_grid_thw,
        return_dict=True,
    )
    video_embeds = video_outputs.pooler_output
    video_embeds_multiscale = video_outputs.deepstack_features
    video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
    _, video_mask, _ = transformer.get_placeholder_mask(
        full_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
    )
    inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
    deepstack_embeds = list(video_embeds_multiscale)
    visual_pos_masks = video_mask[..., 0]
    return inputs_embeds, deepstack_embeds, visual_pos_masks


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

    # Avoid unstable cuDNN SDPA backward for bf16 replay.
    if torch.cuda.is_available():
        torch.backends.cuda.enable_cudnn_sdp(False)

    # FSDP must unshard embeddings and the vision tower before video fusion.
    pixel_values_videos = kw.pop("pixel_values_videos", None)
    if pixel_values_videos is not None:
        video_grid_thw = kw.pop("video_grid_thw")
        fuse_full_ids = kw.pop("fuse_full_ids")
        inputs_embeds, deepstack_embeds, visual_pos_masks = _fuse_video_embeds(
            self, fuse_full_ids, pixel_values_videos, video_grid_thw
        )
        kw["inputs_embeds"] = inputs_embeds
        kw["deepstack_visual_embeds"] = deepstack_embeds
        kw["visual_pos_masks"] = visual_pos_masks

    autocast_ctx = (
        torch.autocast("cuda", autocast_dtype) if autocast_dtype in (torch.float16, torch.bfloat16) else nullcontext()
    )
    with autocast_ctx:
        hidden = self.model(**kw, use_cache=False, return_dict=True).last_hidden_state  # [B, L, H]

    T = float(temperature) if float(temperature) > 0.0 else 1.0
    T_max = int(response_tokens.size(1))
    resp_hidden = hidden[:, prompt_len - 1 : prompt_len - 1 + T_max, :]

    def _logp_chunk(h: torch.Tensor, tok: torch.Tensor) -> torch.Tensor:
        lf = self.lm_head(h).float() / T  # [B, chunk, vocab] FP32
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
        # The instance override survives FSDP class swapping and LoRA injection.
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

        # Reset cached TMRoPE offsets between requests.
        if hasattr(transformer, "model") and hasattr(transformer.model, "rope_deltas"):
            transformer.model.rope_deltas = None

        model_kwargs: dict = {
            "attention_mask": attention_mask,
            "use_cache": True,
            "past_key_values": None,
            "cache_position": torch.arange(int(input_ids.shape[1]), device=device, dtype=torch.long),
        }
        # Multimodal tensors are consumed on the first decode step.
        pvv = _merge_video(conditions.pixel_values_videos)
        vgt = _merge_video(conditions.video_grid_thw)
        vspg = _merge_video(conditions.video_second_per_grid)
        if pvv is not None:
            model_kwargs["pixel_values_videos"] = pvv
        if vgt is not None:
            model_kwargs["video_grid_thw"] = vgt
        if vspg is not None:
            # TMRoPE needs seconds per grid for the temporal axis.
            model_kwargs["video_second_per_grid"] = vspg

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
                if "pixel_values_videos" in model_kwargs:
                    prep_kwargs["pixel_values_videos"] = model_kwargs["pixel_values_videos"]
                if "video_grid_thw" in model_kwargs:
                    prep_kwargs["video_grid_thw"] = model_kwargs["video_grid_thw"]
                if "video_second_per_grid" in model_kwargs:
                    prep_kwargs["video_second_per_grid"] = model_kwargs["video_second_per_grid"]

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

        # Move CONCAT padding before prompts so responses start at one boundary.
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

        # Merge per-sample CONCAT media for the thinker.
        pvv = _merge_video(conditions.pixel_values_videos)
        vgt = _merge_video(conditions.video_grid_thw)
        vspg = _merge_video(conditions.video_second_per_grid)

        forward_kwargs: dict = {
            "response_tokens": response_tokens,
            "prompt_len": prompt_len,
            "temperature": temperature,
            "autocast_dtype": (self.autocast_dtype if device.type == "cuda" else None),
        }

        if pvv is None:
            # Cumulative positions keep text RoPE invariant to padding.
            forward_kwargs["input_ids"] = full_ids
            forward_kwargs["attention_mask"] = full_mask
            forward_kwargs["position_ids"] = (full_mask.long().cumsum(dim=-1) - 1).clamp(min=0)
        else:
            # Compute TMRoPE here, but defer parameter reads to the FSDP forward.
            pvv = pvv.to(device=device, dtype=self.model.dtype)
            vgt = vgt.to(device=device)
            vspg = vspg.to(device=device) if vspg is not None else None
            # Preserve the [3, B, seq] temporal/height/width position layout.
            position_ids, _ = transformer.get_rope_index(
                full_ids,
                image_grid_thw=None,
                video_grid_thw=vgt,
                attention_mask=full_mask,
                use_audio_in_video=False,
                audio_seqlens=None,
                second_per_grids=vspg,
            )  # [3, B, seq]
            # Integer positions prevent FSDP mixed precision from rounding indices.
            position_ids = position_ids.long()
            forward_kwargs["pixel_values_videos"] = pvv
            forward_kwargs["video_grid_thw"] = vgt
            forward_kwargs["fuse_full_ids"] = full_ids
            forward_kwargs["attention_mask"] = full_mask
            forward_kwargs["position_ids"] = position_ids
        per_token = transformer(**forward_kwargs)  # [B, T_max] FP32

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
