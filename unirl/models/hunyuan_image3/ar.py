"""HunyuanImage3 AR stage: typed params + per-token kernel + rollout-level stage."""

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
    """Per-request AR-mode knobs for HunyuanImage 3.0."""

    bot_task: str = "auto"
    max_tokens: int = 2048
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 1024
    stop_token_ids: List[int] = dc_field(default_factory=list)
    cot_text: Optional[str] = None

    system_prompt: Optional[str] = None
    use_system_prompt: Optional[str] = None

    taylor_cache_interval: Optional[int] = None
    taylor_cache_order: Optional[int] = None


@dataclass
class HunyuanImage3ARState:
    """Per-call AR decode state, threaded through the per-token loop."""

    input_ids: torch.Tensor  # [B, T] long; grows by one column per step
    model_kwargs: Dict[str, Any]  # HF-style kwargs threaded across steps
    step_idx: int = 0


class HunyuanImage3ARStep(ARStep[HunyuanImage3Bundle, HunyuanImage3ARConditions, HunyuanImage3ARState]):
    """Per-token transition kernel — owns the model forward."""

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
        """Sample one token from a ``[B, vocab]`` logits tensor."""
        if logits.dim() != 2:
            raise ValueError(f"HunyuanImage3ARStep.sample: expected logits shape [B, vocab], got {tuple(logits.shape)}")

        log_probs_full = F.log_softmax(logits.float(), dim=-1)
        scaled = logits.float() / max(self.temperature, 1e-6)

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

    def init_state(
        self,
        model: HunyuanImage3Bundle,
        conditions: HunyuanImage3ARConditions,
        *,
        max_new_tokens: int,
    ) -> HunyuanImage3ARState:
        """Build the decode state for one ``autoregress`` call. No forward."""
        transformer = model.transformer
        fused = conditions.fused
        input_ids: torch.Tensor = fused.input_ids
        batch_size = int(input_ids.shape[0])

        prompt_len = int(input_ids.shape[1])
        past_kv_initial = self._build_kv_cache(
            transformer, batch_size=batch_size, max_cache_len=prompt_len + int(max_new_tokens)
        )

        cond_vit = conditions.cond_vit
        cond_vit_images = cond_vit.embeds if cond_vit is not None else None
        vit_kwargs: Optional[Dict[str, Any]] = None
        if cond_vit is not None and (cond_vit.spatial_shapes is not None or cond_vit.attn_mask is not None):
            vit_kwargs = {
                "spatial_shapes": cond_vit.spatial_shapes,
                "attention_mask": cond_vit.attn_mask,
            }

        cond_vae = conditions.cond_vae
        cond_vae_images = cond_vae.latents if cond_vae is not None else None

        model_kwargs: Dict[str, Any] = {
            "mode": "gen_text",
            "rope_image_info": [[] for _ in range(batch_size)],
            "attention_mask": fused.attention_mask,  # [B, 1, L, L] bool
            "position_ids": fused.position_ids,  # [B, L] long
            # Unbind stacked RoPE into the model's (cos, sin) pair.
            "custom_pos_emb": (fused.rope_cache[:, 0], fused.rope_cache[:, 1]),
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
        """One token transition: forward, ``sample``, state advance; returns ``(token_id [B], log_prob [B], state)``."""
        transformer = model.transformer
        device = state.input_ids.device
        batch_size = int(state.input_ids.shape[0])
        model_kwargs = state.model_kwargs

        cond_kwargs: Dict[str, Any] = {}
        if state.step_idx == 0:
            cond_kwargs = {
                "cond_vit_images": model_kwargs.get("cond_vit_images"),
                "cond_vit_image_mask": model_kwargs.get("cond_vit_image_mask"),
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

        logits_device = logits.device
        if state.step_idx == 0 and conditions.tokenizer_output is not None:
            real_pos = getattr(conditions.tokenizer_output, "real_pos", None)
            if real_pos is not None:
                real_pos_t = real_pos.to(device=logits_device, dtype=torch.long)
                if real_pos_t.dim() == 2:
                    real_pos_t = real_pos_t[:, -1]
                last_valid = (real_pos_t - 1).clamp(min=0, max=logits.shape[1] - 1)
                next_logits = logits[torch.arange(batch_size, device=logits_device), last_valid]
            else:
                next_logits = logits[:, -1, :]
        else:
            next_logits = logits[:, -1, :]
        if next_logits.device != device:
            next_logits = next_logits.to(device)

        token_id, log_prob = self.sample(next_logits)

        state.input_ids = torch.cat([state.input_ids, token_id.unsqueeze(-1)], dim=1)
        updated = transformer._update_model_kwargs_for_generation(out, model_kwargs)
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
        """Pre-build a ``HunyuanStaticCache`` for the AR loop."""
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
    """Rollout-level AR stage: ``HunyuanImage3ARConditions → TextSegment``."""

    def __init__(
        self,
        *,
        model: HunyuanImage3Bundle,
    ) -> None:
        self.model = model

    def trainable_module(self) -> "torch.nn.Module":
        """Return the bare HI3 decoder — the FSDP/LoRA wrap target."""
        return self.model.transformer.model

    def autoregress(
        self,
        conditions: HunyuanImage3ARConditions,
        *,
        sampling_params: ARSamplingParams,
        params: Optional[HunyuanImage3ARParams] = None,
        **_kwargs: Any,
    ) -> TextSegment:
        """Run AR generation. Returns a varlen-packed ``TextSegment``."""
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
        """Per-token log-prob replay, packed varlen ``[total_tokens]`` aligned with ``segment.log_probs``."""
        fused = conditions.fused
        if fused is None or fused.input_ids is None:
            raise ValueError("HunyuanImage3ARStage.replay: conditions.fused.input_ids is None")
        if segment.tokens is None or segment.cu_seqlens is None or segment.lengths is None:
            raise ValueError(
                "HunyuanImage3ARStage.replay: segment requires tokens with "
                "framework-managed cu_seqlens (construct via TextSegment.pack)"
            )
        if conditions.cond_vit is not None or conditions.cond_vae is not None:
            raise NotImplementedError(
                "HunyuanImage3ARStage.replay: cond-image (i2t / it2i) replay is not "
                "supported — the VAE+ViT cond scatter is not re-applied in the "
                "teacher-forced forward, so per-token logp would omit the image "
                "conditioning the rollout used. In-process comprehension/edit AR RL "
                "needs the cond-image scatter ported into replay first."
            )

        prompt_ids_padded = fused.input_ids
        device = self.model.transformer.model.wte.weight.device
        batch_size = int(prompt_ids_padded.shape[0])

        # Require true prompt lengths; padded lengths corrupt per-token log probabilities.
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
            full_ids = torch.cat([prompt_b, resp_b], dim=0).unsqueeze(0)
            L_full = pl + rl

            causal = torch.tril(torch.ones((L_full, L_full), dtype=torch.bool, device=device))
            mask_4d = torch.full((1, 1, L_full, L_full), neg_inf, dtype=param_dtype, device=device)
            mask_4d.masked_fill_(causal.unsqueeze(0).unsqueeze(0), 0.0)

            # Reset image RoPE state before text-only AR forwards.
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

            raw_logits = logits[0, pl - 1 : pl - 1 + rl, :].float()
            log_probs_full = F.log_softmax(raw_logits, dim=-1)
            flat.append(log_probs_full.gather(-1, resp_b.unsqueeze(-1)).squeeze(-1))

        if not flat:
            return torch.zeros(0, dtype=torch.float32, device=device)
        return torch.cat(flat, dim=0)


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


__all__ = ["HunyuanImage3ARParams", "HunyuanImage3ARStage", "HunyuanImage3ARState", "HunyuanImage3ARStep"]
