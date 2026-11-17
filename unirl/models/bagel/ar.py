"""Bagel AR (text-out) stage: per-token kernel + rollout-level stage."""

from __future__ import annotations

from contextlib import nullcontext
from functools import partial
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from unirl.config.require import require
from unirl.models.types.ar import ARStage, ARStep, pack_generated_text_segment
from unirl.types.sampling import ARSamplingParams
from unirl.types.segments import TextSegment
from unirl.utils.dtypes import parse_torch_dtype

from . import rl_ops
from .conditions import BagelARConditions

if TYPE_CHECKING:
    from .bundle import BagelBundle


class BagelARStep(ARStep):
    """Per-token sampling kernel (``Qwen3ARStep`` mechanics)."""

    def __init__(self, *, temperature: float = 1.0, top_p: float = 1.0, top_k: int = 0) -> None:
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)

    def step(
        self,
        logits: torch.Tensor,
        *,
        generators: Optional[List[torch.Generator]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if logits.dim() != 2:
            raise ValueError(f"BagelARStep.step: expected logits shape [B, vocab], got {tuple(logits.shape)}")

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
            sorted_vals = sorted_vals.masked_fill(cutoff > 0, float("-inf"))
            scaled = torch.full_like(scaled, float("-inf")).scatter(-1, sorted_idx, sorted_vals)

        probs = F.softmax(scaled, dim=-1)
        if generators is None:
            token_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            if len(generators) != int(probs.shape[0]):
                raise ValueError(f"BagelARStep.step: got {len(generators)} generators for batch {probs.shape[0]}.")
            token_id = torch.cat(
                [
                    torch.multinomial(probs[row], num_samples=1, generator=generator)
                    for row, generator in enumerate(generators)
                ]
            )
        log_prob = log_probs_full.gather(-1, token_id.unsqueeze(-1)).squeeze(-1)
        return token_id, log_prob


class BagelARStage(ARStage[BagelARConditions]):
    """Rollout-level AR stage for Bagel (t2t / i2t / it2t)."""

    def __init__(
        self,
        *,
        model: "BagelBundle",
        autocast_precision: str = "bf16",
        logprob_precision: str = "fp32",
        replay_mode: str = "train",
        replay_attention_backend: str = "sdpa",
        forward_batch_size: int,
    ) -> None:
        self.model = model
        self.autocast_dtype = parse_torch_dtype(autocast_precision, field_name="BagelARStage.autocast_precision")
        self.logprob_dtype = parse_torch_dtype(logprob_precision, field_name="BagelARStage.logprob_precision")
        self.forward_batch_size = int(forward_batch_size)
        self.replay_mode = str(replay_mode).strip().lower()
        require(
            self.replay_mode in ("train", "inference"),
            f"BagelARStage: replay_mode must be 'train' or 'inference'; got {replay_mode!r}.",
        )
        self.replay_attention_backend = str(replay_attention_backend).strip().lower()
        require(
            self.replay_attention_backend in ("sdpa", "flex"),
            f"BagelARStage: replay_attention_backend must be 'sdpa' or 'flex'; got {replay_attention_backend!r}.",
        )
        require(
            not model.model.config.llm_config.freeze_und,
            "BagelARStage: llm_config.freeze_und=True — this checkpoint's und path is frozen by design.",
        )

    def trainable_module(self) -> "torch.nn.Module":
        """The MoT transformer — the same root the diffusion stage exposes, so LoRA/FSDP is visible to both stages."""
        return self.model.transformer

    def _autocast_ctx(self, device: torch.device):
        if device.type == "cuda" and self.autocast_dtype in (torch.float16, torch.bfloat16):
            return torch.autocast("cuda", self.autocast_dtype)
        return nullcontext()

    def _prefill(self, splits: List[Dict[str, Any]], *, device: torch.device) -> Dict[str, Any]:
        """Build a fresh KV context from ordered raw splits (rollout AND replay)."""
        bagel = self.model.model
        ctx = rl_ops.init_und_context(bagel)
        for sp in splits:
            kind = sp.get("kind")
            if kind == "text":
                ctx = rl_ops.prefill_text_split(bagel, ctx, text_ids=sp["ids"], device=device)
            elif kind == "vit":
                ctx = rl_ops.prefill_vit_split(
                    bagel, ctx, image_tensor=sp["image"], new_token_ids=self.model.new_token_ids, device=device
                )
            else:
                raise ValueError(f"BagelARStage: unknown prompt split kind {kind!r}; expected 'text' or 'vit'.")
        return ctx

    def autoregress(
        self,
        conditions: BagelARConditions,
        *,
        sampling_params: ARSamplingParams,
        **_kwargs: Any,
    ) -> TextSegment:
        """Run AR generation per sample. Returns a varlen-packed ``TextSegment``."""
        require(conditions.batch_size > 0, "BagelARStage.autoregress: empty conditions.")
        bagel = self.model.model
        device = torch.device(self.model.device)
        rl_ops.require_inference_dispatch(bagel)

        step = BagelARStep(
            temperature=sampling_params.temperature,
            top_p=sampling_params.top_p,
            top_k=sampling_params.top_k,
        )
        eos_id = self.model.new_token_ids["eos_token_id"]
        stop_ids = (
            list(dict.fromkeys([sampling_params.stop_token_id, eos_id]))
            if sampling_params.stop_token_id is not None
            else [eos_id]
        )
        start_id = self.model.new_token_ids["bos_token_id"]

        text_id_lists: Optional[List[torch.Tensor]] = []
        for splits in conditions.prompt_splits:
            ids: List[torch.Tensor] = []
            for split in splits:
                if split.get("kind") != "text":
                    text_id_lists = None
                    break
                ids.append(split["ids"].reshape(-1).to(dtype=torch.long))
            if text_id_lists is None or not ids:
                text_id_lists = None
                break
            text_id_lists.append(torch.cat(ids))
        use_batched = self.forward_batch_size > 1 and text_id_lists is not None and len(text_id_lists) > 1
        generated: List[List[int]] = []
        logps: List[List[float]] = []
        with torch.no_grad(), self._autocast_ctx(device):
            if use_batched:
                for start in range(0, len(text_id_lists), self.forward_batch_size):
                    id_chunk = text_id_lists[start : start + self.forward_batch_size]
                    generators = (
                        rl_ops.fork_sampling_generators(device, len(id_chunk))
                        if step.temperature > 0.0 and len(id_chunk) > 1
                        else None
                    )
                    ctx = rl_ops.prefill_text_batched(bagel, id_chunk, device=device)
                    tokens, token_logps = rl_ops.decode_text_batched(
                        bagel,
                        ctx,
                        start_token_id=start_id,
                        sample_fn=partial(step.step, generators=generators) if generators is not None else step.step,
                        max_new_tokens=sampling_params.max_new_tokens,
                        stop_ids=stop_ids,
                        device=device,
                    )
                    generated.extend(tokens)
                    logps.extend(token_logps)
            else:
                for splits in conditions.prompt_splits:
                    ctx = self._prefill(splits, device=device)
                    tokens_i, logps_i = rl_ops.decode_text(
                        bagel,
                        ctx,
                        start_token_id=start_id,
                        sample_fn=step.step,
                        max_new_tokens=sampling_params.max_new_tokens,
                        stop_ids=stop_ids,
                        device=device,
                    )
                    generated.append(tokens_i)
                    logps.append(logps_i)

        return pack_generated_text_segment(generated, logps, device=device)

    def replay(
        self,
        conditions: BagelARConditions,
        *,
        segment: TextSegment,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Grad-capable log-prob replay, packed varlen ``[total_tokens]`` aligned with ``segment.log_probs``."""
        if segment.tokens is None or segment.cu_seqlens is None or segment.lengths is None:
            raise ValueError(
                "BagelARStage.replay: segment requires tokens with framework-managed "
                "cu_seqlens (construct via TextSegment.pack)"
            )
        require(
            conditions.batch_size == int(segment.lengths.numel()),
            f"BagelARStage.replay: conditions batch ({conditions.batch_size}) != "
            f"segment samples ({int(segment.lengths.numel())}).",
        )
        device = next(self.model.transformer.parameters()).device
        cu = [int(c) for c in segment.cu_seqlens.tolist()]
        lengths = [int(n) for n in segment.lengths.tolist()]
        start_id = self.model.new_token_ids["bos_token_id"]

        if self.replay_mode == "train":
            bagel = self.model.model
            new_token_ids = self.model.new_token_ids
            temp = float(temperature) if float(temperature) > 0.0 else 1.0
            bagel.language_model.train()
            parts: List[torch.Tensor] = []
            with self._autocast_ctx(device):
                for i, splits in enumerate(conditions.prompt_splits):
                    n = lengths[i]
                    if n == 0:
                        continue
                    response = segment.tokens[cu[i] : cu[i] + n].to(device=device, dtype=torch.long)
                    response_input = torch.cat(
                        [torch.tensor([start_id], device=device, dtype=torch.long), response[:-1]], dim=0
                    )
                    packed = rl_ops.pack_und_forward_inputs(
                        bagel,
                        new_token_ids=new_token_ids,
                        splits=splits,
                        response_input=response_input,
                        device=device,
                        attention_backend=self.replay_attention_backend,
                    )
                    logits = rl_ops.und_replay_logits(bagel, packed)
                    logp_full = torch.log_softmax(logits.float() / temp, dim=-1)
                    parts.append(logp_full.gather(-1, response.unsqueeze(-1)).squeeze(-1))
        else:
            bagel = self.model.model
            bagel.language_model.eval()
            rl_ops.require_inference_dispatch(bagel)
            parts = []
            with self._autocast_ctx(device):
                for i, splits in enumerate(conditions.prompt_splits):
                    n = lengths[i]
                    if n == 0:
                        continue
                    response = segment.tokens[cu[i] : cu[i] + n]
                    cut = len(splits)
                    while cut > 0 and splits[cut - 1].get("kind") == "text":
                        cut -= 1
                    for sp in splits[:cut]:
                        if sp.get("kind") not in ("text", "vit"):
                            raise ValueError(f"BagelARStage.replay: unknown split kind {sp.get('kind')!r}.")
                    trailing_ids = [int(t) for sp in splits[cut:] for t in sp["ids"].tolist()]
                    with torch.no_grad():
                        ctx = self._prefill(splits[:cut], device=device)
                    parts.append(
                        rl_ops.score_response_with_prompt(
                            bagel,
                            ctx,
                            prompt_ids=torch.tensor(trailing_ids, dtype=torch.long, device=device),
                            response_ids=response,
                            start_token_id=start_id,
                            temperature=float(temperature),
                            device=device,
                        )
                    )
        if not parts:
            return torch.zeros(0, dtype=self.logprob_dtype, device=device)
        return torch.cat(parts, dim=0).to(dtype=self.logprob_dtype)


__all__ = ["BagelARStage", "BagelARStep"]
