"""Bagel AR (text-out) stage: typed params + per-token kernel + rollout-level stage.

Serves t2t / i2t / it2t — the conditions' ordered prompt splits decide what the
context contains; the stage is split-agnostic. Three classes:

- ``BagelARParams`` — request-shape knobs beyond ``ARSamplingParams``
  (extra ``stop_token_ids``; mirrors ``Qwen3ARParams``).
- ``BagelARStep`` — per-token sampling kernel (full-softmax log-prob captured
  BEFORE top-k/top-p truncation, matching replay's untempered convention;
  verbatim ``Qwen3ARStep`` mechanics).
- ``BagelARStage`` — implements ``ARStage[BagelARConditions]``. Per sample
  (navit bs=1): prefill the KV context from the RAW prompt splits via the
  ``rl_ops`` adapters, then per-token decode (``rl_ops.decode_text``) for
  rollout, or one teacher-forced grad-capable pass (``rl_ops.score_response``)
  for replay. Rollout and replay derive the context from the SAME stored splits
  through the SAME prefill code → prefix K/V parity by construction.

Replay regime: eval() mode with grads enabled (the navit modules dispatch
``forward_train``/``forward_inference`` on ``self.training``) — identical to
``BagelDiffusionStage.replay``. The caller owns the grad scope; the stage owns
only the autocast scope.

This module deliberately avoids importing the vendored modeling (and its hard
``flash_attn`` dependency) at module load — it reaches the model through the
bundle instance at call time — so ``import unirl.models.bagel`` stays CPU-clean.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from unirl.config.require import require
from unirl.models.types.ar import ARStage, ARStep
from unirl.types.sampling import ARSamplingParams
from unirl.types.segments import TextSegment
from unirl.utils.dtypes import parse_torch_dtype

from . import rl_ops
from .conditions import BagelARConditions

if TYPE_CHECKING:
    from .bundle import BagelBundle


@dataclass
class BagelARParams:
    """Per-request AR-mode knobs for Bagel.

    ``stop_token_ids`` is unioned with ``new_token_ids['eos_token_id']``
    (``<|im_end|>``) inside the stage so callers don't repeat the EOS id; the
    sampling shape (temperature / top_p / top_k / max_new_tokens) rides the
    shared :class:`ARSamplingParams`.
    """

    stop_token_ids: List[int] = dc_field(default_factory=list)


class BagelARStep(ARStep):
    """Per-token sampling kernel (``Qwen3ARStep`` mechanics).

    Given logits over the vocabulary at the current position, sample the next
    token and return its log-probability under the *temperature-scaled* full
    softmax computed BEFORE top-k/top-p truncation, so it matches replay's
    ``log_softmax(logits / T)`` without filter masking.
    """

    def __init__(self, *, temperature: float = 1.0, top_p: float = 1.0, top_k: int = 0) -> None:
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)

    def step(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if logits.dim() != 2:
            raise ValueError(f"BagelARStep.step: expected logits shape [B, vocab], got {tuple(logits.shape)}")

        if self.temperature <= 0.0:
            # Greedy: argmax under the full (untempered) softmax.
            log_probs_full = F.log_softmax(logits.float(), dim=-1)
            token_id = log_probs_full.argmax(dim=-1)
            log_prob = log_probs_full.gather(-1, token_id.unsqueeze(-1)).squeeze(-1)
            return token_id, log_prob

        scaled = logits.float() / self.temperature

        # Behavior log-prob under the temperature-scaled distribution, BEFORE
        # top-k/top-p truncation, so old_logp == replay new_logp at the same
        # weights (rl_ops.score_response computes log_softmax(logits / T)).
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


class BagelARStage(ARStage[BagelARConditions]):
    """Rollout-level AR stage for Bagel (t2t / i2t / it2t).

    Drives the MoT und path through the ``rl_ops`` navit adapters: per-sample
    context prefill from raw splits, per-token decode with KV cache (rollout),
    one-shot teacher-forced scoring (replay). Packs per-sample ``(tokens,
    log_probs)`` into a varlen :class:`TextSegment`.
    """

    def __init__(
        self,
        *,
        model: "BagelBundle",
        autocast_precision: str = "bf16",
        logprob_precision: str = "fp32",
        replay_mode: str = "train",
    ) -> None:
        self.model = model
        self.autocast_dtype = parse_torch_dtype(autocast_precision, field_name="BagelARStage.autocast_precision")
        self.logprob_dtype = parse_torch_dtype(logprob_precision, field_name="BagelARStage.logprob_precision")
        # Replay scorer for the GRPO ratio's new_logp:
        #   "train"     — one grad forward_train per sample (nested mask: image full +
        #                 text causal); the und path INCLUDING the image is trained.
        #                 Exact attention, but a different kernel than the rollout's
        #                 forward_inference, so the on-policy ratio is ~1±1e-2.
        #   "inference" — image prefilled no_grad (frozen) + one grad forward_inference
        #                 over [prompt+response]; same kernel as rollout (ratio ~1),
        #                 FSDP-safe, but the image understanding is not trained.
        self.replay_mode = str(replay_mode).strip().lower()
        require(
            self.replay_mode in ("train", "inference"),
            f"BagelARStage: replay_mode must be 'train' or 'inference'; got {replay_mode!r}.",
        )
        # A checkpoint trained with freeze_und detaches the und hidden states in
        # forward_train — a signal the und path was never meant to train. The
        # inference-path replay is unaffected mechanically, but fail loudly.
        # (Chain guarded so fake bundles without the full config tree construct.)
        llm_cfg = getattr(getattr(getattr(model, "model", None), "config", None), "llm_config", None)
        require(
            not getattr(llm_cfg, "freeze_und", False),
            "BagelARStage: llm_config.freeze_und=True — this checkpoint's und path is frozen by design.",
        )

    def trainable_module(self) -> "torch.nn.Module":
        """The MoT transformer (``bundle.transformer``) — same root the diffusion
        stage exposes, so LoRA/FSDP injection is visible to both stages."""
        return self.model.transformer

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _autocast_ctx(self, device: torch.device):
        if device.type == "cuda" and self.autocast_dtype in (torch.float16, torch.bfloat16):
            return torch.autocast("cuda", self.autocast_dtype)
        return nullcontext()

    def _prefill(self, splits: List[Dict[str, Any]], *, device: torch.device) -> Dict[str, Any]:
        """Build a fresh KV context from ordered raw splits (rollout AND replay).

        Grad behavior follows the ambient mode: under the rollout ``no_grad`` the
        prefill is grad-free; under replay's ``enable_grad`` it carries gradients
        (the und path is the trained surface — see ``BagelARConditions``).
        """
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

    def _resolve_stop_ids(self, params: Optional[BagelARParams], sampling_params: ARSamplingParams) -> List[int]:
        ids: List[int] = []
        if params is not None and params.stop_token_ids:
            ids.extend(int(t) for t in params.stop_token_ids)
        if sampling_params.stop_token_id is not None:
            ids.append(int(sampling_params.stop_token_id))
        ids.append(int(self.model.new_token_ids["eos_token_id"]))  # <|im_end|>, as in the vendored gen_text
        return list(dict.fromkeys(ids))

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    def autoregress(
        self,
        conditions: BagelARConditions,
        *,
        sampling_params: ARSamplingParams,
        params: Optional[BagelARParams] = None,
        **_kwargs: Any,
    ) -> TextSegment:
        """Run AR generation per sample. Returns a varlen-packed ``TextSegment``."""
        require(conditions.batch_size > 0, "BagelARStage.autoregress: empty conditions.")
        bagel = self.model.model
        device = torch.device(self.model.device)
        rl_ops.require_inference_dispatch(bagel)

        step = BagelARStep(
            temperature=float(sampling_params.temperature),
            top_p=float(sampling_params.top_p),
            top_k=int(sampling_params.top_k),
        )
        stop_ids = self._resolve_stop_ids(params, sampling_params)
        start_id = int(self.model.new_token_ids["bos_token_id"])

        generated: List[List[int]] = []
        logps: List[List[float]] = []
        with torch.no_grad(), self._autocast_ctx(device):
            for splits in conditions.prompt_splits:
                ctx = self._prefill(splits, device=device)
                tokens_i, logps_i = rl_ops.decode_text(
                    bagel,
                    ctx,
                    start_token_id=start_id,
                    sample_fn=step.step,
                    max_new_tokens=int(sampling_params.max_new_tokens),
                    stop_ids=stop_ids,
                    device=device,
                )
                generated.append(tokens_i)
                logps.append(logps_i)

        return TextSegment.pack(
            tokens=[torch.tensor(t, dtype=torch.long, device=device) for t in generated],
            log_probs=[torch.tensor(lp, dtype=torch.float32, device=device) for lp in logps],
        )

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------

    @staticmethod
    def _split_image_and_prompt_ids(splits: List[Dict[str, Any]]) -> Tuple[Optional[Any], List[int]]:
        """From a sample's ordered splits, pull the (single) ViT image tensor and the
        concatenated prompt token ids (already ``[bos]+enc+[eos]`` wrapped)."""
        image: Optional[Any] = None
        prompt_ids: List[int] = []
        for sp in splits:
            kind = sp.get("kind")
            if kind == "vit":
                image = sp["image"]
            elif kind == "text":
                prompt_ids.extend(int(t) for t in sp["ids"].tolist())
            else:
                raise ValueError(f"BagelARStage.replay: unknown split kind {kind!r}.")
        return image, prompt_ids

    def replay(
        self,
        conditions: BagelARConditions,
        *,
        segment: TextSegment,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Per-token grad-capable log-prob replay (``new_logp``) over a stored segment.

        Dispatches on ``self.replay_mode``: ``"train"`` (one grad
        ``forward_train`` per sample, image+text trained, nested mask) or
        ``"inference"`` (no_grad frozen-image prefill + one grad
        ``forward_inference`` over ``[prompt+response]``, kernel-matched to rollout).
        Returns packed varlen ``[total_tokens]`` aligned with ``segment.log_probs``.
        Each mode sets the language-model train/eval mode it needs and does NOT
        restore it: the navit dispatch (and activation-checkpoint recompute in
        backward) reads ``self.training``, so the mode must persist through the
        caller's ``backward()``; the rollout engine re-sets eval() around every
        ``generate``. Empty-response samples contribute zero tokens.
        """
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
        start_id = int(self.model.new_token_ids["bos_token_id"])

        if self.replay_mode == "train":
            parts = self._replay_train(
                conditions,
                segment=segment,
                cu=cu,
                lengths=lengths,
                start_id=start_id,
                temperature=temperature,
                device=device,
            )
        else:
            parts = self._replay_inference(
                conditions,
                segment=segment,
                cu=cu,
                lengths=lengths,
                start_id=start_id,
                temperature=temperature,
                device=device,
            )
        if not parts:
            return torch.zeros(0, dtype=self.logprob_dtype, device=device)
        return torch.cat(parts, dim=0).to(dtype=self.logprob_dtype)

    def _replay_train(
        self,
        conditions: BagelARConditions,
        *,
        segment: TextSegment,
        cu: List[int],
        lengths: List[int],
        start_id: int,
        temperature: float,
        device: torch.device,
    ) -> List[torch.Tensor]:
        """Train-mode replay: one grad ``forward_train`` per sample over ``[image | prompt |
        response_input]`` with the image-full/text-causal nested mask. Trains the und
        path including the image. forward_train ≠ rollout's forward_inference, so the
        on-policy ratio carries a ~1e-2 kernel gap (see the recipe note)."""
        bagel = self.model.model
        new_token_ids = self.model.new_token_ids
        temp = float(temperature) if float(temperature) > 0.0 else 1.0
        bagel.language_model.train()  # navit forward_train dispatch; persists through backward
        parts: List[torch.Tensor] = []
        with self._autocast_ctx(device):
            for i, splits in enumerate(conditions.prompt_splits):
                n = lengths[i]
                if n == 0:
                    continue
                response = segment.tokens[cu[i] : cu[i] + n].to(device=device, dtype=torch.long)
                image, prompt_ids = self._split_image_and_prompt_ids(splits)
                response_input = torch.cat(
                    [torch.tensor([start_id], device=device, dtype=torch.long), response[:-1]], dim=0
                )
                packed = rl_ops.pack_und_forward_inputs(
                    bagel,
                    new_token_ids=new_token_ids,
                    prompt_ids=prompt_ids,
                    image=image,
                    response_input=response_input,
                    device=device,
                )
                logits = rl_ops.und_replay_logits(bagel, packed)  # [n, V]
                logp_full = torch.log_softmax(logits.float() / temp, dim=-1)
                parts.append(logp_full.gather(-1, response.unsqueeze(-1)).squeeze(-1))
        return parts

    def _replay_inference(
        self,
        conditions: BagelARConditions,
        *,
        segment: TextSegment,
        cu: List[int],
        lengths: List[int],
        start_id: int,
        temperature: float,
        device: torch.device,
    ) -> List[torch.Tensor]:
        """Inference-mode replay: image prefilled under no_grad (frozen), then one grad
        ``forward_inference`` over ``[prompt+response]``. Kernel-matched to rollout
        (ratio ~1), FSDP-safe (single grad forward), image not trained."""
        bagel = self.model.model
        bagel.language_model.eval()  # navit forward_inference dispatch
        rl_ops.require_inference_dispatch(bagel)
        parts: List[torch.Tensor] = []
        with self._autocast_ctx(device):
            for i, splits in enumerate(conditions.prompt_splits):
                n = lengths[i]
                if n == 0:
                    continue
                response = segment.tokens[cu[i] : cu[i] + n]
                image, prompt_ids = self._split_image_and_prompt_ids(splits)
                with torch.no_grad():
                    ctx = rl_ops.init_und_context(bagel)
                    if image is not None:
                        ctx = rl_ops.prefill_vit_split(
                            bagel, ctx, image_tensor=image, new_token_ids=self.model.new_token_ids, device=device
                        )
                parts.append(
                    rl_ops.score_response_with_prompt(
                        bagel,
                        ctx,
                        prompt_ids=torch.tensor(prompt_ids, dtype=torch.long, device=device),
                        response_ids=response,
                        start_token_id=start_id,
                        temperature=float(temperature),
                        device=device,
                    )
                )
        return parts


__all__ = ["BagelARParams", "BagelARStage", "BagelARStep"]
