"""Qwen3 SFT task adapter — autoregressive next-token cross-entropy.

The AR counterpart to the diffusion SFT tasks: proves the generic SFT skeleton
(:mod:`unirl.train.sft`) is model-agnostic by driving a causal-LM family whose
supervised loss is next-token cross-entropy over the response tokens (prompt
tokens masked out), NOT flow-matching MSE.

Reuses the existing Qwen3 RL machinery wholesale:

* :meth:`Qwen3Bundle.from_config` loads the ``AutoModelForCausalLM`` + tokenizer.
* :func:`unirl.models.qwen3.ar._replay_aware_forward` is the dual-mode forward
  the AR RL path already installs on the transformer. Called WITH
  ``response_tokens`` + ``prompt_len`` it returns ``[B, T_max]`` FP32 per-token
  log-probs (``logp = logit[tok] - logsumexp``), computed via a chunked
  ``lm_head`` (no full ``[B, L, vocab]`` materialization) and gradient
  checkpointing — memory-safe under FSDP2, and it already disables the flaky
  cuDNN-SDP backward. SFT cross-entropy is then simply ``-logp.mean()`` over the
  response, so we reuse this verbatim rather than reimplementing a shift-label CE.

Batch note: the SFT policy calls ``compute_loss`` per record (micro-batch = 1,
grad-accum across the shard), so every forward here is ``B = 1``.

Record schema (one JSONL row): ``{"sample_id": str, "prompt": str, "response": str}``.
"""

from __future__ import annotations

import logging
from types import MethodType
from typing import Any, Dict, Optional, Tuple

import torch

from unirl.models.qwen3.ar import _replay_aware_forward
from unirl.models.qwen3.bundle import Qwen3Bundle
from unirl.models.qwen3.config import Qwen3PipelineConfig
from unirl.train.sft.task import SFTTaskBase
from unirl.utils.dtypes import parse_torch_dtype

logger = logging.getLogger(__name__)


class Qwen3SFTTask(SFTTaskBase):
    """Instruction-tuning / chat SFT for Qwen3: masked next-token cross-entropy."""

    block_class_names: Tuple[str, ...] = ("Qwen3DecoderLayer",)

    def __init__(self, *, bundle: Qwen3Bundle, config: Qwen3PipelineConfig) -> None:
        self.bundle = bundle
        self.config = config
        self.tokenizer = bundle.tokenizer
        self.autocast_dtype = parse_torch_dtype(
            getattr(config, "autocast_precision", "bf16"), field_name="Qwen3SFTTask.autocast_precision"
        )
        # Install the dual-mode forward the AR path uses (idempotent via the
        # ``__func__`` identity check). SFT does not build a Qwen3ARStage, so we
        # install it here; it wins over the class forward, survives the FSDP2
        # class swap + LoRA injection, and returns per-token logp when called
        # with ``response_tokens``.
        transformer = bundle.transformer
        if getattr(transformer.forward, "__func__", None) is not _replay_aware_forward:
            transformer.forward = MethodType(_replay_aware_forward, transformer)

    @classmethod
    def from_config(cls, config: Qwen3PipelineConfig) -> "Qwen3SFTTask":
        return cls(bundle=Qwen3Bundle.from_config(config), config=config)

    # ------------------------------------------------------------------
    # Data loading (worker-side; the record carries text, tokens build here)
    # ------------------------------------------------------------------

    def load_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        prompt = record.get("prompt")
        response = record.get("response")
        if not isinstance(prompt, str) or not isinstance(response, str):
            raise ValueError(
                f"Qwen3SFTTask record needs str 'prompt' and 'response'; "
                f"got prompt={type(prompt).__name__}, response={type(response).__name__}"
            )
        tok = self.tokenizer
        # Chat-template the prompt (adds the generation prompt) so training matches
        # how the model is prompted at inference; the response is appended raw + EOS.
        prompt_ids = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
        )
        response_ids = tok.encode(response, add_special_tokens=False)
        eos = tok.eos_token_id
        if eos is not None:
            response_ids = list(response_ids) + [eos]
        loaded = dict(record)
        loaded["prompt_ids"] = list(prompt_ids)
        loaded["response_ids"] = list(response_ids)
        return loaded

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_loss(
        self, loaded: Dict[str, Any], *, generator: Optional[torch.Generator] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Masked next-token CE for ONE sample: ``-mean(logp)`` over response tokens."""
        device = self.bundle.device
        prompt_ids = loaded["prompt_ids"]
        response_ids = loaded["response_ids"]
        prompt_len = len(prompt_ids)
        if prompt_len == 0 or len(response_ids) == 0:
            raise ValueError("Qwen3SFTTask.compute_loss: empty prompt or response after tokenization")

        full_ids = torch.tensor([prompt_ids + response_ids], dtype=torch.long, device=device)  # [1, P+T]
        response_tokens = torch.tensor([response_ids], dtype=torch.long, device=device)  # [1, T]
        attention_mask = torch.ones_like(full_ids)
        position_ids = torch.arange(full_ids.shape[1], device=device).unsqueeze(0)

        self.bundle.transformer.train()
        # Returns [1, T] FP32 per-token log-probs at the response positions.
        logp = self.bundle.transformer(
            input_ids=full_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            response_tokens=response_tokens,
            prompt_len=prompt_len,
            temperature=1.0,
            autocast_dtype=(self.autocast_dtype if device.type == "cuda" else None),
        )
        loss = -logp.mean()
        metrics = {
            "loss/total": float(loss.detach().item()),
            "train/ppl": float(torch.exp(loss.detach()).item()),
            "train/response_len": float(len(response_ids)),
        }
        return loss, metrics

    # ------------------------------------------------------------------
    # Eval sampling — all FSDP ranks enter (collective weights); rank 0 kept.
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self, loaded: Dict[str, Any], *, generator: Optional[torch.Generator] = None
    ) -> Dict[str, Any]:
        device = self.bundle.device
        prompt_ids = torch.tensor([loaded["prompt_ids"]], dtype=torch.long, device=device)
        self.bundle.transformer.eval()
        # No response_tokens -> the dual-mode forward delegates to the stock
        # class forward, so ``generate`` works normally.
        out = self.bundle.transformer.generate(
            input_ids=prompt_ids,
            attention_mask=torch.ones_like(prompt_ids),
            max_new_tokens=int(getattr(self.config, "sample_max_new_tokens", 64)),
            do_sample=False,
        )
        gen_ids = out[0, prompt_ids.shape[1] :].tolist()
        self.bundle.transformer.train()
        return {"generated_text": self.tokenizer.decode(gen_ids, skip_special_tokens=True)}


__all__ = ["Qwen3SFTTask"]
