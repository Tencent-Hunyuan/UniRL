"""Shared behavior-distribution processing for Qwen3-Omni Talker.

Rollout and replay must call the same processor.  In particular, behavior
log-probabilities are computed *after* repetition penalty, token suppression,
temperature, top-k, and top-p have changed the distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TalkerSamplingConfig:
    temperature: float = 0.9
    top_k: int = 50
    top_p: float = 1.0
    repetition_penalty: float = 1.05
    suppress_token_ids: Tuple[int, ...] = ()
    eos_token_id: Optional[int] = None
    do_sample: bool = True

    def __post_init__(self) -> None:
        if self.temperature < 0.0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        # Match the user-facing generation convention: temperature=0 is a
        # deterministic request, not an invalid division.  Persist the resolved
        # mode so rollout and replay cannot interpret the same payload
        # differently.
        if self.temperature == 0.0 and self.do_sample:
            object.__setattr__(self, "do_sample", False)
        if self.top_k < 0:
            raise ValueError(f"top_k must be >= 0, got {self.top_k}")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")
        if self.repetition_penalty <= 0.0:
            raise ValueError(f"repetition_penalty must be > 0, got {self.repetition_penalty}")

    def to_dict(self) -> dict:
        return {
            "temperature": float(self.temperature),
            "top_k": int(self.top_k),
            "top_p": float(self.top_p),
            "repetition_penalty": float(self.repetition_penalty),
            "suppress_token_ids": tuple(int(token_id) for token_id in self.suppress_token_ids),
            "eos_token_id": None if self.eos_token_id is None else int(self.eos_token_id),
            "do_sample": bool(self.do_sample),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "TalkerSamplingConfig":
        if not isinstance(payload, dict):
            raise TypeError(f"Talker sampling payload must be a dict, got {type(payload).__name__}")
        required = {
            "temperature",
            "top_k",
            "top_p",
            "repetition_penalty",
            "suppress_token_ids",
            "eos_token_id",
            "do_sample",
        }
        missing = sorted(required - set(payload))
        unknown = sorted(set(payload) - required)
        if missing or unknown:
            raise ValueError(
                "Talker sampling payload must be complete and canonical; "
                f"missing={missing}, unknown={unknown}"
            )
        return cls(
            temperature=float(payload["temperature"]),
            top_k=int(payload["top_k"]),
            top_p=float(payload["top_p"]),
            repetition_penalty=float(payload["repetition_penalty"]),
            suppress_token_ids=tuple(int(token_id) for token_id in payload["suppress_token_ids"]),
            eos_token_id=None if payload["eos_token_id"] is None else int(payload["eos_token_id"]),
            do_sample=bool(payload["do_sample"]),
        )


class TalkerSamplingProcessor:
    """HF-compatible Talker logits processor/warper with exact behavior logp."""

    def __init__(self, config: TalkerSamplingConfig) -> None:
        self.config = config

    def _apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        token_history: Optional[torch.Tensor],
    ) -> torch.Tensor:
        penalty = float(self.config.repetition_penalty)
        if penalty == 1.0 or token_history is None or token_history.numel() == 0:
            return logits
        if token_history.dim() != 2 or token_history.shape[0] != logits.shape[0]:
            raise ValueError(
                "token_history must be [B, T] with the same batch size as logits; "
                f"got history={tuple(token_history.shape)}, logits={tuple(logits.shape)}"
            )
        history = token_history.to(device=logits.device, dtype=torch.long)
        if bool(((history < 0) | (history >= logits.shape[-1])).any()):
            raise ValueError("token_history contains ids outside the Talker codec vocabulary")
        selected = torch.gather(logits, 1, history)
        selected = torch.where(selected < 0, selected * penalty, selected / penalty)
        return logits.scatter(1, history, selected)

    def process(
        self,
        logits: torch.Tensor,
        *,
        token_history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return processed float32 logits defining the actual behavior policy."""
        if logits.dim() != 2:
            raise ValueError(f"expected logits [B, vocab], got {tuple(logits.shape)}")
        work = logits.float().clone()
        work = self._apply_repetition_penalty(work, token_history)

        if self.config.suppress_token_ids:
            suppress = torch.as_tensor(self.config.suppress_token_ids, dtype=torch.long, device=work.device)
            if bool(((suppress < 0) | (suppress >= work.shape[-1])).any()):
                raise ValueError("suppress_token_ids contains ids outside the Talker codec vocabulary")
            work.index_fill_(1, suppress, float("-inf"))
        if bool(torch.isneginf(work).all(dim=-1).any()):
            raise RuntimeError("Talker sampling processors removed every token from at least one row")

        if self.config.do_sample:
            work = work / float(self.config.temperature)

            top_k = min(int(self.config.top_k), work.shape[-1])
            if top_k > 0:
                threshold = torch.topk(work, top_k, dim=-1).values[..., -1, None]
                work = work.masked_fill(work < threshold, float("-inf"))

            if self.config.top_p < 1.0:
                # Match transformers.TopPLogitsWarper: sort ascending, remove the
                # low-probability tail whose cumulative mass is <= 1 - top_p,
                # while always retaining at least one token.
                sorted_logits, sorted_indices = torch.sort(work, descending=False, dim=-1)
                cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                sorted_remove = cumulative_probs <= (1.0 - float(self.config.top_p))
                sorted_remove[..., -1:] = False
                remove = torch.zeros_like(sorted_remove).scatter(1, sorted_indices, sorted_remove)
                work = work.masked_fill(remove, float("-inf"))
        else:
            # Greedy decoding is a deterministic behavior policy.  Its exact
            # distribution is a point mass, not softmax(raw_logits).
            selected = work.argmax(dim=-1, keepdim=True)
            greedy = torch.full_like(work, float("-inf"))
            work = greedy.scatter(1, selected, torch.zeros_like(selected, dtype=work.dtype))

        if bool(torch.isneginf(work).all(dim=-1).any()):
            raise RuntimeError("Talker sampling processors removed every token from at least one row")
        return work

    def log_probs(
        self,
        logits: torch.Tensor,
        *,
        token_history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return F.log_softmax(self.process(logits, token_history=token_history), dim=-1)

    def score(
        self,
        logits: torch.Tensor,
        token_ids: torch.Tensor,
        *,
        token_history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        token_ids = token_ids.to(device=logits.device, dtype=torch.long).view(-1, 1)
        if token_ids.shape[0] != logits.shape[0]:
            raise ValueError("token_ids batch size must match logits")
        return self.log_probs(logits, token_history=token_history).gather(1, token_ids).squeeze(1)

    def sample(
        self,
        logits: torch.Tensor,
        *,
        token_history: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        processed = self.process(logits, token_history=token_history)
        if self.config.do_sample:
            token_ids = torch.multinomial(processed.softmax(dim=-1), 1, generator=generator).squeeze(1)
        else:
            token_ids = processed.argmax(dim=-1)
        logp = F.log_softmax(processed, dim=-1).gather(1, token_ids[:, None]).squeeze(1)
        return token_ids, logp

    def is_eos(self, token_ids: torch.Tensor) -> torch.Tensor:
        if self.config.eos_token_id is None:
            return torch.zeros_like(token_ids, dtype=torch.bool)
        return token_ids == int(self.config.eos_token_id)


def suppress_special_codec_ids(*, vocab_size: int, codec_eos_token_id: int) -> tuple[int, ...]:
    """Official Qwen3-Omni suppression set for the upper 1024 codec ids."""
    start = max(0, int(vocab_size) - 1024)
    eos = int(codec_eos_token_id)
    return tuple(token_id for token_id in range(start, int(vocab_size)) if token_id != eos)


def append_token_history(
    token_history: Optional[torch.Tensor],
    token_ids: torch.Tensor,
) -> torch.Tensor:
    token_ids = token_ids.to(dtype=torch.long).view(-1, 1)
    if token_history is None:
        return token_ids
    return torch.cat((token_history.to(device=token_ids.device, dtype=torch.long), token_ids), dim=1)


__all__ = [
    "TalkerSamplingConfig",
    "TalkerSamplingProcessor",
    "append_token_history",
    "suppress_special_codec_ids",
]
