"""SGLang LLM rollout-engine configuration (new ``BaseRolloutEngine`` protocol).

Consumed by :class:`SGLangLLMRolloutEngine`; the rollout actor (or, as a
composed-engine child, :meth:`SGLangLLMEngineConfig.make_engine`) constructs it
with ``config=<this>`` + ``device`` / ``strategy`` / ``rank`` / ``model_config``.

Unlike :class:`unirl.rollout.engine.sglang_diffusion.config.SGLangDiffusionEngineConfig`
(which delegates the model path to the per-actor ``model_config``), this
LLM engine carries its own ``pretrained_model_ckpt_path`` field — PE-style
LLM rollouts don't surface a diffusion ``ModelBundleConfig``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from unirl.config.require import require
from unirl.rollout.engine.base import BaseEngineConfig


@dataclass
class SGLangLLMEngineConfig(BaseEngineConfig):
    """Configuration for the SGLang LLM (SRT) rollout engine."""

    def make_engine(self, **deps: Any):
        from unirl.rollout.engine.sglang_llm.engine import SGLangLLMRolloutEngine

        return SGLangLLMRolloutEngine(config=self, **deps)

    # --- Model ---
    pretrained_model_ckpt_path: str = ""

    # --- Parallelism & GPU ---
    tp_size: Optional[int] = None

    # --- SGLang network ---
    host: Optional[str] = None
    port: Optional[int] = None

    # --- Concurrency / async ---
    concurrency: int = 8

    # --- Sample expansion contract ---
    # ARTrainer pre-expands the request by samples_per_prompt (P prompts → P*N
    # entries, one per GRPO sibling), so the engine must emit exactly ONE
    # completion per entry (n=1) — matching the trainside pipeline, else samples
    # double-count (P*N entries × N each). Standalone callers (e.g. the smoke
    # driver) pass unexpanded prompts and want the engine to fan out
    # n=samples_per_prompt itself; they leave this False.
    samples_pre_expanded: bool = False

    # --- VLM multimodal ---
    # Image token placeholder injected into the chat template at image
    # positions.  Model-specific: e.g. "<|vision_start|><|image_pad|><|vision_end|>"
    # for Qwen2.5-VL.  None (default) = text-only mode.
    image_token: Optional[str] = None

    # --- LLM sampling (forwarded to SGLang /generate sampling_params) ---
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    # top_k truncation — must match the trainside ARSamplingParams default (1024)
    # so the sglang rollout samples from the SAME distribution. Was never forwarded
    # before (sglang defaulted to -1 = full vocab) → engine sampling mismatch.
    top_k: int = 1024

    # --- Chat template ---
    # System message prepended to every prompt (e.g. "/no_think" to suppress
    # Qwen3's thinking mode), used as the fallback when a per-request stage
    # config doesn't carry one. Must match the trainside pipeline's
    # system_instruction so generation and replay see the same prompt.
    system_instruction: Optional[str] = None
    # Extra kwargs forwarded to tokenizer.apply_chat_template (e.g.
    # {enable_thinking: false} for Qwen3 — without it the model emits a long
    # <think> block that overruns max_new_tokens before reaching the answer).
    chat_template_kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)

    # --- Escape hatch for advanced ServerArgs / engine knobs ---
    engine_kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.engine_kwargs is None:
            self.engine_kwargs = {}
        require(
            bool(self.pretrained_model_ckpt_path),
            "SGLangLLMEngineConfig.pretrained_model_ckpt_path must be set",
        )
        require(
            self.tp_size is None or self.tp_size >= 1,
            f"SGLangLLMEngineConfig.tp_size must be >= 1 when set; got {self.tp_size!r}",
        )
        require(
            self.concurrency >= 1,
            f"SGLangLLMEngineConfig.concurrency must be >= 1; got {self.concurrency!r}",
        )
        require(
            self.max_new_tokens >= 1,
            f"SGLangLLMEngineConfig.max_new_tokens must be >= 1; got {self.max_new_tokens!r}",
        )
        require(
            self.temperature > 0,
            f"SGLangLLMEngineConfig.temperature must be > 0; got {self.temperature!r}",
        )
        require(
            0.0 < self.top_p <= 1.0,
            f"SGLangLLMEngineConfig.top_p must be in (0, 1]; got {self.top_p!r}",
        )


__all__ = ["SGLangLLMEngineConfig"]
