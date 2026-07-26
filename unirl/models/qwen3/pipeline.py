"""Qwen3Pipeline — ``Sample → Sample`` end-to-end for Qwen3.

Implements the AR-only two-tier flow::

    Texts ──chat_template──▶ Qwen3ARConditions ──autoregress──▶ TextSegment
                                                                      │
                                                                      ▼
                                                              tokenizer.decode
                                                                      │
                                                                      ▼
                                                                    Texts

Hydra constructs a pipeline via
``Qwen3Pipeline.from_config(Qwen3PipelineConfig)`` (see ``config.py``);
``from_config`` loads the :class:`Qwen3Bundle` then constructs the two
stages.

No σ schedule
-------------
Qwen3 is a pure causal LM with no diffusion side; ``generate()`` reads no σ
schedule — the hosting engine's σ-pinning is a no-op for AR-only pipelines.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from unirl.models.types.ar import ARSamplingParams
from unirl.models.types.pipeline import Pipeline
from unirl.types.primitives import Texts
from unirl.types.sample import Sample, Turn

from .ar import Qwen3ARParams, Qwen3ARStage
from .bundle import Qwen3Bundle
from .chat_template import Qwen3ChatTemplateStage
from .conditions import Qwen3ARConditions
from .config import Qwen3PipelineConfig


class Qwen3Pipeline(Pipeline):
    """Qwen3 AR generate pipeline: ``Sample → Sample``.

    Consumes a request ``Sample`` whose frontier (last) Part is a pre-forked AR
    gen shell carrying ``ARSamplingParams``. Reads the full role-tagged trajectory
    via ``sample.text_conditioning()`` (one message per turn; an optional
    ``{"system_instruction": str}`` override rides on the input Part's
    ``control["chat"]``) and fills the frontier Part:

    - ``segment: TextSegment`` — the generated tokens + full-softmax log-probs.
    - ``primitive: Texts`` — detokenized response strings.

    ``Part.conditions`` carries the encoded prompt conditions; trainer-side replay
    teacher-forces over those *stored* ids (it re-types them via
    ``conditions_cls.from_dict``), so the encode here is the single source of truth
    and the importance ratio stays consistent.
    """

    def __init__(
        self,
        *,
        bundle: Qwen3Bundle,
        chat_template: Optional[Qwen3ChatTemplateStage] = None,
        ar: Optional[Qwen3ARStage] = None,
        autocast_precision: str = "bf16",
        logprob_precision: str = "fp32",
    ) -> None:
        super().__init__()
        self.bundle = bundle
        # Mirror SD3Pipeline: build the stages from the (shared) bundle when not
        # supplied, so the v2 trainer can construct the pipeline via
        # ``remote_hydra(pipeline_cfg, bundle=...)`` and share ONE bundle across
        # the pipeline (rollout) and the FSDPBackend (training) — required for
        # on-policy trainside PE. ``from_config`` still passes both explicitly.
        self.chat_template = chat_template if chat_template is not None else Qwen3ChatTemplateStage(bundle)
        self.ar = (
            ar
            if ar is not None
            else Qwen3ARStage(model=bundle, autocast_precision=autocast_precision, logprob_precision=logprob_precision)
        )

    @classmethod
    def from_bundle(
        cls,
        bundle: Qwen3Bundle,
        *,
        system_instruction: Optional[str] = None,
        autocast_precision: str = "bf16",
        logprob_precision: str = "fp32",
        enable_thinking: bool = False,
        max_prompt_length: int = 4096,
    ) -> "Qwen3Pipeline":
        """Wire chat-template + AR stages around an already-loaded bundle.

        The v2 trainer loads the bundle once and injects it
        (``remote_hydra(pipeline_cfg, bundle=...)``); ``from_config`` would load a
        second copy. ``system_instruction`` (e.g. ``/no_think``) and
        ``enable_thinking`` are applied to the chat template here so they are
        not lost on the bundle-injected path.
        """
        chat_template = Qwen3ChatTemplateStage(
            bundle,
            system_instruction=system_instruction,
            enable_thinking=enable_thinking,
            max_prompt_length=max_prompt_length,
        )
        ar = Qwen3ARStage(
            model=bundle,
            autocast_precision=autocast_precision,
            logprob_precision=logprob_precision,
        )
        return cls(
            bundle=bundle,
            chat_template=chat_template,
            ar=ar,
            autocast_precision=autocast_precision,
            logprob_precision=logprob_precision,
        )

    @classmethod
    def from_config(cls, config: Qwen3PipelineConfig) -> "Qwen3Pipeline":
        """Build the full pipeline from a config."""
        bundle = Qwen3Bundle.from_config(config)
        chat_template = Qwen3ChatTemplateStage(
            bundle,
            system_instruction=config.system_instruction,
            enable_thinking=config.enable_thinking,
            max_prompt_length=config.max_prompt_length,
        )
        ar = Qwen3ARStage(
            model=bundle,
            autocast_precision=config.autocast_precision,
            logprob_precision=config.logprob_precision,
        )
        return cls(bundle=bundle, chat_template=chat_template, ar=ar)

    def _conditions_for(self, turns: List[Turn], control: Optional[Dict[str, Any]] = None) -> Qwen3ARConditions:
        """Chat-template + tokenize the trajectory ``turns`` → :class:`Qwen3ARConditions`.

        The rollout encode path: production replay teacher-forces over the *stored*
        conditions this produces (not a re-encode). An optional per-request
        ``system_instruction`` override rides on the input Part's ``control["chat"]``.
        """
        chat_overrides: Dict[str, Any] = dict((control or {}).get("chat") or {})
        if "system_instruction" in chat_overrides:
            chat_stage = Qwen3ChatTemplateStage(
                self.bundle,
                system_instruction=chat_overrides["system_instruction"],
                max_prompt_length=self.chat_template.max_prompt_length,
                enable_thinking=self.chat_template.enable_thinking,
            )
        else:
            chat_stage = self.chat_template
        return chat_stage.embed(turns)

    def generate(self, sample: Sample) -> Sample:
        """Run Qwen3 AR generation end-to-end, filling the frontier (pre-forked) gen Part."""
        frontier = sample.parts[-1]
        ar = frontier.sampling_params
        if not isinstance(ar, ARSamplingParams):
            raise TypeError(
                f"Qwen3Pipeline.generate: frontier gen Part must carry ARSamplingParams, "
                f"got {type(ar).__name__ if ar is not None else 'None'}"
            )

        # Full role-tagged trajectory (one turn per message), frontier-aligned —
        # text_conditioning() fails loud on any non-text turn.
        turns = sample.text_conditioning()
        conds = self._conditions_for(turns, sample.parts[0].control)

        # Normalize the gen shell's ARSamplingParams through Qwen3ARParams (parity
        # with the prior req-sourced path: stop_token_id reset, types coerced).
        params = Qwen3ARParams(
            max_tokens=ar.max_new_tokens,
            temperature=ar.temperature,
            top_p=ar.top_p,
            top_k=ar.top_k,
        )
        sampling_params = ARSamplingParams(
            max_new_tokens=int(params.max_tokens),
            temperature=float(params.temperature),
            top_p=float(params.top_p),
            top_k=int(params.top_k),
            stop_token_id=None,
        )

        segment = self.ar.autoregress(conds, sampling_params=sampling_params, params=params)
        decoded = self._detokenize(segment)

        # Fill the frontier shell, carrying the encoded conditions for trainer-side
        # replay: Part.conditions is the train stack's source (GRPO re-types them via
        # conditions_cls.from_dict in compute_loss_and_backward).
        filled = frontier.fill(segment=segment, primitives={"text": decoded}, conditions=conds.to_dict())
        return Sample(parts=[*sample.parts[:-1], filled], reward_compute_s=sample.reward_compute_s)

    def _detokenize(self, segment) -> Texts:
        """Decode each per-sample varlen token chunk via the bundle tokenizer."""
        if segment.tokens is None or segment.cu_seqlens is None:
            return Texts(texts=[])
        cu = [int(c) for c in segment.cu_seqlens.tolist()]
        tokenizer = self.bundle.tokenizer
        out: list = []
        n = len(cu) - 1
        for i in range(n):
            chunk = segment.tokens[cu[i] : cu[i + 1]]
            ids = chunk.tolist() if chunk.numel() > 0 else []
            out.append(tokenizer.decode(ids, skip_special_tokens=True))
        return Texts(texts=out)


__all__ = ["Qwen3Pipeline"]
