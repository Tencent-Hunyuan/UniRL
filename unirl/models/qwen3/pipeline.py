"""Qwen3Pipeline — RolloutReq → RolloutResp end-to-end for Qwen3.

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
Qwen3 is a pure causal LM with no diffusion side. ``generate()`` never
reads ``req.sigmas`` — the hosting engine's
:func:`unirl.sde.runtime.ensure_req_sigmas` call is a no-op
upstream for AR-only pipelines.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from unirl.models.types.ar import ARSamplingParams
from unirl.models.types.pipeline import Pipeline
from unirl.types.primitives import Texts
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp, RolloutTrack
from unirl.types.sampling import get_ar_params

from .ar import Qwen3ARParams, Qwen3ARStage
from .bundle import Qwen3Bundle
from .chat_template import Qwen3ChatTemplateStage
from .conditions import Qwen3ARConditions
from .config import Qwen3PipelineConfig


class Qwen3Pipeline(Pipeline):
    """Qwen3 AR generate pipeline.

    Reads from ``RolloutReq``:

    - ``primitives["text"]: Texts`` — required prompts.
    - ``stage_params["ar"]: dict`` — kwargs for :class:`Qwen3ARParams`
      (``max_tokens`` / ``temperature`` / ``top_p`` / ``top_k`` /
      ``stop_token_ids``).
    - ``stage_params["chat"]: dict`` — optional
      ``{"system_instruction": str}`` override for the chat-template
      stage; when absent the stage's compose-time ``system_instruction``
      is used.

    Writes ``RolloutResp.tracks["ar"]`` (one :class:`RolloutTrack`):

    - ``conditions["prompt"]: TextTokenCondition`` — the chat-template
      output (``input_ids`` + ``attention_mask``).
    - ``segment: TextSegment`` — the generated tokens +
      full-softmax log-probs.
    - ``decoded: Texts`` — detokenized response strings.
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
    ) -> "Qwen3Pipeline":
        """Wire chat-template + AR stages around an already-loaded bundle.

        The v2 trainer loads the bundle once and injects it
        (``remote_hydra(pipeline_cfg, bundle=...)``); ``from_config`` would load a
        second copy. ``system_instruction`` (e.g. ``/no_think``) and
        ``enable_thinking`` are applied to the chat template here so they are
        not lost on the bundle-injected path.
        """
        chat_template = Qwen3ChatTemplateStage(
            bundle, system_instruction=system_instruction, enable_thinking=enable_thinking
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
        )
        ar = Qwen3ARStage(
            model=bundle,
            autocast_precision=config.autocast_precision,
            logprob_precision=config.logprob_precision,
        )
        return cls(bundle=bundle, chat_template=chat_template, ar=ar)

    def generate(self, req: RolloutReq) -> RolloutResp:
        """Run Qwen3 AR generation end-to-end."""
        texts = req.primitives.get("text")
        if not isinstance(texts, Texts):
            raise TypeError(
                f"Qwen3Pipeline.generate: req.primitives['text'] must be Texts, "
                f"got {type(texts).__name__ if texts is not None else 'None'}"
            )

        # Optional per-request system-instruction override.
        chat_overrides: Dict[str, Any] = dict(req.stage_config.get("chat") or {})
        if "system_instruction" in chat_overrides:
            chat_stage = Qwen3ChatTemplateStage(
                self.bundle,
                system_instruction=chat_overrides["system_instruction"],
                max_prompt_length=self.chat_template.max_prompt_length,
                enable_thinking=self.chat_template.enable_thinking,
            )
        else:
            chat_stage = self.chat_template

        conds: Qwen3ARConditions = chat_stage.embed(texts)

        # Extract typed AR sampling params from the request.
        ar = get_ar_params(req.sampling_params)
        if ar is not None:
            params = Qwen3ARParams(
                max_tokens=ar.max_new_tokens,
                temperature=ar.temperature,
                top_p=ar.top_p,
                top_k=ar.top_k,
            )
        else:
            params = Qwen3ARParams()

        sampling_params = ARSamplingParams(
            max_new_tokens=int(params.max_tokens),
            temperature=float(params.temperature),
            top_p=float(params.top_p),
            top_k=int(params.top_k),
            stop_token_id=None,
        )

        segment = self.ar.autoregress(conds, sampling_params=sampling_params, params=params)

        decoded = self._detokenize(segment)

        return RolloutResp(
            tracks={
                "ar": RolloutTrack(
                    sample_ids=list(req.sample_ids),
                    parent_ids=list(req.group_ids),
                    conditions=conds.to_dict(),
                    segment=segment,
                    decoded=decoded,
                ),
            }
        )

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
