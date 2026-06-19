"""PEPipeline — RolloutReq → RolloutResp end-to-end for Prompt Enhancement.

Implements the two-phase composed flow with sampling-param-driven
fan-out (``N = ar.samples_per_prompt`` rewrites/prompt, ``M =
diffusion.samples_per_prompt`` images/rewrite)::

    P prompts ──llm.generate──▶ P*N rewrites ──diffusion.generate──▶ P*N*M images
                                 (track "ar")                         (track "diffusion")

PE composes two child :class:`Pipeline` instances at the *pipeline*
layer, not the stage layer. Each child remains a fully self-contained
unit (its bundle, stages, CFG-empty-negative handling, etc.) and is
reusable in non-PE pipelines. PE's job is request fan-out, sequencing,
lineage, and response merging. The child pipelines are 1:1 — PE
replicates inputs (prompt ×N, each rewrite ×M) so the branch factors
become lineage (``parent_ids`` / ``parent_track``) for GRPO grouping.

σ schedule contract
-------------------
Forwarded verbatim to the diffusion child. The LLM child never reads
``req.sigmas`` (see :class:`Qwen3Pipeline.generate`). The hosting engine
adapter pins ``req.sigmas`` on the parent PE request via
:func:`unirl.sde.runtime.ensure_req_sigmas` before calling
``pe_pipeline.generate``; PE then passes that schedule through to the
diffusion sub-request unchanged.
"""

from __future__ import annotations

import logging
from typing import Optional

from unirl.models.types.pipeline import Pipeline
from unirl.types.primitives import Texts
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp, _track_with_field

from .bundle import PEBundle
from .instruction import postprocess_pe_texts

logger = logging.getLogger(__name__)


class PEPipeline(Pipeline):
    """PE generate pipeline.

    Reads from ``RolloutReq``:

    - ``primitives["text"]: Texts`` — raw user prompts, fed to the LLM.
    - ``sampling_params: Dict[str, BaseSamplingParams]`` — the ``"ar"`` entry
      (``ARSamplingParams``, ``samples_per_prompt = N`` rewrites/prompt) drives
      the LLM child and the ``"diffusion"`` entry (``DiffusionSamplingParams``,
      ``samples_per_prompt = M`` images/rewrite) drives the diffusion child.
    - ``stage_config["chat"]: dict`` (optional) — forwarded to the LLM
      chat-template stage as a per-request system-instruction override.
    - ``sigmas: Tensor[T+1]`` — engine-pinned; forwarded to the diffusion
      child only.
    - ``request_conditions: Dict[str, Condition]`` — forwarded to the
      diffusion child verbatim. Non-text per-sample primitives (e.g.
      ``"negative_text"``) are NOT forwarded under branching.

    Writes a two-track ``RolloutResp`` with explicit lineage (sample
    counts fan out by the sampling params — see :meth:`generate`):

    - ``tracks["ar"]: RolloutTrack`` — from the LLM (``parent_track=None``,
      ``parent_ids=prompt`` → GRPO groups by prompt): ``segment=TextSegment``,
      ``decoded=Texts`` (rewritten prompts), ``conditions={"prompt": ...}``.
    - ``tracks["diffusion"]: RolloutTrack`` — from the diffusion
      (``parent_track="ar"``, ``parent_ids=rewrite`` → GRPO groups by
      rewrite): ``segment=LatentSegment``, ``decoded=Images``,
      ``conditions={"text": TextEmbedCondition, ...}``.
    """

    def __init__(
        self,
        *,
        diffusion_pipeline: Pipeline,
        llm_pipeline: Pipeline,
        pe_instruction: Optional[str] = None,
        pe_marker: Optional[str] = None,
        pe_max_chars: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.diffusion_pipeline = diffusion_pipeline
        self.llm_pipeline = llm_pipeline
        # PE prompt-rewrite knobs, mirroring the sglang ComposedRolloutEngine
        # (composed/config.py): ``pe_instruction`` is injected as the LLM
        # child's chat ``system_instruction`` so the rewriter actually enhances
        # the prompt; ``pe_marker`` (+ optional ``pe_max_chars``) governs the
        # marker-based extraction of the cleaned rewrite from the LLM output
        # before it conditions the diffusion child. Both default to ``None``,
        # which preserves the prior "forward the bare prompt verbatim" behavior.
        self.pe_instruction = pe_instruction
        self.pe_marker = pe_marker
        self.pe_max_chars = pe_max_chars
        # Surfaces the composed bundle so downstream code (training-side
        # weight policies, eval introspection, ...) can reach
        # pe_pipeline.bundle.{diffusion, llm} without duplicating the
        # child-pipeline reference.
        self.bundle = PEBundle(
            diffusion=diffusion_pipeline.bundle,
            llm=llm_pipeline.bundle,
        )

    # ------------------------------------------------------------------
    # Stage / schedule accessors (used by a trainside rollout engine)
    # ------------------------------------------------------------------

    @property
    def diffusion(self):
        """The trainable diffusion stage (delegates to the diffusion child).

        Lets a trainside rollout engine resolve the diffusion module via
        ``getattr(pe_pipeline, "diffusion").trainable_module()`` —
        ``stage_attrs=["diffusion", "ar"]`` eval-scopes both PE models.
        """
        return self.diffusion_pipeline.diffusion

    @property
    def ar(self):
        """The trainable AR stage (delegates to the LLM child)."""
        return self.llm_pipeline.ar

    def build_schedule_policy(self):
        """σ-schedule policy for the diffusion track (delegates to the diffusion child).

        PE forwards ``req.sigmas`` to the diffusion sub-request unchanged, so
        the parent schedule *is* the diffusion child's. A trainside engine
        calls this to pin sigmas on the parent request before ``generate``;
        the composed ``PEBundle`` has no ``pretrained_path``/``shift`` of its
        own, so we reach through to the diffusion child.
        """
        diff = self.diffusion_pipeline
        builder = getattr(diff, "build_schedule_policy", None)
        if callable(builder):
            return builder()
        from unirl.sde.runtime import FlowMatchSchedulePolicy

        return FlowMatchSchedulePolicy.from_pretrained(
            getattr(diff.bundle, "pretrained_path", None),
            shift=float(diff.shift),
        )

    def generate(self, req: RolloutReq) -> RolloutResp:
        """Run the PE flow with two-level, sampling-param-driven fan-out.

        ``ar.samples_per_prompt = N`` and ``diffusion.samples_per_prompt = M``
        drive the branching::

            P prompts ──make_root_track(N)──▶ P*N rewrites  (root "ar" track)
                      ──fork_track(M)───────▶ P*N*M images   ("diffusion" track)

        The child pipelines are 1:1 (they neither expand nor drop samples),
        so PE replicates the inputs explicitly — the raw prompt repeated N×
        for the LLM, each rewrite repeated M× for the diffusion child — and
        the branch factors land as lineage (``parent_ids`` / ``parent_track``)
        so GRPO groups by prompt on "ar" and by rewrite on "diffusion".
        """
        texts = req.primitives.get("text")
        if not isinstance(texts, Texts):
            raise TypeError(
                "PEPipeline.generate: req.primitives['text'] must be a Texts primitive; "
                f"got {type(texts).__name__ if texts is not None else 'None'}. "
                "The LLM child requires the raw user prompt at primitives['text']."
            )

        ar_params = req.sampling_params.get("ar")
        diff_params = req.sampling_params.get("diffusion")
        n_rewrites = int(ar_params.samples_per_prompt) if ar_params is not None else 1
        n_images = int(diff_params.samples_per_prompt)

        # ── Level 1: P → P*N AR rewrites. Root track grouped by prompt
        # (parent_track=None, parent_ids=prompt). Replicate each raw prompt
        # N× so the 1:1 LLM child emits N independent rewrites per prompt.
        ar_shell = req.make_root_track(track_name="ar", branch=n_rewrites)
        llm_texts = Texts(texts=[t for t in texts.texts for _ in range(n_rewrites)])
        llm_req = self._build_llm_req(
            req, sample_ids=ar_shell.sample_ids, group_ids=ar_shell.parent_ids, texts=llm_texts
        )
        llm_resp = self.llm_pipeline.generate(llm_req)

        # The rewritten prompts live on the LLM track's ``decoded`` field as a
        # single :class:`Texts`. Both Qwen3Pipeline and any future AR LLM
        # following the pipeline contract emit a track named ``"ar"``.
        llm_track = llm_resp.tracks.get("ar")
        rewritten = llm_track.decoded if llm_track is not None else None
        if not isinstance(rewritten, Texts):
            raise RuntimeError(
                "PEPipeline.generate: LLM child returned tracks['ar'].decoded of "
                f"type {type(rewritten).__name__ if rewritten is not None else 'None'}; "
                "expected Texts on tracks['ar'].decoded so the diffusion child can "
                "consume it as primitives['text']."
            )
        if len(rewritten.texts) != len(ar_shell.sample_ids):
            raise RuntimeError(
                f"PEPipeline.generate: LLM child returned {len(rewritten.texts)} rewritten "
                f"text(s) but the AR track expects {len(ar_shell.sample_ids)} (= P*N). The "
                "LLM child must be 1:1 over its (already N-replicated) request."
            )
        ar_track = _track_with_field(ar_shell, "segment", llm_track.segment)
        ar_track = _track_with_field(ar_track, "decoded", rewritten)
        ar_track = _track_with_field(ar_track, "conditions", dict(llm_track.conditions))

        # Optional marker-based PE extraction (mirrors ComposedRolloutEngine,
        # composed/engine.py): keep only the substring after ``pe_marker`` so the
        # diffusion child conditions on the cleaned rewrite instead of the LLM's
        # reasoning preamble; off-format / empty outputs fall back to the original
        # user prompt. Rewrite ``ar_track.decoded`` in place so wandb / logging and
        # the diffusion conditioning see the same cleaned text. ``texts.texts`` are
        # the P original prompts; slot k of the P*N rewrites maps to prompt
        # ``k // n_rewrites``.
        if self.pe_marker:
            cleaned_texts, stats = postprocess_pe_texts(
                rewritten.texts,
                user_prompts=texts.texts,
                samples_per_prompt=n_rewrites,
                marker=self.pe_marker,
                max_chars=self.pe_max_chars,
            )
            if any(stats.values()):
                logger.info(
                    "PEPipeline: PE-extract — marker=%r, %d/%d empty, %d truncated, %d fallback_to_original",
                    self.pe_marker,
                    stats["empty"],
                    len(rewritten.texts),
                    stats["truncated"],
                    stats["fallback"],
                )
            rewritten = Texts(texts=cleaned_texts)
            ar_track = _track_with_field(ar_track, "decoded", rewritten)

        # ── Level 2: P*N → P*N*M images. Fork from "ar" (parent_track="ar",
        # parent_ids=rewrite). Replicate each rewrite M× for the 1:1 diffusion
        # child; the rewritten prompt is swapped into primitives["text"].
        diff_shell = ar_track.fork_track(parent_name="ar", child_name="diffusion", branch=n_images)
        diff_texts = Texts(texts=[t for t in rewritten.texts for _ in range(n_images)])
        diff_req = self._build_diffusion_req(
            req, sample_ids=diff_shell.sample_ids, group_ids=diff_shell.parent_ids, texts=diff_texts
        )
        diff_resp = self.diffusion_pipeline.generate(diff_req)

        diff_inner = diff_resp.tracks.get("image")
        if diff_inner is None:
            raise RuntimeError(
                "PEPipeline.generate: diffusion child returned no 'image' track "
                f"(got {sorted(diff_resp.tracks.keys())})."
            )
        if len(diff_inner.sample_ids) != len(diff_shell.sample_ids):
            raise RuntimeError(
                f"PEPipeline.generate: diffusion child returned {len(diff_inner.sample_ids)} "
                f"sample(s) but the diffusion track expects {len(diff_shell.sample_ids)} "
                "(= P*N*M). The diffusion child must be 1:1 over its (already M-replicated) request."
            )
        diff_track = _track_with_field(diff_shell, "segment", diff_inner.segment)
        diff_track = _track_with_field(diff_track, "decoded", diff_inner.decoded)
        diff_track = _track_with_field(diff_track, "conditions", dict(diff_inner.conditions))
        diff_track = _track_with_field(diff_track, "media_preview", diff_inner.media_preview)

        return RolloutResp(tracks={"ar": ar_track, "diffusion": diff_track})

    # ------------------------------------------------------------------
    # Child-request construction
    # ------------------------------------------------------------------

    def _build_llm_req(
        self,
        req: RolloutReq,
        *,
        sample_ids: list[str],
        group_ids: list[str],
        texts: Texts,
    ) -> RolloutReq:
        """Construct the LLM-side child RolloutReq for the N-replicated set.

        Carries the (N-replicated) prompts and the AR sampling params; drops
        sigmas, request_conditions, non-text primitives, and diffusion params.

        Forwards the parent's ``stage_config["chat"]`` and, when
        ``self.pe_instruction`` is set, injects it as the chat
        ``system_instruction`` (overwriting any inherited value) so the
        rewriter enhances the prompt — matching ComposedRolloutEngine
        (composed/engine.py), which likewise forces ``pe_instruction`` onto the
        AR/chat ``system_instruction`` so generation always uses the recipe's
        PE prompt.
        """
        chat_cfg = dict(req.stage_config.get("chat") or {})
        if self.pe_instruction:
            chat_cfg["system_instruction"] = self.pe_instruction
        stage_config = {"chat": chat_cfg} if chat_cfg else {}
        return RolloutReq(
            sample_ids=list(sample_ids),
            group_ids=list(group_ids),
            primitives={"text": texts},
            request_conditions={},
            sampling_params={"ar": req.sampling_params.get("ar")},
            stage_config=stage_config,
            sigmas=None,
        )

    def _build_diffusion_req(
        self,
        req: RolloutReq,
        *,
        sample_ids: list[str],
        group_ids: list[str],
        texts: Texts,
    ) -> RolloutReq:
        """Construct the diffusion-side child RolloutReq for the M-replicated set.

        Carries the (M-replicated) rewritten prompts as primitives['text'],
        forwards request_conditions + sigmas verbatim, and extracts the
        diffusion sampling params. Non-text per-sample primitives are not
        forwarded (matches ComposedRolloutEngine); SD3's empty-negative
        default applies. Add typed replication here if a recipe needs e.g.
        negative_text under branching.
        """
        return RolloutReq(
            sample_ids=list(sample_ids),
            group_ids=list(group_ids),
            primitives={"text": texts},
            request_conditions=dict(req.request_conditions),
            sampling_params={"diffusion": req.sampling_params.get("diffusion")},
            sigmas=req.sigmas,
        )


__all__ = ["PEPipeline"]
