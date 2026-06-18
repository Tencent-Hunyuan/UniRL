"""HunyuanImage3Pipeline — RolloutReq → RolloutResp dispatcher.

Per-task generate logic lives in ``modes/<task>.py`` (one file each for
``t2t``, ``i2t``, ``t2i``, ``it2i``, ``t2ti``). This module is a thin
dispatcher:
it instantiates / composes the shared stages (``Bundle``,
``TextEmbedStage``, ``DiffusionStage``, ``ARStage``, ``VAEEncodeStage``,
``VAEDecodeStage``, ``VitEncodeStage``) and routes ``generate(req)`` to
the matching ``modes.<task>.generate`` based on
``req.stage_params["task"]``.

Hydra registers ``model/hunyuan_image3`` against
``HunyuanImage3Pipeline.from_config`` via ``config.py``; that path
remains unchanged across the per-mode split.

Detokenization (``_detokenize_text_segment``) stays on this class
because multiple modes need it. σ schedule construction is no longer
the pipeline's concern — the engine adapter pins ``req.sigmas`` via
:func:`unirl.sde.runtime.ensure_req_sigmas` before invoking
``generate``; modes read ``req.sigmas`` directly.
"""

from __future__ import annotations

from typing import List, Optional

from unirl.models.types.pipeline import Pipeline
from unirl.sde.kernels import CPSSDEStrategy, StepStrategy
from unirl.types.primitives import Texts
from unirl.types.rollout_req import RolloutReq
from unirl.types.rollout_resp import RolloutResp

from .ar import HunyuanImage3ARStage
from .bundle import HunyuanImage3Bundle
from .config import HunyuanImage3PipelineConfig
from .diffusion import (
    HunyuanImage3DiffusionStage,
    HunyuanImage3DiffusionStep,
)
from .text_embed import HunyuanImage3TextEmbedStage
from .vae import HunyuanImage3VAEDecodeStage, HunyuanImage3VAEEncodeStage
from .vit_encode import HunyuanImage3VitEncodeStage


class HunyuanImage3Pipeline(Pipeline):
    """HunyuanImage 3.0 generate pipeline.

    Reads from ``RolloutReq``:

    - ``primitives["text"]: Texts`` — required prompts.
    - ``primitives["negative_text"]`` — rejected for t2i / it2i: the HI3
      tokenizer never consumes negative-prompt text; CFG derives from
      ``guidance_scale > 1.0``.
    - ``primitives["image"]: Images`` — required for i2t / it2i.
    - ``stage_params["task"]: str`` — one of ``{"t2t", "i2t", "t2i",
      "it2i", "t2ti"}``. Defaults to ``"t2i"`` if absent.
    - ``stage_params["bot_task"]: str`` — chat-template flag forwarded to
      ``HunyuanImage3TextEmbedStage.embed_for_gen_image`` (t2i / it2i),
      or the CoT-chain preset for t2ti (default ``"think_recaption"``).
    - ``stage_params["diffusion"]: dict`` — kwargs for
      :class:`HunyuanImage3DiffusionParams` (t2i / it2i).
    - ``stage_params["ar"]: dict`` — kwargs for AR (t2t / i2t / t2ti).

    t2ti (text → CoT text + image) requires
    ``ComposedSamplingParams(ar=..., diffusion=...)`` and returns TWO
    tracks: ``"ar"`` (root, the CoT TextSegment) and ``"image"``
    (``parent_track="ar"``, the LatentSegment). Fan-out
    (``samples_per_prompt``) is NOT honored by t2ti — replication
    belongs to the engine adapter, as with the other HI3 modes.

    Writes to ``RolloutResp``:

    - ``conditions``: per-task — see each ``modes/<task>.py``.
    - ``tracks["ar"].segment: TextSegment`` for AR-mode tasks.
    - ``tracks["image"].segment: LatentSegment`` for diffusion-mode tasks.
    - ``tracks["ar"].decoded: Texts`` (AR-mode) /
      ``tracks["image"].decoded: Images`` (diffusion-mode).
    """

    def __init__(
        self,
        *,
        bundle: HunyuanImage3Bundle,
        text_embed: HunyuanImage3TextEmbedStage,
        diffusion: HunyuanImage3DiffusionStage,
        vae_decode: HunyuanImage3VAEDecodeStage,
        vae_encode: HunyuanImage3VAEEncodeStage,
        ar: HunyuanImage3ARStage,
        vit_encode: HunyuanImage3VitEncodeStage,
        shift: float = 3.0,
    ) -> None:
        super().__init__()
        self.bundle = bundle
        self.text_embed = text_embed
        self.diffusion = diffusion
        self.vae_decode = vae_decode
        self.vae_encode = vae_encode
        self.ar = ar
        self.vit_encode = vit_encode
        self.shift = shift

    @classmethod
    def from_config(
        cls,
        config: HunyuanImage3PipelineConfig,
        *,
        strategy: Optional[StepStrategy] = None,
    ) -> "HunyuanImage3Pipeline":
        """Build the full pipeline from a config.

        ``strategy`` is the SDE step strategy. Defaults to
        :class:`CPSSDEStrategy`; callers running GRPO with a specific
        Flow / Dance / DPM2 strategy should pass an explicit instance
        built from ``cfg.sampling.sde_strategy``.
        """
        return cls._assemble(
            HunyuanImage3Bundle.from_config(config),
            config=config,
            strategy=strategy,
        )

    @classmethod
    def from_meta_config(
        cls,
        config: HunyuanImage3PipelineConfig,
        *,
        strategy: Optional[StepStrategy] = None,
    ) -> "HunyuanImage3Pipeline":
        """Build the pipeline with every parameter on meta-device.

        Used for the 80B path — no weight memory allocated anywhere.
        Caller materializes via :meth:`HunyuanImage3Bundle.materialize`
        (which covers the FSDP-wrapped decoder + wrapper-level heads +
        opt-in vae / vit) after constructing the FSDPPolicy that wraps
        the diffusion stage.
        """
        return cls._assemble(
            HunyuanImage3Bundle.from_meta_config(config),
            config=config,
            strategy=strategy,
        )

    @classmethod
    def from_bundle(
        cls,
        bundle: HunyuanImage3Bundle,
        *,
        config: HunyuanImage3PipelineConfig,
        strategy: Optional[StepStrategy] = None,
    ) -> "HunyuanImage3Pipeline":
        """Assemble the pipeline from an ALREADY-built (possibly shared) bundle.

        ``from_config`` / ``from_meta_config`` each build their own bundle; this
        instead takes a bundle the caller already constructed. Trainers build ONE
        bundle and share it across the FSDP backend and this pipeline, so replay
        reads the trained weights — see :class:`~unirl.trainer.unified_model.`
        ``UnifiedModelTrainer``, whose ``pipeline_cfg`` targets this with ``bundle=`` auto-
        injected from the shared sibling.
        """
        return cls._assemble(bundle, config=config, strategy=strategy)

    @classmethod
    def _assemble(
        cls,
        bundle: HunyuanImage3Bundle,
        *,
        config: HunyuanImage3PipelineConfig,
        strategy: Optional[StepStrategy],
    ) -> "HunyuanImage3Pipeline":
        text_embed = HunyuanImage3TextEmbedStage(bundle)
        step = HunyuanImage3DiffusionStep()
        diffusion = HunyuanImage3DiffusionStage(
            model=bundle,
            step=step,
            strategy=strategy if strategy is not None else CPSSDEStrategy(),
            autocast_precision=config.autocast_precision,
            trajectory_precision=config.trajectory_precision,
            logprob_precision=config.logprob_precision,
        )
        vae_decode = HunyuanImage3VAEDecodeStage(bundle)
        vae_encode = HunyuanImage3VAEEncodeStage(bundle)
        ar = HunyuanImage3ARStage(model=bundle)
        vit_encode = HunyuanImage3VitEncodeStage(bundle)
        return cls(
            bundle=bundle,
            text_embed=text_embed,
            diffusion=diffusion,
            vae_decode=vae_decode,
            vae_encode=vae_encode,
            ar=ar,
            vit_encode=vit_encode,
            shift=float(config.shift),
        )

    def generate(self, req: RolloutReq) -> RolloutResp:
        """Dispatch to the per-task generate function in ``modes/``.

        ``stage_params["task"]`` selects the topology. Lazy-imports the
        modes package to avoid the circular ``modes -> pipeline`` ref
        (mode files type-annotate ``pipeline: "HunyuanImage3Pipeline"``).
        """
        from .modes import i2t, it2i, t2i, t2t, t2ti

        task = req.stage_config.get("task", "t2i")
        if task == "t2t":
            return t2t.generate(self, req)
        if task == "i2t":
            return i2t.generate(self, req)
        if task == "t2i":
            return t2i.generate(self, req)
        if task == "it2i":
            return it2i.generate(self, req)
        if task == "t2ti":
            return t2ti.generate(self, req)
        raise ValueError(
            f"HunyuanImage3Pipeline.generate: unknown task={task!r}; "
            f"expected one of 't2t', 'i2t', 't2i', 'it2i', 't2ti'."
        )

    # ------------------------------------------------------------------
    # Helpers shared by multiple modes.
    # ------------------------------------------------------------------

    def _detokenize_text_segment(self, text_seg, *, skip_special_tokens: bool = True) -> Texts:
        """Detokenize a varlen ``TextSegment`` back into a ``Texts`` primitive.

        Reads ``text_seg.tokens`` + ``text_seg.cu_seqlens`` to slice each
        sample's tokens, runs ``self.bundle.tokenizer.decode`` per sample,
        and packages the results into ``Texts``. Returns empty strings
        when the bundle has no tokenizer (used by fake-bundle tests).

        ``skip_special_tokens=False`` keeps control markers like
        ``</think>`` / ``</recaption>`` in the decoded text — t2ti's
        bridge needs them to truncate and re-feed the CoT.

        Shape contract:
            text_seg.tokens     : packed varlen [sum_lengths] long
            text_seg.cu_seqlens : [B+1] long
            returned Texts.texts: list[str] of length B
        """
        tokenizer = self.bundle.tokenizer
        if text_seg.tokens is None or text_seg.cu_seqlens is None:
            return Texts(texts=[])
        n_segs = int(text_seg.cu_seqlens.shape[0]) - 1
        if tokenizer is None:
            return Texts(texts=["" for _ in range(n_segs)])
        out: List[str] = []
        for k in range(n_segs):
            a = int(text_seg.cu_seqlens[k].item())
            b = int(text_seg.cu_seqlens[k + 1].item())
            ids = text_seg.tokens[a:b].tolist()
            # clean_up_tokenization_spaces=False: the HunyuanImage3 BPE tokenizer
            # warns that the WordPiece-oriented cleanup is destructive for BPE
            # (inserts spaces between characters) — disable it for coherent text.
            out.append(
                tokenizer.decode(ids, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=False)
            )
        return Texts(texts=out)


__all__ = ["HunyuanImage3Pipeline"]
