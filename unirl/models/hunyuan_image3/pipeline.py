"""HunyuanImage3Pipeline — ``Sample → Sample`` dispatcher."""

from __future__ import annotations

from typing import Any, List, Optional

from unirl.models.types.pipeline import Pipeline
from unirl.sde.kernels import CPSSDEStrategy, StepStrategy
from unirl.types.primitives import Texts
from unirl.types.sample import Sample

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
    """HunyuanImage 3.0 generate pipeline: ``Sample → Sample``."""

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
        """Build the full pipeline from a config."""
        return cls._assemble(
            HunyuanImage3Bundle.from_config(config),
            config=config,
            strategy=strategy,
        )

    @classmethod
    def latent_shape(cls, *, model_config: Any, sampling_spec: Any) -> tuple:
        """Per-sample latent shape ``(C, H_lat, W_lat)`` for driver-side noise"""
        height = int(sampling_spec.height)
        width = int(sampling_spec.width)
        if height <= 0 or width <= 0 or height % 16 or width % 16:
            raise NotImplementedError(
                f"HunyuanImage3Pipeline.latent_shape: {height}x{width} is not a multiple of the "
                "16x VAE factor; opting out of the driver x_T recipe (engine RNG fallback)."
            )
        return (32, height // 16, width // 16)

    @classmethod
    def from_meta_config(
        cls,
        config: HunyuanImage3PipelineConfig,
        *,
        strategy: Optional[StepStrategy] = None,
    ) -> "HunyuanImage3Pipeline":
        """Build the pipeline with every parameter on meta-device."""
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
        """Assemble the pipeline from an ALREADY-built (possibly shared) bundle."""
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
            diffuse_kv_cache=bool(config.diffuse_kv_cache),
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

    def generate(self, sample: Sample) -> Sample:
        """Dispatch to the per-task generate function in ``modes/``."""
        from .modes import i2t, it2i, t2i, t2t, t2ti

        task = (sample.parts[0].control or {}).get("task", "t2i")
        if task == "t2t":
            return t2t.generate(self, sample)
        if task == "i2t":
            return i2t.generate(self, sample)
        if task == "t2i":
            return t2i.generate(self, sample)
        if task == "it2i":
            return it2i.generate(self, sample)
        if task == "t2ti":
            return t2ti.generate(self, sample)
        raise ValueError(
            f"HunyuanImage3Pipeline.generate: unknown task={task!r}; "
            f"expected one of 't2t', 'i2t', 't2i', 'it2i', 't2ti'."
        )

    def _detokenize_text_segment(self, text_seg, *, skip_special_tokens: bool = True) -> Texts:
        """Detokenize a varlen ``TextSegment`` back into a ``Texts`` primitive."""
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
            out.append(
                tokenizer.decode(ids, skip_special_tokens=skip_special_tokens, clean_up_tokenization_spaces=False)
            )
        return Texts(texts=out)


__all__ = ["HunyuanImage3Pipeline"]
