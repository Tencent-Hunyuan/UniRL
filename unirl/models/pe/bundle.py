"""PEBundle — composed weights container for Prompt Enhancement.

Holds two child :class:`Bundle` instances side-by-side:

- ``diffusion`` — a diffusion bundle (e.g. :class:`SD3Bundle`) that owns
  the transformer + VAE + text encoders for image generation.
- ``llm`` — an autoregressive LM bundle (e.g. :class:`Qwen3Bundle`) that
  owns the causal-LM transformer + tokenizer for prompt rewriting.

Pure container. No ``from_config`` constructor: PE is loaded via
:class:`PEPipeline`, which constructs both child pipelines (each of
which loads its own bundle via the child's ``Bundle.from_config``) and
wires their bundles together. The composed bundle is always reachable
as ``pe_pipeline.bundle.{diffusion,llm}``.

Satisfies the :class:`Bundle` Protocol
(:mod:`unirl.models.types.bundle`) trivially — the Protocol is
empty by design.
"""

from __future__ import annotations

from unirl.models.types.bundle import Bundle


class PEBundle(Bundle):
    """PE bundle: a diffusion ``Bundle`` + an AR LLM ``Bundle``."""

    def __init__(
        self,
        *,
        diffusion: Bundle,
        llm: Bundle,
    ) -> None:
        super().__init__()
        self.diffusion = diffusion
        self.llm = llm


__all__ = ["PEBundle"]
