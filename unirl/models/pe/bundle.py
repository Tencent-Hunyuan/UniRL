"""PEBundle — composed weights container for Prompt Enhancement."""

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
