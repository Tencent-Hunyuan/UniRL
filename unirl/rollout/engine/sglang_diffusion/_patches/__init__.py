"""UniRL in-process monkey-patches for stock-upstream sglang diffusion."""

from unirl.rollout.engine.sglang_diffusion._patches.hijack import SglangDiffusionHijack

__all__ = ["SglangDiffusionHijack"]
