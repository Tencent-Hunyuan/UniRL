"""UniRL runtime patches for FastVideo's RL rollout seams."""

from unirl.rollout.engine.fastvideo._patches.hijack import FastVideoHijack

__all__ = ["FastVideoHijack"]
