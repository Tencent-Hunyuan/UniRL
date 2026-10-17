"""UniRL in-process monkey-patches for stock-upstream FastVideo; pin and contracts in ../README.md."""

from unirl.rollout.engine.fastvideo._patches.hijack import patch_fastvideo
from unirl.rollout.engine.fastvideo._patches.unipc import FastVideoUniPCPlan

__all__ = ["FastVideoUniPCPlan", "patch_fastvideo"]
