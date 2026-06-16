"""UniRL in-process monkey-patches for stock-upstream sglang diffusion.

Re-hosts the ``sglang-drl`` fork's RL additions as runtime patches (LIN-365) so
UniRL can depend on stock upstream sglang instead of a hard fork. Install
via ``SglangDiffusionHijack.hijack()`` -- see ``hijack.py``.
"""

from unirl.rollout.engine.sglang_diffusion._patches.hijack import SglangDiffusionHijack

__all__ = ["SglangDiffusionHijack"]
