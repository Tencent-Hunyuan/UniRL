"""SGLang rollout engine.

This package houses the rewrite of ``unirl/samplers/sglang/`` onto the
new :class:`unirl.rollout.engine.base.BaseRolloutEngine` protocol. It
coexists with the legacy package — recipes opt into it via
``cfg.rollout.engine=sglang``.
"""

from unirl.rollout.engine.sglang.config import SGLangEngineConfig
from unirl.rollout.engine.sglang.engine import SGLangRolloutEngine

__all__ = ["SGLangRolloutEngine", "SGLangEngineConfig"]
