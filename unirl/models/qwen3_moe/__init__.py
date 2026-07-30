"""Qwen3-MoE model package for UniRL.

A VeOmni-patched Qwen3-MoE causal-LM bundle that carries ``get_parallel_plan``
(``Shard(0)`` on stacked expert weights) and a fused MoE op, so it can train
under :class:`unirl.train.backend.veomni.VeOmniBackend` with expert parallelism
(``fsdp_cfg.ep_size > 1``). Reuses the dense Qwen3 AR stage / conditions
(:mod:`unirl.models.qwen3.ar`) — the replay forward is architecture-agnostic.
"""

from __future__ import annotations

from .bundle import Qwen3MoeBundle

__all__ = ["Qwen3MoeBundle"]
