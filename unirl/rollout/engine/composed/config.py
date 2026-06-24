"""Composed rollout-engine configuration.

Registered under ``rollout/engine: composed_pe`` with ``_target_`` pointing at
:class:`ComposedRolloutEngine`. The composed engine holds two child engines
(``ar`` and ``diffusion``) and orchestrates the prompt-enhancement (PE)
serial flow internally — it is a peer of the existing ``sglang``,
``sglang``, ``vllm_omni``, and ``trainside`` engines as far as the
actor / pipeline layer is concerned.

Each child (``ar`` / ``diffusion``) is a rollout engine config carrying its
own ``_target_``; the worker walker constructs it from the recipe and
``ComposedRolloutEngine`` builds the engine via ``config.<child>.make_engine``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from unirl.rollout.engine.base import BaseEngineConfig


@dataclass
class ComposedRolloutEngineConfig(BaseEngineConfig):
    """Two-stage prompt-enhancement (PE) composed rollout engine."""

    ar: Any  # BaseEngineConfig (kept Any: built from its own _target_)
    diffusion: Any  # BaseEngineConfig (kept Any: built from its own _target_)

    sleep_diffusion_on_start: bool = True

    # System instruction injected into the AR child's ``stage_config`` so
    # the LLM rewrites the user's prompt (PE = prompt enhancement). ``None``
    # forwards the bare user prompt to AR.
    pe_instruction: Optional[str] = None

    # If set (e.g. ``"Revised Prompt:"``), only the suffix after the LAST
    # occurrence of the marker is forwarded to diffusion; off-format
    # outputs fall back to the original user prompt.
    pe_marker: Optional[str] = None

    # Optional char cap applied AFTER marker extraction.
    pe_max_chars: Optional[int] = None


__all__ = ["ComposedRolloutEngineConfig"]
