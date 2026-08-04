"""Environments and tools for agentic rollout (LIN-492). See ``unirl/rollout/env/README.md``.

:class:`~unirl.rollout.env.protocol.Environment` is the world side of a turn;
:class:`~unirl.rollout.env.tool_environment.ToolEnvironment` (with its
:mod:`~unirl.rollout.env.tools`) is the first concrete environment. The loop that
drives an environment lives in :mod:`unirl.rollout.harness.tool_agent` (worker-side,
hosted by :class:`~unirl.rollout.engine.agentic.engine.AgenticRolloutEngine`).
"""

from unirl.rollout.env.protocol import Environment
from unirl.rollout.env.tool_environment import ToolEnvironment, parse_tool_call
from unirl.rollout.env.tools import CalculatorTool, StatefulTool, Tool

__all__ = [
    "Environment",
    "ToolEnvironment",
    "parse_tool_call",
    "Tool",
    "StatefulTool",
    "CalculatorTool",
]
