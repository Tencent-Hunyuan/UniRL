"""Driver-side agent-loop package (LIN-492). See ``unirl/rollout/loop/README.md``.

One class per module: :class:`~unirl.rollout.loop.engine_port.RolloutEnginePort`,
:class:`~unirl.rollout.loop.environment.Environment`,
:class:`~unirl.rollout.loop.agent_loop.AgentLoop`, and the first concrete environment
:class:`~unirl.rollout.loop.tool_environment.ToolEnvironment` (with its
:mod:`~unirl.rollout.loop.tools`).
"""

from unirl.rollout.loop.agent_loop import AgentLoop
from unirl.rollout.loop.engine_port import RolloutEnginePort
from unirl.rollout.loop.environment import Environment
from unirl.rollout.loop.tool_environment import ToolEnvironment, parse_tool_call
from unirl.rollout.loop.tools import CalculatorTool, StatefulTool, Tool

__all__ = [
    "AgentLoop",
    "Environment",
    "RolloutEnginePort",
    "ToolEnvironment",
    "parse_tool_call",
    "Tool",
    "StatefulTool",
    "CalculatorTool",
]
