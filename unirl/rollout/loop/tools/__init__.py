"""Tools a :class:`~unirl.rollout.loop.tool_environment.ToolEnvironment` dispatches to (LIN-492).

One class per module: the interfaces :class:`~unirl.rollout.loop.tools.tool.Tool` (stateless) and
:class:`~unirl.rollout.loop.tools.tool.StatefulTool` (session-scoped), the reference
:class:`~unirl.rollout.loop.tools.calculator.CalculatorTool`, the persistent-REPL
:class:`~unirl.rollout.loop.tools.sandbox.SandboxTool`, and the deep-research web tools
:class:`~unirl.rollout.loop.tools.search.SearchTool` / :class:`~unirl.rollout.loop.tools.visit.VisitTool`.
"""

from unirl.rollout.loop.tools.calculator import CalculatorTool
from unirl.rollout.loop.tools.sandbox import SandboxTool
from unirl.rollout.loop.tools.search import SearchTool
from unirl.rollout.loop.tools.tool import StatefulTool, Tool
from unirl.rollout.loop.tools.visit import VisitTool

__all__ = ["Tool", "StatefulTool", "CalculatorTool", "SandboxTool", "SearchTool", "VisitTool"]
