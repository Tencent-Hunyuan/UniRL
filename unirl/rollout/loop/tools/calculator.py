"""CalculatorTool — a safe arithmetic tool (LIN-492).

The first concrete :class:`~unirl.rollout.loop.tools.tool.Tool`: evaluates an arithmetic
expression with no infrastructure (no sandbox, no network). Evaluation is a hand-rolled AST walk
over a numeric whitelist — never :func:`eval` — so a hallucinated ``__import__(...)``, name, or
attribute access is rejected rather than executed.
"""

from __future__ import annotations

import ast
import operator
from typing import Any, Dict

from unirl.rollout.loop.tools.tool import Tool

# Binary / unary operators the calculator allows. Anything else (names, calls, attributes,
# comparisons, subscripts, ...) is rejected by ``_eval`` below.
_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval(node: ast.AST) -> float:
    """Recursively evaluate a whitelisted arithmetic AST node."""
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        # bool is an int subclass — exclude it so ``True * 3`` isn't silently arithmetic.
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"non-numeric constant: {node.value!r}")
        return node.value
    if isinstance(node, ast.BinOp):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"operator not allowed: {type(node.op).__name__}")
        return op(_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _UNARYOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unary operator not allowed: {type(node.op).__name__}")
        return op(_eval(node.operand))
    raise ValueError(f"expression element not allowed: {type(node).__name__}")


class CalculatorTool(Tool):
    """Evaluate a basic arithmetic expression (``+ - * / // % **``, parentheses, unary ±)."""

    name = "calculator"

    def json_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Evaluate a basic arithmetic expression and return the numeric result. "
                    "Use this for any non-trivial arithmetic instead of computing it yourself."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "The arithmetic expression to evaluate, e.g. '1234 * 5678'.",
                        }
                    },
                    "required": ["expression"],
                },
            },
        }

    def execute(self, arguments: Dict[str, Any]) -> str:
        expression = arguments.get("expression")
        if not isinstance(expression, str) or not expression.strip():
            raise ValueError("calculator requires a non-empty 'expression' string argument")
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"could not parse expression {expression!r}: {exc.msg}") from None
        result = _eval(tree)
        # Render integral results without a trailing ``.0`` (what a human/model expects).
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return str(result)


__all__ = ["CalculatorTool"]
