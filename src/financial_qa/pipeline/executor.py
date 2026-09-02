"""AST-validated arithmetic expression executor.

The LLM emits a Python arithmetic expression as a string.  This executor
parses it with the ``ast`` module, walks every node to verify only safe
arithmetic operations are present, and then evaluates the expression using
only Python's built-in number types — no ``eval()``, no ``exec()``.

This design eliminates hallucinated arithmetic as a failure class: the model
never computes the result itself; it only declares *how* to compute it.
"""

import ast
import operator
from typing import Any

_ALLOWED_NODE_TYPES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,
    # Operators
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.Pow,
    ast.FloorDiv,
    ast.USub,
    ast.UAdd,
)

_BINARY_OPS: dict[type, Any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.FloorDiv: operator.floordiv,
}

_UNARY_OPS: dict[type, Any] = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


class ExecutionError(ValueError):
    """Raised when an expression cannot be safely evaluated."""


class Executor:
    """Safely evaluates arithmetic expressions via AST inspection.

    Example:
        >>> Executor().run("(14.5 - 12.0) / 12.0 * 100")
        20.833333333333332
    """

    def run(self, expression: str) -> float:
        """Parse, validate, and evaluate an arithmetic expression.

        Args:
            expression: A Python arithmetic expression string produced by the LLM.

        Returns:
            The computed result as a float.

        Raises:
            ExecutionError: If the expression is malformed, contains forbidden
                constructs, or results in a arithmetic error (e.g. division by zero).
        """
        if not expression or not expression.strip():
            raise ExecutionError("expression cannot be empty")

        try:
            tree = ast.parse(expression.strip(), mode="eval")
        except SyntaxError as exc:
            raise ExecutionError(f"unparseable expression: {exc}") from exc

        self._validate(tree)

        try:
            result = self._evaluate(tree.body)
        except ZeroDivisionError as exc:
            raise ExecutionError("division by zero") from exc

        return float(result)

    def _validate(self, tree: ast.AST) -> None:
        """Walk the AST and reject any node type not in the allow-list."""
        for node in ast.walk(tree):
            if not isinstance(node, _ALLOWED_NODE_TYPES):
                raise ExecutionError(
                    f"forbidden AST node '{type(node).__name__}' in expression"
                )

    def _evaluate(self, node: ast.expr) -> float:
        """Recursively evaluate a validated AST node."""
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise ExecutionError(
                    f"forbidden constant type '{type(node.value).__name__}'"
                )
            return float(node.value)

        if isinstance(node, ast.BinOp):
            left = self._evaluate(node.left)
            right = self._evaluate(node.right)
            op_fn = _BINARY_OPS.get(type(node.op))
            if op_fn is None:
                raise ExecutionError(f"unsupported operator '{type(node.op).__name__}'")
            return float(op_fn(left, right))

        if isinstance(node, ast.UnaryOp):
            operand = self._evaluate(node.operand)
            op_fn = _UNARY_OPS.get(type(node.op))
            if op_fn is None:
                raise ExecutionError(f"unsupported unary operator '{type(node.op).__name__}'")
            return float(op_fn(operand))

        raise ExecutionError(f"unexpected node '{type(node).__name__}'")
